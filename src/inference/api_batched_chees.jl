# ChEES-HMC (issue #161). See docs/chees-hmc.md for the authoritative spec of the
# whole #161 effort.
#
# Increment 1 (merged) is the scaffold: Halton-jittered fixed-length HMC with a
# SHARED (ensemble) dual-averaging step size plus a pooled diagonal mass.
#
# Increment 2 (this file) adds the cross-chain ChEES trajectory-length adaptation,
# active during warmup, matching `blackjax.adaptation.chees_adaptation` (see the
# per-function comments below for the exact source lines confirmed). A real-valued
# trajectory TIME `T` replaces the fixed step count: each iteration integrates
# `L_iter = max(1, ceil(jitter_val * T / ε))` leapfrog steps, and during warmup `T`
# is driven by Adam gradient ASCENT on the ChEES criterion (whitened by the
# diagonal mass), with the step size dual-averaged to the HARMONIC mean of the
# chains' acceptance probabilities. After warmup `T` is frozen; jitter continues.
#
# It reuses the shared-adaptation building blocks of `batched_hmc`
# (`per_chain_adaptation=false`): `WarmupDriver` dual averaging + pooled diagonal
# mass, `_batched_leapfrog!`, `_batched_hamiltonian!`/`_hamiltonian`,
# `_batched_acceptance_probability!`, and the #162 init retry loop. NUTS/HMC are
# untouched.

# --- ChEES constants (BlackJAX chees_adaptation.py) ------------------------------
# LOG_UPDATE_CLIP = 0.35 (blackjax line 23: "Clip the final log-space update ...
# ~log(2)/2 ~= 0.35"); EPS_FLOAT = 1e-20 (line 25).
const _CHEES_LOG_UPDATE_CLIP = 0.35
const _CHEES_EPS = 1e-20

# Base-2 van der Corput (Halton) radical inverse. Deterministic, consumes no RNG.
# `_halton_base2(1), _halton_base2(2), ...` = 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
function _halton_base2(index::Integer)
    index >= 1 || throw(ArgumentError("Halton index must be >= 1, got $index"))
    result = 0.0
    fraction = 0.5
    remaining = index
    while remaining > 0
        result += fraction * (remaining & 1)
        remaining >>= 1
        fraction *= 0.5
    end
    return result
end

# Per-iteration jitter value in [1 - jitter_amount, 1). Matches BlackJAX
#   jitter_gn(i) = halton_sequence(i) * jitter_amount + (1 - jitter_amount)
# (chees_adaptation.py lines 763-765). `iteration` (1-based) indexes the Halton
# sequence, so the jitter schedule is deterministic and independent of the RNG.
# At `jitter_amount == 0` the value is identically 1.
_chees_jitter_value(jitter_amount::Real, iteration::Integer) =
    _halton_base2(iteration) * Float64(jitter_amount) + (1.0 - Float64(jitter_amount))

# Per-iteration jittered leapfrog step count. Matches BlackJAX
#   integration_steps_fn = ceil(jitter_gn(i) * (trajectory_length / step_size))
# (chees_adaptation.py lines 767-771 with num_leapfrog_steps = T/ε from line 862),
# clamped to a minimum of one integrator step. This REPLACES the increment-1
# scaffold's `floor(num_leapfrog_steps * (1 + jitter_amount * h))`: the trajectory
# length is now the adapted TIME `T`, and the step count is `ceil(jitter_val*T/ε)`.
#
# The raw value is snapped to a nearby integer before `ceil` so float round-off in
# `T/ε` cannot flip the count: with `T = num_leapfrog_steps * ε` the division may
# land at e.g. 12.0000000002, which a bare `ceil` would round up to 13. Snapping
# makes the `jitter_amount = 0 => L = ceil(T/ε) = num_leapfrog_steps` invariant exact.
function _chees_leapfrog_steps_from_jitter(jitter_value::Real, trajectory_length::Real, step_size::Real)
    raw = Float64(jitter_value) * Float64(trajectory_length) / Float64(step_size)
    nearest = round(raw)
    steps = abs(raw - nearest) <= 1e-9 * max(1.0, abs(raw)) ? Int(nearest) : ceil(Int, raw)
    return max(1, steps)
end

# Convenience wrapper computing both the jitter value and the step count for a
# given iteration. `jitter_amount = 0 => jitter_val = 1 => L = ceil(T/ε)`, so at
# the initial `T = num_leapfrog_steps * ε` this is exactly `num_leapfrog_steps`.
function _chees_jittered_leapfrog_steps(
    trajectory_length::Real,
    step_size::Real,
    jitter_amount::Real,
    iteration::Integer,
)
    return _chees_leapfrog_steps_from_jitter(
        _chees_jitter_value(jitter_amount, iteration),
        trajectory_length,
        step_size,
    )
end

# Self-contained scalar Adam optimizer on `log T`, mirroring the `optax.adam`
# instance BlackJAX's ChEES tests/examples pass in
# (`optax.adam(learning_rate=0.5, b1=0, b2=0.95)`; eps defaults to 1e-8). BlackJAX
# leaves `optim` to the caller (chees_adaptation.py line 741); these are the
# hyperparameters its own ChEES suite uses. `_chees_adam_ascend!` returns the
# ASCENT update `+lr * mhat / (sqrt(vhat) + eps)` (see `_chees_trajectory_update!`
# for the sign discussion).
mutable struct _ChEESAdamState
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    learning_rate::Float64
    m::Float64
    v::Float64
    t::Int
end

_ChEESAdamState(learning_rate::Real, beta1::Real, beta2::Real, epsilon::Real) =
    _ChEESAdamState(Float64(beta1), Float64(beta2), Float64(epsilon), Float64(learning_rate), 0.0, 0.0, 0)

function _chees_adam_ascend!(state::_ChEESAdamState, gradient::Float64)
    state.t += 1
    state.m = state.beta1 * state.m + (1.0 - state.beta1) * gradient
    state.v = state.beta2 * state.v + (1.0 - state.beta2) * gradient * gradient
    m_hat = state.m / (1.0 - state.beta1^state.t)
    v_hat = state.v / (1.0 - state.beta2^state.t)
    return state.learning_rate * m_hat / (sqrt(v_hat) + state.epsilon)
end

# Cross-chain ChEES trajectory-length adaptation state (the increment-2 core).
# `trajectory_length` is the TIME `T` consumed by proposals; `log_trajectory_length_ma`
# is the moving average that `T = exp(.)` tracks; `step` is the 1-based moving-average
# counter (BlackJAX inits `step = 1`, chees_adaptation.py line 522).
mutable struct _ChEESTrajectoryState
    trajectory_length::Float64
    log_trajectory_length_ma::Float64
    adam::_ChEESAdamState
    step::Int
    decay_rate::Float64
    max_leapfrog_steps::Int
end

# True iff every entry of column `column` is finite (BlackJAX's weighted mean masks
# non-finite chain rows; chees_adaptation.py lines 241-242).
function _chees_column_finite(matrix::AbstractMatrix, column::Integer)
    @inbounds for row in axes(matrix, 1)
        isfinite(matrix[row, column]) || return false
    end
    return true
end

# One cross-chain ChEES trajectory-length update (BlackJAX `compute_parameters`,
# chees_adaptation.py lines 376-511). All quantities are P x C (params x chains).
#
# Whitening (Σ = `inverse_mass_matrix`, the diagonal inverse mass), lines 450-458:
#   inv_sqrt = Σ^{-1/2}
#   Δx'_w = (proposal - proposals_mean) .* inv_sqrt          # whitened endpoint offset
#   Δx_w  = (initial  - initials_mean ) .* inv_sqrt          # whitened start offset
#   v'_w  = p' .* Σ .* inv_sqrt   (= p' .* Σ^{1/2})          # whitened endpoint velocity
# proposals_mean is the ACCEPTANCE-weighted, divergence-masked mean (lines 376-379);
# initials_mean is the plain mean over chains (lines 384-386).
#
# Per-chain gradient (lines 460-466):
#   g_c = jitter_val * T * ( <Δx'_w,Δx'_w> - <Δx_w,Δx_w> ) * <Δx'_w, v'_w>
# Cross-chain reduction (lines 468-471):
#   trajectory_gradient = Σ_{~div} accept_c * g_c / Σ_{~div} (accept_c + EPS)
#
# SIGN: `trajectory_gradient ≈ +d(ChEES)/d(log T)` and the criterion is MAXIMIZED
# (docstring line 259: "Maximizing the ... ChEES criterion"), so we ASCEND:
# `log T += clip(+adam_update, ±0.35)`. BlackJAX passes the raw gradient to
# `optim.update` and applies via `optax.apply_updates` (lines 474-481); with the
# `optax.adam` it uses that composition DESCENDS, so to realize the documented
# maximization we ascend directly. See docs/chees-hmc.md for this deviation note.
function _chees_trajectory_update!(
    traj::_ChEESTrajectoryState,
    initials::AbstractMatrix,
    proposals::AbstractMatrix,
    momenta::AbstractMatrix,
    accept_prob::AbstractVector,
    divergent::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    jitter_value::Float64,
    step_size_for_clip::Float64,
)
    num_params = size(initials, 1)
    num_chains = size(initials, 2)

    proposals_mean = zeros(num_params)
    initials_mean = zeros(num_params)
    weight_sum = 0.0
    initials_count = 0
    @inbounds for chain_index = 1:num_chains
        if _chees_column_finite(initials, chain_index)
            for row = 1:num_params
                initials_mean[row] += initials[row, chain_index]
            end
            initials_count += 1
        end
        if !divergent[chain_index] && _chees_column_finite(proposals, chain_index)
            weight = accept_prob[chain_index]
            weight_sum += weight
            for row = 1:num_params
                proposals_mean[row] += weight * proposals[row, chain_index]
            end
        end
    end
    if initials_count > 0
        @inbounds for row = 1:num_params
            initials_mean[row] /= initials_count
        end
    end
    inverse_weight_sum = 1.0 / (weight_sum + _CHEES_EPS)
    @inbounds for row = 1:num_params
        proposals_mean[row] *= inverse_weight_sum
    end

    gradient_numerator = 0.0
    acceptance_denominator = 0.0
    @inbounds for chain_index = 1:num_chains
        divergent[chain_index] && continue
        (_chees_column_finite(proposals, chain_index) && _chees_column_finite(momenta, chain_index)) ||
            continue
        p_accept = accept_prob[chain_index]
        acceptance_denominator += p_accept + _CHEES_EPS
        squared_endpoint = 0.0
        squared_start = 0.0
        endpoint_velocity = 0.0
        for row = 1:num_params
            sigma = inverse_mass_matrix[row]
            inv_sqrt = 1.0 / sqrt(sigma)
            whitened_endpoint = (proposals[row, chain_index] - proposals_mean[row]) * inv_sqrt
            whitened_start = (initials[row, chain_index] - initials_mean[row]) * inv_sqrt
            whitened_velocity = momenta[row, chain_index] * sigma * inv_sqrt
            squared_endpoint += whitened_endpoint * whitened_endpoint
            squared_start += whitened_start * whitened_start
            endpoint_velocity += whitened_endpoint * whitened_velocity
        end
        gradient_numerator += p_accept * (squared_endpoint - squared_start) * endpoint_velocity
    end

    return _chees_apply_trajectory_gradient!(
        traj, gradient_numerator, acceptance_denominator, jitter_value, step_size_for_clip,
    )
end

# The scalar ChEES state machine: turn the reduced cross-chain estimator terms
# (`gradient_numerator` = sum_c p_accept*(||dxp||^2 - ||dx||^2)*<dxp,vp>,
# `acceptance_denominator` = sum_c (p_accept + eps)) into the next trajectory
# length via the Adam ascent, log-T clip, and moving average. Shared verbatim by
# the CPU estimator above and the device path (issue #220), which produces the
# SAME two scalars from on-device reductions instead of P x C host matrices, so
# the Adam/log-T/clip/moving-average logic is never forked.
function _chees_apply_trajectory_gradient!(
    traj::_ChEESTrajectoryState,
    gradient_numerator::Float64,
    acceptance_denominator::Float64,
    jitter_value::Float64,
    step_size_for_clip::Float64,
)
    trajectory_gradient =
        acceptance_denominator > 0.0 ?
        jitter_value * traj.trajectory_length * gradient_numerator / acceptance_denominator : 0.0

    log_trajectory_length = log(traj.trajectory_length)
    saved_m, saved_v, saved_t = traj.adam.m, traj.adam.v, traj.adam.t
    raw_update = _chees_adam_ascend!(traj.adam, trajectory_gradient)
    update = clamp(raw_update, -_CHEES_LOG_UPDATE_CLIP, _CHEES_LOG_UPDATE_CLIP)
    log_trajectory_length_new = log_trajectory_length + update
    if !isfinite(log_trajectory_length_new)
        # BlackJAX reverts BOTH log_trajectory_length and the optimizer state on a
        # non-finite update (chees_adaptation.py lines 482-489).
        log_trajectory_length_new = log_trajectory_length
        traj.adam.m, traj.adam.v, traj.adam.t = saved_m, saved_v, saved_t
    end

    update_weight = Float64(traj.step)^(-traj.decay_rate)
    traj.log_trajectory_length_ma =
        (1.0 - update_weight) * traj.log_trajectory_length_ma + update_weight * log_trajectory_length_new
    new_trajectory_length = exp(traj.log_trajectory_length_ma)
    # Clip to [ε, max_leapfrog_steps * ε] (chees_adaptation.py lines 497-501).
    new_trajectory_length =
        clamp(new_trajectory_length, step_size_for_clip, traj.max_leapfrog_steps * step_size_for_clip)
    traj.trajectory_length = new_trajectory_length
    traj.step += 1
    return new_trajectory_length
end

# Harmonic mean of the chains' acceptance probabilities over non-divergent chains,
# the dual-averaging target statistic ChEES uses (BlackJAX chees_adaptation.py
# lines 358-363: `harmonic_mean = 1 / mean(1/accept, where=~div)`, replaced by 0
# when non-finite). `_update_step_size!` forms `target - accept_statistic`
# internally, matching `da_update(da_state, target - harmonic_mean)`.
function _chees_harmonic_mean_accept(accept_prob::AbstractVector, divergent::AbstractVector)
    total = 0.0
    count = 0
    @inbounds for index in eachindex(accept_prob, divergent)
        divergent[index] && continue
        total += 1.0 / accept_prob[index]
        count += 1
    end
    count == 0 && return 0.0
    harmonic_mean = count / total
    return isfinite(harmonic_mean) ? harmonic_mean : 0.0
end

"""
    batched_chees(model, args=(), constraints=choicemap(); num_chains, num_samples, kwargs...) -> HMCChains

Run ChEES-HMC (Hoffman, Radul & Sountsov, AISTATS 2021): many parallel chains of
fixed-length **jittered** HMC whose trajectory length is adapted from
cross-chain statistics. Every chain does identical work each step (no
control-flow divergence, no per-leaf host sync), so it maps to a handful of
kernels with one sync per iteration — the GPU-native route. Returns an
[`HMCChains`](@ref).

Key keyword arguments:

- `num_chains`, `num_samples` (required), `num_warmup`.
- `step_size`, `num_leapfrog_steps`, `jitter_amount` (in `[0, 1]`).
- `adapt_trajectory_length` and the `trajectory_*` Adam controls for the
  ChEES trajectory-length ascent; `max_leapfrog_steps` clamps the result.
- `target_accept` (defaults to the ChEES-optimal `0.651`), `adapt_step_size`,
  `adapt_mass_matrix`.
- `backend`, `precision`: pass a `KernelAbstractions.Backend` to run
  device-resident.

See also [`batched_nuts`](@ref) (the reference/default sampler) and
[`batched_hmc`](@ref).
"""
function batched_chees(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    step_size::Real=0.1,
    num_leapfrog_steps::Int=10,
    jitter_amount::Real=1.0,
    adapt_trajectory_length::Bool=true,
    max_leapfrog_steps::Int=1000,
    trajectory_decay_rate::Real=0.5,
    trajectory_learning_rate::Real=0.5,
    trajectory_adam_beta1::Real=0.0,
    trajectory_adam_beta2::Real=0.95,
    trajectory_adam_epsilon::Real=1e-8,
    initial_params=nothing,
    init_strategy::Symbol=:prior,
    init_max_retries::Int=100,
    target_accept::Real=0.651,
    adapt_step_size::Bool=true,
    adapt_mass_matrix::Bool=true,
    find_reasonable_step_size::Bool=false,
    divergence_threshold::Real=1000.0,
    mass_matrix_regularization::Real=1e-3,
    mass_matrix_min_samples::Int=10,
    callback=nothing,
    callback_every::Int=10,
    backend=nothing,
    precision=nothing,
    rng::AbstractRNG=Random.default_rng(),
    _trajectory_trace::Union{Nothing,AbstractVector{Float64}}=nothing,
)
    init_strategy in (:prior, :uniform) ||
        throw(ArgumentError("batched_chees init_strategy must be :prior or :uniform, got $(repr(init_strategy))"))
    init_max_retries >= 0 ||
        throw(ArgumentError("batched_chees init_max_retries must be >= 0, got $init_max_retries"))
    0.0 <= jitter_amount <= 1.0 ||
        throw(ArgumentError("batched_chees jitter_amount must be in [0, 1], got $jitter_amount"))
    max_leapfrog_steps >= 1 ||
        throw(ArgumentError("batched_chees max_leapfrog_steps must be >= 1, got $max_leapfrog_steps"))
    0.0 < trajectory_decay_rate <= 1.0 ||
        throw(ArgumentError("batched_chees trajectory_decay_rate must be in (0, 1], got $trajectory_decay_rate"))
    trajectory_learning_rate > 0.0 ||
        throw(ArgumentError("batched_chees trajectory_learning_rate must be > 0, got $trajectory_learning_rate"))
    0.0 <= trajectory_adam_beta1 < 1.0 ||
        throw(ArgumentError("batched_chees trajectory_adam_beta1 must be in [0, 1), got $trajectory_adam_beta1"))
    0.0 <= trajectory_adam_beta2 < 1.0 ||
        throw(ArgumentError("batched_chees trajectory_adam_beta2 must be in [0, 1), got $trajectory_adam_beta2"))
    trajectory_adam_epsilon > 0.0 ||
        throw(ArgumentError("batched_chees trajectory_adam_epsilon must be > 0, got $trajectory_adam_epsilon"))

    # Device-resident ChEES-HMC (issue #161 increment 4). When `backend` is given the
    # jittered leapfrog trajectory runs device-resident (host-side RNG + O(num_chains)
    # bookkeeping, one sync per iteration); the cross-chain ChEES trajectory-length
    # adaptation runs on the host during warmup from a per-iteration proposal/momentum
    # download, and sampling downloads nothing for adaptation. Results are
    # statistically -- and, on the CPU() reference backend at Float64, numerically --
    # equivalent to the host path below, which is untouched when `backend === nothing`.
    # The ChEES-specific kwargs are validated above so a bad argument fails before the
    # device workspace lowers/allocates.
    if backend !== nothing
        backend isa KernelAbstractions.Backend ||
            throw(ArgumentError("batched_chees `backend` must be a KernelAbstractions.Backend or nothing, got $(typeof(backend))"))
        device_precision = precision === nothing ? default_device_precision(backend) : precision
        return _run_device_batched_chees(
            model,
            args,
            constraints;
            num_chains=num_chains,
            num_samples=num_samples,
            num_warmup=num_warmup,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            jitter_amount=jitter_amount,
            adapt_trajectory_length=adapt_trajectory_length,
            max_leapfrog_steps=max_leapfrog_steps,
            trajectory_decay_rate=trajectory_decay_rate,
            trajectory_learning_rate=trajectory_learning_rate,
            trajectory_adam_beta1=trajectory_adam_beta1,
            trajectory_adam_beta2=trajectory_adam_beta2,
            trajectory_adam_epsilon=trajectory_adam_epsilon,
            initial_params=initial_params,
            init_strategy=init_strategy,
            init_max_retries=init_max_retries,
            target_accept=target_accept,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            find_reasonable_step_size=find_reasonable_step_size,
            divergence_threshold=divergence_threshold,
            mass_matrix_regularization=mass_matrix_regularization,
            mass_matrix_min_samples=mass_matrix_min_samples,
            callback=callback,
            callback_every=callback_every,
            backend=backend,
            precision=device_precision,
            rng=rng,
            _trajectory_trace=_trajectory_trace,
        )
    end

    # Signature-aware sizing (#95), mirroring batched_hmc/batched_nuts.
    signature_layout = _batched_signature_layout(model, constraints)
    num_params = parametercount(signature_layout)
    constrained_num_params = parametervaluecount(signature_layout)
    _validate_batched_hmc_arguments(
        num_chains,
        num_params,
        num_samples,
        num_warmup,
        step_size,
        num_leapfrog_steps,
        target_accept,
        divergence_threshold,
        mass_matrix_regularization,
        mass_matrix_min_samples,
        args,
        constraints,
    )

    batch_args = _validate_batched_args(model, args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    position = _initial_batched_hmc_positions(
        model,
        batch_args,
        batch_constraints,
        initial_params,
        rng,
        num_params,
        constrained_num_params,
        num_chains;
        init_strategy=init_strategy,
    )
    inverse_mass_matrix = ones(num_params)
    workspace = BatchedHMCWorkspace(model, position, batch_args, batch_constraints, inverse_mass_matrix)
    current_logjoint = Vector{Float64}(undef, num_chains)
    current_gradient = workspace.current_gradient
    # Retry non-finite starting points (issue #162), mirroring batched_nuts: only
    # re-drawable inits (prior/uniform/Pathfinder) retry; a fixed `initial_params`
    # array reproduces the same value, so it fails fast.
    local gradient
    init_attempt = 0
    while true
        _, gradient = _batched_logjoint_and_gradient_unconstrained!(
            current_logjoint,
            workspace.gradient_cache,
            position,
        )
        copyto!(current_gradient, gradient)
        bad_columns = _nonfinite_init_columns(current_logjoint, current_gradient)
        isempty(bad_columns) && break
        if !_init_is_redrawable(initial_params) || init_attempt >= init_max_retries
            retried = _init_is_redrawable(initial_params) ? " after $init_max_retries re-draw(s)" : ""
            throw(
                ArgumentError(
                    "initial batched ChEES parameters produced a non-finite unconstrained logjoint or gradient in $(length(bad_columns)) of $num_chains chain(s)$retried; try init_strategy=:uniform or supply finite initial_params, or check the constraint values for NaN/Inf",
                ),
            )
        end
        init_attempt += 1
        _redraw_batched_initial_positions!(
            position,
            bad_columns,
            model,
            batch_args,
            batch_constraints,
            initial_params,
            init_strategy,
            rng,
            num_params,
            constrained_num_params,
            num_chains,
        )
    end

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, constrained_num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    integration_steps_values = Matrix{Int}(undef, num_samples, num_chains)
    total_iterations = num_warmup + num_samples
    chees_step_size = Float64(step_size)
    chees_target_accept = Float64(target_accept)
    chees_divergence_threshold = Float64(divergence_threshold)
    chees_jitter_amount = Float64(jitter_amount)

    if find_reasonable_step_size || (num_warmup > 0 && adapt_step_size)
        chees_step_size = _find_reasonable_batched_step_size(
            workspace,
            model,
            position,
            current_logjoint,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            chees_step_size,
            chees_divergence_threshold,
            rng,
        )
    end
    driver = WarmupDriver(
        num_params,
        num_warmup,
        chees_step_size,
        chees_target_accept;
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_regularization=mass_matrix_regularization,
        mass_matrix_min_samples=mass_matrix_min_samples,
    )
    refind = BatchedStepSizeSearch(
        workspace,
        model,
        position,
        current_logjoint,
        current_gradient,
        batch_args,
        batch_constraints,
        chees_divergence_threshold,
        rng,
    )

    # Trajectory-length adaptation state. `T` is initialized to
    # `num_leapfrog_steps * ε` (using the step size entering the loop), so with
    # `jitter_amount = 0` the first iteration integrates exactly
    # `ceil(T/ε) = num_leapfrog_steps` steps -- preserving the scaffold's fixed
    # behavior when adaptation is off. (BlackJAX instead starts `T = ε`, i.e. one
    # step, and adapts up; we warm-start from the user's `num_leapfrog_steps` -- a
    # documented deviation, see docs/chees-hmc.md.)
    initial_trajectory_length = num_leapfrog_steps * chees_step_size
    trajectory_state = _ChEESTrajectoryState(
        initial_trajectory_length,
        log(initial_trajectory_length),
        _ChEESAdamState(
            trajectory_learning_rate,
            trajectory_adam_beta1,
            trajectory_adam_beta2,
            trajectory_adam_epsilon,
        ),
        1,
        Float64(trajectory_decay_rate),
        max_leapfrog_steps,
    )
    # Snapshot buffer for the per-iteration start positions (ChEES `initials`).
    chees_initials =
        (num_warmup > 0 && adapt_trajectory_length) ? Matrix{Float64}(undef, num_params, num_chains) :
        Matrix{Float64}(undef, 0, 0)

    sample_index = 0
    cumulative_divergences = 0
    for iteration = 1:total_iterations
        chees_step_size = driver.step_size
        inverse_mass_matrix = driver.inverse_mass_matrix
        # Per-iteration Halton-jittered step count `L_iter = max(1, ceil(jitter_val
        # * T / ε))`. `jitter_value` is reused in the ChEES gradient below (BlackJAX
        # feeds the SAME jitter draw to both, chees_adaptation.py lines 461 & 769).
        # The Halton draw consumes no RNG, so momentum/accept draws are unchanged.
        jitter_value = _chees_jitter_value(chees_jitter_amount, iteration)
        leapfrog_steps =
            _chees_leapfrog_steps_from_jitter(jitter_value, trajectory_state.trajectory_length, chees_step_size)
        _update_sqrt_inverse_mass_matrix!(workspace.sqrt_inverse_mass_matrix, inverse_mass_matrix)
        _sample_batched_momentum!(workspace.momentum, rng, workspace.sqrt_inverse_mass_matrix)
        proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_matrix,
            batch_args,
            batch_constraints,
            chees_step_size,
            leapfrog_steps,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_matrix,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        energy_error = workspace.energy_error
        fill!(energy_error, Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)

        for chain_index = 1:num_chains
            if valid[chain_index]
                proposed_hamiltonian[chain_index] = _hamiltonian(
                    proposed_logjoint[chain_index],
                    view(proposal_momentum, :, chain_index),
                    inverse_mass_matrix,
                )
                log_accept_ratio[chain_index] =
                    current_hamiltonian[chain_index] - proposed_hamiltonian[chain_index]
                energy_error[chain_index] = proposed_hamiltonian[chain_index] - current_hamiltonian[chain_index]
                divergent_step[chain_index] =
                    !isfinite(energy_error[chain_index]) ||
                    energy_error[chain_index] > chees_divergence_threshold
            end
        end

        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)

        # Snapshot the START positions BEFORE the accept loop overwrites them; the
        # ChEES `initials` are the pre-transition states of every chain.
        if iteration <= num_warmup && adapt_trajectory_length
            copyto!(chees_initials, position)
        end

        accepted_step = workspace.accepted_step
        fill!(accepted_step, false)
        for chain_index = 1:num_chains
            if valid[chain_index] && log(rand(rng)) < min(0.0, log_accept_ratio[chain_index])
                copyto!(view(position, :, chain_index), view(proposal_position, :, chain_index))
                copyto!(view(current_gradient, :, chain_index), view(proposal_gradient, :, chain_index))
                current_logjoint[chain_index] = proposed_logjoint[chain_index]
                accepted_step[chain_index] = true
            end
        end

        cumulative_divergences += count(divergent_step)

        if iteration <= num_warmup
            _mass_adaptation_weights!(
                driver.variance_state,
                workspace.mass_adaptation_weights,
                accepted_step,
                accept_prob,
                divergent_step,
            )
            # Step-size dual averaging targets the HARMONIC mean of the chains'
            # acceptance probabilities (ChEES, not the arithmetic mean the scaffold
            # used via `_mean_batched_adaptation_probability`).
            accept_statistic = _chees_harmonic_mean_accept(accept_prob, divergent_step)
            warmup_update!(
                driver,
                iteration,
                accept_statistic,
                position,
                workspace.mass_adaptation_weights,
                refind,
            )
            # Cross-chain ChEES trajectory-length update. Uses the (possibly just
            # re-searched) `driver.step_size` for the [ε, max·ε] clip, matching
            # BlackJAX's use of `new_step_size` there.
            if adapt_trajectory_length
                _chees_trajectory_update!(
                    trajectory_state,
                    chees_initials,
                    proposal_position,
                    proposal_momentum,
                    accept_prob,
                    divergent_step,
                    inverse_mass_matrix,
                    jitter_value,
                    driver.step_size,
                )
            end
            _trajectory_trace === nothing || push!(_trajectory_trace, trajectory_state.trajectory_length)
            if iteration == num_warmup
                warmup_finalize!(driver)
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, chees_step_size, cumulative_divergences)
        end

        if iteration > num_warmup
            sample_index += 1
            for chain_index = 1:num_chains
                copyto!(view(unconstrained_samples, :, sample_index, chain_index), view(position, :, chain_index))
                _write_signature_constrained_sample!(
                    constrained_samples,
                    model,
                    view(position, :, chain_index),
                    sample_index,
                    _batched_args(batch_args, chain_index),
                    _batched_constraints(batch_constraints, chain_index),
                    chain_index,
                )
                logjoint_values[sample_index, chain_index] = current_logjoint[chain_index]
                acceptance_stats[sample_index, chain_index] = accept_prob[chain_index]
                energies[sample_index, chain_index] =
                    accepted_step[chain_index] ? proposed_hamiltonian[chain_index] : current_hamiltonian[chain_index]
                energy_errors[sample_index, chain_index] = energy_error[chain_index]
                accepted[sample_index, chain_index] = accepted_step[chain_index]
                divergent[sample_index, chain_index] = divergent_step[chain_index]
                integration_steps_values[sample_index, chain_index] = leapfrog_steps
            end
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :sample, sample_index, num_samples, chees_step_size, cumulative_divergences)
        end
    end

    mass_matrix = copy(driver.inverse_mass_matrix)
    chains = Vector{HMCChain}(undef, num_chains)
    for chain_index = 1:num_chains
        chains[chain_index] = HMCChain(
            :chees,
            model,
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            unconstrained_samples[:, :, chain_index],
            constrained_samples[:, :, chain_index],
            vec(logjoint_values[:, chain_index]),
            vec(acceptance_stats[:, chain_index]),
            vec(energies[:, chain_index]),
            vec(energy_errors[:, chain_index]),
            vec(accepted[:, chain_index]),
            vec(divergent[:, chain_index]),
            driver.step_size,
            copy(mass_matrix),
            num_leapfrog_steps,
            0,
            zeros(Int, num_samples),
            vec(integration_steps_values[:, chain_index]),
            chees_target_accept,
            copy(driver.mass_adaptation_windows),
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end
