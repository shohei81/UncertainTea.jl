# MEADS -- Maximum-Eigenvalue Adaptation of Damping and Step-size (issue #233).
#
# MEADS (Hoffman & Sountsov, "Tuning-Free Generalized Hamiltonian Monte Carlo",
# AISTATS 2022) is the self-tuning many-chain companion to `batched_chees`, and
# completes the #161 device-native pair. It is a generalized-HMC ensemble sampler:
# every iteration each chain does exactly ONE leapfrog step with a partially
# refreshed (persistent) momentum and a Metropolis-with-persistent-slice accept
# (the paper's Algorithm 1). There is no U-turn recursion and no warmup /
# dual-averaging phase: the step size, damping, slice drift and a diagonal
# preconditioner are recomputed every iteration from cross-chain ensemble
# statistics, so adaptation runs continuously and the whole chain is usable.
#
# The adaptation follows the paper exactly (Algorithm 3). Chains are split into
# `num_folds` folds; fold k's parameters are computed from the states of its
# NEIGHBOR fold (k-1) mod K, and one fold (rotating with the iteration index) is
# skipped each iteration to keep the complementary-folds dependency structure a
# DAG (Figure 1/2). All fold statistics are host-side O(P*C) column reductions
# (max-eigenvalue estimates of the whitened gradient- and position-covariances,
# and componentwise variances). Moving them on-device is a follow-up (the #220
# pattern for ChEES); this file is the CPU (host batched) reference.
#
# Reuse: this shares the batched leapfrog + Hamiltonian building blocks with
# `batched_hmc`/`batched_chees` (`batched_leapfrog_trajectory!` per-chain overload,
# `_batched_hamiltonian!`, `_sample_batched_momentum!`), the `BatchedHMCWorkspace`,
# the signature-aware sizing (#95), and the #162 init-retry loop. It does NOT
# touch `batched_nuts`/`batched_hmc`/`batched_chees`.

# --- MEADS constants -------------------------------------------------------------
# The step-size safety margin: eps := 0.5 / sqrt(lambda_max(-Hbar)) (paper Sec 4.2,
# the 1/2 giving "some margin of error"; the 2/sqrt(lambda_max) stability bound with
# a factor 1/2). Floors keep the reductions finite when a fold is degenerate.
const _MEADS_STEP_MARGIN = 0.5
const _MEADS_EIG_FLOOR = 1e-12
const _MEADS_STD_FLOOR = 1e-10

# Largest-eigenvalue estimator (paper Algorithm 2 + Eq. 7): for a matrix whose
# columns are the N ensemble vectors x_n in R^D, estimate lambda_max of the second
# moment E[x x'] via the trace ratio tr(Sigma^2)/tr(Sigma) = (sum lambda^2)/(sum
# lambda), which only needs reasonable trace estimates (not individual
# eigenvalues). With the Gram matrix S = X' X (S[n,n'] = x_n . x_n'):
#   lambda_bar   := tr(S) / N                          (~ tr(Sigma))
#   lambda2_bar  := (1/(N(N-1))) sum_{n != n'} S[n,n']^2   (~ tr(Sigma^2))
#   return lambda2_bar / lambda_bar.
# `columns` is P x N (parameters x ensemble members), matching the sampler's
# column-major (param-major) layout. Returns 0.0 for a degenerate (<2-member or
# zero-norm) ensemble, which the callers treat as "no information" (eps capped to 1,
# damping driven by the floor).
function _meads_max_eig(columns::AbstractMatrix)
    num_params = size(columns, 1)
    num_members = size(columns, 2)
    num_members >= 2 || return 0.0
    trace_sum = 0.0
    @inbounds for member = 1:num_members
        norm_squared = 0.0
        for param = 1:num_params
            value = columns[param, member]
            norm_squared += value * value
        end
        trace_sum += norm_squared
    end
    lambda_bar = trace_sum / num_members
    lambda_bar > 0.0 || return 0.0
    off_diagonal_sum = 0.0
    @inbounds for member_a = 1:num_members
        for member_b = 1:num_members
            member_a == member_b && continue
            gram = 0.0
            for param = 1:num_params
                gram += columns[param, member_a] * columns[param, member_b]
            end
            off_diagonal_sum += gram * gram
        end
    end
    lambda2_bar = off_diagonal_sum / (num_members * (num_members - 1))
    return lambda2_bar / lambda_bar
end

# Per-fold MEADS parameter map (paper Algorithm 3, lines 5-10), computed from a
# NEIGHBOR fold's states/gradients:
#   mu_d, sigma_d     := mean, std over the neighbor fold's states (diagonal
#                        preconditioner; inverse mass = sigma^2)
#   theta_bar_{n,d}   := (theta_{n,d} - mu_d) / sigma_d        (whitened states)
#   g_bar_{n,d}       := grad_{n,d} * sigma_d                  (whitened gradients)
#   eps               := min(1, 0.5 / sqrt(max_eig(g_bar)))    (line 8)
#   gamma             := max(1/(t*eps), 1/sqrt(max_eig(theta_bar)))  (line 9, floor)
#   alpha             := 1 - exp(-2*eps*gamma)                 (line 10, damping)
#   delta             := alpha / 2                             (line 10, slice drift)
# Writes sigma into `std_out` and returns (eps, alpha, delta).
function _meads_fold_parameters!(
    std_out::AbstractVector{Float64},
    whitened_positions::AbstractMatrix{Float64},
    whitened_gradients::AbstractMatrix{Float64},
    position::AbstractMatrix{Float64},
    current_gradient::AbstractMatrix{Float64},
    neighbor_columns::AbstractVector{Int},
    iteration::Int,
)
    num_params = size(position, 1)
    num_members = length(neighbor_columns)
    # Mean and (population) std over the neighbor fold's columns, per parameter.
    @inbounds for param = 1:num_params
        mean_value = 0.0
        for member in neighbor_columns
            mean_value += position[param, member]
        end
        mean_value /= num_members
        variance = 0.0
        for member in neighbor_columns
            delta_value = position[param, member] - mean_value
            variance += delta_value * delta_value
        end
        variance /= num_members
        std_value = sqrt(variance)
        std_value = std_value > _MEADS_STD_FLOOR ? std_value : _MEADS_STD_FLOOR
        std_out[param] = std_value
        inverse_std = 1.0 / std_value
        for (member_index, member) in enumerate(neighbor_columns)
            whitened_positions[param, member_index] = (position[param, member] - mean_value) * inverse_std
            whitened_gradients[param, member_index] = current_gradient[param, member] * std_value
        end
    end

    positions_view = view(whitened_positions, :, 1:num_members)
    gradients_view = view(whitened_gradients, :, 1:num_members)
    eig_gradient = _meads_max_eig(gradients_view)
    eig_position = _meads_max_eig(positions_view)

    step_size = min(1.0, _MEADS_STEP_MARGIN / sqrt(max(eig_gradient, _MEADS_EIG_FLOOR)))
    damping_floor = 1.0 / (iteration * step_size)
    gamma = max(damping_floor, 1.0 / sqrt(max(eig_position, _MEADS_EIG_FLOOR)))
    alpha = 1.0 - exp(-2.0 * step_size * gamma)
    alpha = clamp(alpha, 0.0, 1.0)
    delta = alpha / 2.0
    return step_size, alpha, delta
end

# Persistent-slice drift (paper Algorithm 1, line 10): u_tilde := ((u+1+delta) mod
# 2) - 1, keeping u_tilde in [-1, 1). `mod(x, 2)` lands in [0, 2), so the result is
# in [-1, 1).
_meads_slice_drift(slice_value::Float64, delta::Float64) = mod(slice_value + 1.0 + delta, 2.0) - 1.0

"""
    batched_meads(model, args=(), constraints=choicemap(); num_chains, num_samples, kwargs...) -> HMCChains

Run MEADS (Maximum-Eigenvalue Adaptation of Damping and Step-size; Hoffman &
Sountsov, AISTATS 2022): a tuning-free generalized-HMC ensemble sampler. Every
chain takes ONE leapfrog step per iteration with a partially refreshed persistent
momentum and a Metropolis-with-persistent-slice accept, so every lane does
identical work each step (no U-turn recursion, no control-flow divergence). There
is no warmup / dual-averaging phase: the step size, damping, slice drift and a
diagonal preconditioner are recomputed each iteration from cross-chain ensemble
statistics (complementary K-fold adaptation), so the whole chain is usable.
Returns an [`HMCChains`](@ref).

Key keyword arguments:

- `num_chains`, `num_samples` (required), `num_warmup` (burn-in draws to discard;
  adaptation is continuous, so this is optional).
- `num_folds` (default 4): the number of complementary folds; `num_chains` must be
  at least `2 * num_folds` (each fold needs >= 2 members for the ensemble
  statistics).
- `init_strategy`, `initial_params`, `init_max_retries`: initialization, as `batched_nuts`.
- `divergence_threshold`.

This is the CPU (host batched) reference; the fold reductions run on the host. See
also [`batched_chees`](@ref) (the many-chain trajectory-length-adapting sampler)
and [`batched_nuts`](@ref) (the reference/default sampler). See docs/chees-hmc.md
for the MEADS design notes and the exact paper equations.
"""
function batched_meads(
    model::TeaModel,
    args=(),
    constraints=choicemap();
    num_chains::Int,
    num_samples::Int,
    num_warmup::Int=0,
    num_folds::Int=4,
    initial_params=nothing,
    init_strategy::Symbol=:prior,
    init_max_retries::Int=100,
    divergence_threshold::Real=1000.0,
    callback=nothing,
    callback_every::Int=10,
    backend=nothing,
    precision=nothing,
    rng::AbstractRNG=Random.default_rng(),
    _adapt_trace::Union{Nothing,AbstractVector}=nothing,
)
    init_strategy in (:prior, :uniform) ||
        throw(ArgumentError("batched_meads init_strategy must be :prior or :uniform, got $(repr(init_strategy))"))
    init_max_retries >= 0 ||
        throw(ArgumentError("batched_meads init_max_retries must be >= 0, got $init_max_retries"))
    num_folds >= 2 ||
        throw(ArgumentError("batched_meads num_folds must be >= 2, got $num_folds"))
    num_chains >= 2 * num_folds || throw(
        ArgumentError(
            "batched_meads needs num_chains >= 2 * num_folds (each fold's ensemble statistics need >= 2 members); got num_chains=$num_chains, num_folds=$num_folds",
        ),
    )
    num_samples > 0 || throw(ArgumentError("batched_meads num_samples must be > 0, got $num_samples"))
    num_warmup >= 0 || throw(ArgumentError("batched_meads num_warmup must be >= 0, got $num_warmup"))
    if backend !== nothing
        throw(
            ArgumentError(
                "batched_meads currently supports only the host (CPU) backend; the device fold reductions are a follow-up (issue #233). Call without `backend`.",
            ),
        )
    end
    precision === nothing ||
        throw(ArgumentError("batched_meads `precision` requires a device backend, which is not yet supported"))

    # Signature-aware sizing (#95), mirroring batched_hmc/batched_nuts/batched_chees.
    signature_layout = _batched_signature_layout(model, constraints)
    num_params = parametercount(signature_layout)
    constrained_num_params = parametervaluecount(signature_layout)
    num_params > 0 || throw(ArgumentError("batched_meads requires a model with at least one latent parameter"))

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

    # Retry non-finite starting points (issue #162), mirroring batched_chees.
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
                    "initial batched MEADS parameters produced a non-finite unconstrained logjoint or gradient in $(length(bad_columns)) of $num_chains chain(s)$retried; try init_strategy=:uniform or supply finite initial_params",
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

    # Complementary folds: contiguous partition of the chains. Fold k's parameters
    # are computed from its neighbor fold (k-1) mod K (`neighbor_of`), and the fold
    # `skip = ((t-1) mod K) + 1` is not updated at iteration t (rotating), keeping
    # the dependency structure a DAG.
    fold_of_chain = Vector{Int}(undef, num_chains)
    fold_columns = Vector{Vector{Int}}(undef, num_folds)
    base = div(num_chains, num_folds)
    remainder = num_chains - base * num_folds
    start_index = 1
    for fold = 1:num_folds
        fold_size = base + (fold <= remainder ? 1 : 0)
        stop_index = start_index + fold_size - 1
        fold_columns[fold] = collect(start_index:stop_index)
        for chain = start_index:stop_index
            fold_of_chain[chain] = fold
        end
        start_index = stop_index + 1
    end
    neighbor_of(fold) = fold == 1 ? num_folds : fold - 1

    # Persistent generalized-HMC state carried across iterations: the WHITENED
    # (N(0,I)-target) momentum and the slice variable u ~ Uniform(-1, 1) (paper
    # Algorithm 3, lines 1-2). The preconditioner is folded into the leapfrog via a
    # per-chain diagonal inverse mass (= neighbor fold's sigma^2), so leapfrog with
    # scalar step eps and inverse mass sigma^2 reproduces the paper's per-dimension
    # step eps*sigma with a unit-variance momentum.
    unit_momentum = Matrix{Float64}(undef, num_params, num_chains)
    for chain = 1:num_chains
        for param = 1:num_params
            unit_momentum[param, chain] = randn(rng)
        end
    end
    slice_u = Vector{Float64}(undef, num_chains)
    for chain = 1:num_chains
        slice_u[chain] = 2.0 * rand(rng) - 1.0
    end

    # Per-chain parameter buffers (filled from each chain's fold each iteration).
    chain_step = Vector{Float64}(undef, num_chains)
    chain_alpha = Vector{Float64}(undef, num_chains)
    chain_delta = Vector{Float64}(undef, num_chains)
    inverse_mass_columns = Matrix{Float64}(undef, num_params, num_chains)
    sqrt_inverse_mass_columns = Matrix{Float64}(undef, num_params, num_chains)
    fold_std = Vector{Float64}(undef, num_params)
    max_fold_size = maximum(length, fold_columns)
    whitened_positions = Matrix{Float64}(undef, num_params, max_fold_size)
    whitened_gradients = Matrix{Float64}(undef, num_params, max_fold_size)
    refreshed_unit = Matrix{Float64}(undef, num_params, num_chains)

    unconstrained_samples = Array{Float64}(undef, num_params, num_samples, num_chains)
    constrained_samples = Array{Float64}(undef, constrained_num_params, num_samples, num_chains)
    logjoint_values = Matrix{Float64}(undef, num_samples, num_chains)
    acceptance_stats = Matrix{Float64}(undef, num_samples, num_chains)
    energies = Matrix{Float64}(undef, num_samples, num_chains)
    energy_errors = Matrix{Float64}(undef, num_samples, num_chains)
    accepted = falses(num_samples, num_chains)
    divergent = falses(num_samples, num_chains)
    integration_steps_values = zeros(Int, num_samples, num_chains)

    meads_divergence_threshold = Float64(divergence_threshold)
    total_iterations = num_warmup + num_samples
    sample_index = 0
    cumulative_divergences = 0
    final_step_size = 0.0

    for iteration = 1:total_iterations
        skip_fold = ((iteration - 1) % num_folds) + 1

        # --- Per-fold parameter maps from the NEIGHBOR fold's pre-update states ----
        step_sum = 0.0
        alpha_sum = 0.0
        mass_max = 0.0
        for fold = 1:num_folds
            neighbor = neighbor_of(fold)
            step_size, alpha, delta = _meads_fold_parameters!(
                fold_std,
                whitened_positions,
                whitened_gradients,
                position,
                current_gradient,
                fold_columns[neighbor],
                iteration,
            )
            for chain in fold_columns[fold]
                chain_step[chain] = step_size
                chain_alpha[chain] = alpha
                chain_delta[chain] = delta
                for param = 1:num_params
                    std_value = fold_std[param]
                    inverse_mass_columns[param, chain] = std_value * std_value
                    sqrt_inverse_mass_columns[param, chain] = std_value
                end
            end
            step_sum += step_size
            alpha_sum += alpha
            mass_max = max(mass_max, maximum(fold_std))
        end
        final_step_size = step_sum / num_folds
        if _adapt_trace !== nothing
            push!(_adapt_trace, (step=step_sum / num_folds, damping=alpha_sum / num_folds, max_std=mass_max))
        end

        # --- Partial momentum refresh (whitened): m_tilde = sqrt(1-a) m + sqrt(a) xi
        # (paper Algorithm 1, line 9) -- then convert to original-coordinate momentum
        # p = m_tilde / sigma for the leapfrog. Draw a fresh unit normal for every
        # chain (uniform work per lane); skipped-fold draws are discarded below.
        for chain = 1:num_chains
            alpha = chain_alpha[chain]
            keep = sqrt(max(0.0, 1.0 - alpha))
            refresh = sqrt(max(0.0, alpha))
            for param = 1:num_params
                xi = randn(rng)
                refreshed = keep * unit_momentum[param, chain] + refresh * xi
                refreshed_unit[param, chain] = refreshed
                workspace.momentum[param, chain] = refreshed / sqrt_inverse_mass_columns[param, chain]
            end
        end

        # --- One leapfrog step per chain (paper Algorithm 1, lines 1-6 / line 11) --
        proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, valid = _batched_leapfrog!(
            workspace,
            model,
            position,
            current_gradient,
            inverse_mass_columns,
            batch_args,
            batch_constraints,
            chain_step,
            1,
        )

        current_hamiltonian = _batched_hamiltonian!(
            workspace.current_hamiltonian,
            current_logjoint,
            workspace.momentum,
            inverse_mass_columns,
        )
        proposed_hamiltonian = workspace.proposed_hamiltonian
        copyto!(proposed_hamiltonian, current_hamiltonian)
        log_accept_ratio = workspace.log_accept_ratio
        fill!(log_accept_ratio, -Inf)
        energy_error = workspace.energy_error
        fill!(energy_error, Inf)
        divergent_step = workspace.divergent_step
        fill!(divergent_step, true)

        for chain = 1:num_chains
            if valid[chain]
                proposed_hamiltonian[chain] = _hamiltonian(
                    proposed_logjoint[chain],
                    view(proposal_momentum, :, chain),
                    view(inverse_mass_columns, :, chain),
                )
                log_accept_ratio[chain] = current_hamiltonian[chain] - proposed_hamiltonian[chain]
                energy_error[chain] = proposed_hamiltonian[chain] - current_hamiltonian[chain]
                divergent_step[chain] =
                    !isfinite(energy_error[chain]) || energy_error[chain] > meads_divergence_threshold
            end
        end
        accept_prob = _batched_acceptance_probability!(workspace.accept_prob, log_accept_ratio)

        # --- Persistent-slice Metropolis accept (paper Algorithm 1, lines 10-16) ---
        # Accept iff |u_tilde| <= exp(-dH) i.e. log|u_tilde| <= log_accept_ratio.
        # Accept: keep (theta', m', u_tilde/ratio). Reject: keep (theta, -m_tilde,
        # u_tilde). The leapfrog returns the NEGATED forward momentum, so the carried
        # whitened momentum on accept is -proposal_momentum * sigma.
        accepted_step = workspace.accepted_step
        fill!(accepted_step, false)
        for chain = 1:num_chains
            drifted_u = _meads_slice_drift(slice_u[chain], chain_delta[chain])
            fold = fold_of_chain[chain]
            if fold == skip_fold
                # Skipped fold: identity map (theta, m, u all carried unchanged).
                continue
            end
            if valid[chain] && log(abs(drifted_u)) <= log_accept_ratio[chain]
                copyto!(view(position, :, chain), view(proposal_position, :, chain))
                copyto!(view(current_gradient, :, chain), view(proposal_gradient, :, chain))
                current_logjoint[chain] = proposed_logjoint[chain]
                for param = 1:num_params
                    unit_momentum[param, chain] =
                        -proposal_momentum[param, chain] * sqrt_inverse_mass_columns[param, chain]
                end
                # u_tilde / ratio = u_tilde * exp(dH) = u_tilde * exp(energy_error).
                slice_u[chain] = drifted_u * exp(energy_error[chain])
                accepted_step[chain] = true
            else
                for param = 1:num_params
                    unit_momentum[param, chain] = -refreshed_unit[param, chain]
                end
                slice_u[chain] = drifted_u
            end
        end

        for chain = 1:num_chains
            fold_of_chain[chain] == skip_fold && continue
            divergent_step[chain] && (cumulative_divergences += 1)
        end

        if iteration <= num_warmup
            isnothing(callback) || _invoke_progress_callback(
                callback, callback_every, :warmup, iteration, num_warmup, final_step_size, cumulative_divergences)
            continue
        end

        sample_index += 1
        for chain = 1:num_chains
            copyto!(view(unconstrained_samples, :, sample_index, chain), view(position, :, chain))
            _write_signature_constrained_sample!(
                constrained_samples,
                model,
                view(position, :, chain),
                sample_index,
                _batched_args(batch_args, chain),
                _batched_constraints(batch_constraints, chain),
                chain,
            )
            logjoint_values[sample_index, chain] = current_logjoint[chain]
            if fold_of_chain[chain] == skip_fold
                # Carried (identity) state: a valid posterior draw. Record it as a
                # trivially-kept transition so it does not distort the divergence /
                # acceptance diagnostics.
                acceptance_stats[sample_index, chain] = 1.0
                accepted[sample_index, chain] = true
                divergent[sample_index, chain] = false
                energy_errors[sample_index, chain] = 0.0
                integration_steps_values[sample_index, chain] = 0
            else
                acceptance_stats[sample_index, chain] = accept_prob[chain]
                accepted[sample_index, chain] = accepted_step[chain]
                divergent[sample_index, chain] = divergent_step[chain]
                energy_errors[sample_index, chain] = energy_error[chain]
                integration_steps_values[sample_index, chain] = 1
            end
            # Hamiltonian at the recorded state (potential + carried whitened kinetic).
            kinetic = 0.0
            for param = 1:num_params
                value = unit_momentum[param, chain]
                kinetic += value * value
            end
            energies[sample_index, chain] = -current_logjoint[chain] + 0.5 * kinetic
        end
        isnothing(callback) || _invoke_progress_callback(
            callback, callback_every, :sample, sample_index, num_samples, final_step_size, cumulative_divergences)
    end

    chains = Vector{HMCChain}(undef, num_chains)
    for chain = 1:num_chains
        mass_column = collect(view(inverse_mass_columns, :, chain))
        chains[chain] = HMCChain(
            :meads,
            model,
            _batched_args(batch_args, chain),
            _batched_constraints(batch_constraints, chain),
            unconstrained_samples[:, :, chain],
            constrained_samples[:, :, chain],
            vec(logjoint_values[:, chain]),
            vec(acceptance_stats[:, chain]),
            vec(energies[:, chain]),
            vec(energy_errors[:, chain]),
            vec(accepted[:, chain]),
            vec(divergent[:, chain]),
            final_step_size,
            mass_column,
            1,
            0,
            zeros(Int, num_samples),
            vec(integration_steps_values[:, chain]),
            NaN,
            HMCMassAdaptationWindowSummary[],
            nothing,
        )
    end

    return HMCChains(model, args, constraints, chains)
end
