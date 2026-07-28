# Shared warmup orchestration for the HMC/NUTS sampler drivers.
# Owns dual-averaging step-size adaptation and windowed running-variance mass
# adaptation so the single-chain and batched drivers share one state machine.

# `metric_kind` is :diag (default; the run is bitwise identical to the legacy
# path and `dense_covariance_state`/`dense_metric` stay nothing) or :dense. In
# dense mode the diagonal `variance_state` is still maintained verbatim (it drives
# the mass-adaptation weights, window bookkeeping, and the reported diagonal
# `mass_matrix`), and a parallel `dense_covariance_state` accumulates the full
# covariance from the same clipped samples to produce the `DenseMetric` used by
# the integrator.
mutable struct WarmupDriver
    warmup_schedule::WarmupSchedule
    dual_state::DualAveragingState
    variance_state::RunningVarianceState
    inverse_mass_matrix::Vector{Float64}
    step_size::Float64
    mass_window_index::Int
    mass_adaptation_windows::Vector{HMCMassAdaptationWindowSummary}
    num_params::Int
    num_warmup::Int
    adapt_step_size::Bool
    adapt_mass_matrix::Bool
    target_accept::Float64
    mass_matrix_regularization::Float64
    mass_matrix_min_samples::Int
    metric_kind::Symbol
    dense_covariance_state::Union{Nothing,DenseRunningCovarianceState}
    dense_metric::Union{Nothing,DenseMetric}
end

function WarmupDriver(
    num_params::Int,
    num_warmup::Int,
    initial_step_size::Real,
    target_accept::Real;
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
    metric::Symbol=:diag,
)
    metric in (:diag, :dense) || throw(ArgumentError("metric must be :diag or :dense, got :$metric"))
    step_size = Float64(initial_step_size)
    accept = Float64(target_accept)
    warmup_schedule = _warmup_schedule(num_warmup)
    dual_state = _dual_averaging_state(step_size, accept)
    initial_window_length =
        isempty(warmup_schedule.slow_window_ends) ? (_RUNNING_VARIANCE_CLIP_START + 16) :
        _warmup_window_length(warmup_schedule, 1)
    variance_state = _running_variance_state(num_params, initial_window_length)
    if metric === :dense
        dense_covariance_state = _dense_running_covariance_state(num_params, initial_window_length)
        dense_metric = DenseMetric(Matrix{Float64}(I, num_params, num_params))
    else
        dense_covariance_state = nothing
        dense_metric = nothing
    end
    return WarmupDriver(
        warmup_schedule,
        dual_state,
        variance_state,
        ones(num_params),
        step_size,
        1,
        HMCMassAdaptationWindowSummary[],
        num_params,
        num_warmup,
        adapt_step_size,
        adapt_mass_matrix,
        accept,
        Float64(mass_matrix_regularization),
        mass_matrix_min_samples,
        metric,
        dense_covariance_state,
        dense_metric,
    )
end

# The metric object the sampling loop threads through the integrator: the raw
# diagonal Vector for :diag (bitwise-identical legacy path) or the DenseMetric
# for :dense.
_driver_metric(driver::WarmupDriver) =
    driver.metric_kind === :dense ? driver.dense_metric : driver.inverse_mass_matrix

# Callable step-size re-search structs. They stand in for the closures the four
# drivers used, invoked only at window ends. Kept mutable so single-chain callers
# can refresh position/logjoint before each warmup step; batched callers hold the
# in-place-mutated buffers directly.

mutable struct ScalarStepSizeSearch{M,C,A,K,R<:AbstractRNG}
    model::M
    gradient_cache::C
    args::A
    constraints::K
    rng::R
    position::Vector{Float64}
    current_logjoint::Float64
end

function (search::ScalarStepSizeSearch)(step_size::Float64, inverse_mass_matrix::Union{Vector{Float64},MassMetric})
    return _find_reasonable_step_size(
        search.model,
        search.position,
        search.current_logjoint,
        search.gradient_cache,
        inverse_mass_matrix,
        search.args,
        search.constraints,
        step_size,
        search.rng,
    )
end

mutable struct BatchedStepSizeSearch{M,A,K,R<:AbstractRNG}
    workspace::Union{Nothing,BatchedHMCWorkspace}
    model::M
    position::Matrix{Float64}
    current_logjoint::Vector{Float64}
    current_gradient::Matrix{Float64}
    args::A
    constraints::K
    energy_bound::Float64
    rng::R
end

function (search::BatchedStepSizeSearch)(step_size::Float64, inverse_mass_matrix::Vector{Float64})
    workspace = search.workspace
    if workspace === nothing
        workspace = BatchedHMCWorkspace(
            search.model,
            search.position,
            search.args,
            search.constraints,
            inverse_mass_matrix,
        )
        search.workspace = workspace
    end
    return _find_reasonable_batched_step_size(
        workspace,
        search.model,
        search.position,
        search.current_logjoint,
        search.current_gradient,
        inverse_mass_matrix,
        search.args,
        search.constraints,
        step_size,
        search.energy_bound,
        search.rng,
    )
end

# One warmup iteration. `accept_statistic` drives dual averaging; `mass_weights`
# are the caller-computed per-sample weights (scalar for single chain, vector for
# batched) accumulated over `positions` (Vector or Matrix). `refind` is invoked
# only at window ends, exactly where the original drivers re-ran the search.
function warmup_update!(
    driver::WarmupDriver,
    iteration::Int,
    accept_statistic::Float64,
    positions,
    mass_weights,
    refind,
)
    if driver.adapt_step_size
        driver.step_size = _update_step_size!(driver.dual_state, accept_statistic)
    end

    schedule = driver.warmup_schedule
    if driver.adapt_mass_matrix &&
       driver.mass_window_index <= length(schedule.slow_window_ends) &&
       iteration > schedule.initial_buffer
        _update_running_variance!(driver.variance_state, positions, mass_weights)
        if driver.metric_kind === :dense
            _update_dense_covariance!(driver.dense_covariance_state, positions, mass_weights)
        end
        if iteration == schedule.slow_window_ends[driver.mass_window_index]
            mass_updated = false
            if _running_variance_effective_count(driver.variance_state) >= driver.mass_matrix_min_samples
                driver.inverse_mass_matrix =
                    _inverse_mass_matrix(driver.variance_state, driver.mass_matrix_regularization)
                if driver.metric_kind === :dense
                    driver.dense_metric = DenseMetric(
                        _dense_inverse_mass_matrix(driver.dense_covariance_state, driver.mass_matrix_regularization),
                    )
                end
                mass_updated = true
            end
            push!(
                driver.mass_adaptation_windows,
                _mass_adaptation_window_summary(
                    schedule,
                    driver.mass_window_index,
                    driver.variance_state,
                    driver.inverse_mass_matrix,
                    mass_updated,
                ),
            )
            driver.mass_window_index += 1
            if driver.mass_window_index <= length(schedule.slow_window_ends)
                next_window_length = _warmup_window_length(schedule, driver.mass_window_index)
                driver.variance_state = _running_variance_state(driver.num_params, next_window_length)
                if driver.metric_kind === :dense
                    driver.dense_covariance_state =
                        _dense_running_covariance_state(driver.num_params, next_window_length)
                end
            else
                driver.variance_state = _running_variance_state(driver.num_params)
                if driver.metric_kind === :dense
                    driver.dense_covariance_state = _dense_running_covariance_state(driver.num_params)
                end
            end
            if driver.adapt_step_size && iteration < driver.num_warmup
                driver.step_size = refind(driver.step_size, _driver_metric(driver))
                driver.dual_state = _dual_averaging_state(driver.step_size, driver.target_accept)
            end
        end
    end

    return driver.step_size
end

# Applied once, at iteration == num_warmup, after the last warmup_update!.
function warmup_finalize!(driver::WarmupDriver)
    if driver.adapt_step_size
        driver.step_size = _final_step_size(driver.dual_state)
    end
    if driver.adapt_mass_matrix &&
       _running_variance_effective_count(driver.variance_state) >= driver.mass_matrix_min_samples
        driver.inverse_mass_matrix =
            _inverse_mass_matrix(driver.variance_state, driver.mass_matrix_regularization)
        if driver.metric_kind === :dense
            driver.dense_metric = DenseMetric(
                _dense_inverse_mass_matrix(driver.dense_covariance_state, driver.mass_matrix_regularization),
            )
        end
    end
    return driver.step_size
end

# ---- pooled-mass / per-chain-step warmup driver --------------------------------
#
# The third adaptation mode (issue #137): a SHARED diagonal mass matrix pooled
# across all chains, plus a PER-CHAIN step size. This is exactly what the device
# masked NUTS/HMC path needs -- the device leapfrog kernels consume one shared
# `inverse_mass` P-vector (so the host per-chain-mass mode is unusable on device),
# but each chain still gets its own dual-averaged step, which is the operative fix
# for prior-draw stranding.
#
# The shared mass machinery is COMPOSED from an inner `WarmupDriver` built with
# `adapt_step_size=false`, so its `warmup_update!`/`warmup_finalize!` run the exact
# pooled running-variance + windowed mass math (bit-identical to shared mode) and
# never touch a step size. This driver adds the per-chain dual-averaging states and
# re-runs each chain's reasonable-step-size search against the freshly-pooled shared
# mass at every mass window end (mirroring the scalar driver's window-end re-search,
# but per chain).
mutable struct PooledMassPerChainStepDriver
    mass::WarmupDriver
    dual_states::Vector{DualAveragingState}
    step_sizes::Vector{Float64}
    target_accept::Float64
    num_warmup::Int
    adapt_step_size::Bool
end

function PooledMassPerChainStepDriver(
    num_params::Int,
    num_warmup::Int,
    initial_step_sizes::AbstractVector{Float64},
    target_accept::Real;
    adapt_step_size::Bool,
    adapt_mass_matrix::Bool,
    mass_matrix_regularization::Real,
    mass_matrix_min_samples::Int,
)
    accept = Float64(target_accept)
    # The inner driver owns ONLY the shared mass state; adapt_step_size=false keeps
    # its own scalar step untouched so it never runs a step re-search. Its initial
    # step is irrelevant (never read); use the first chain's for tidiness.
    mass = WarmupDriver(
        num_params,
        num_warmup,
        isempty(initial_step_sizes) ? 1.0 : initial_step_sizes[1],
        accept;
        adapt_step_size=false,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_regularization=mass_matrix_regularization,
        mass_matrix_min_samples=mass_matrix_min_samples,
    )
    dual_states = [_dual_averaging_state(step, accept) for step in initial_step_sizes]
    return PooledMassPerChainStepDriver(
        mass,
        dual_states,
        collect(Float64, initial_step_sizes),
        accept,
        num_warmup,
        adapt_step_size,
    )
end

# The shared diagonal mass metric threaded through every chain's integrator.
_driver_metric(driver::PooledMassPerChainStepDriver) = _driver_metric(driver.mass)

# One warmup iteration for the pooled-mass / per-chain-step driver.
# `accept_statistics` is a per-chain vector (each chain's own masked accept stat,
# 0.0 for a divergent chain). `positions` is the full P x C matrix (the mass is
# pooled over ALL chains, exactly once). `mass_weights` is the per-chain weight
# vector. `refind_per_chain` is a vector of callables, invoked at each mass window
# end to re-search each chain's step against the just-updated shared mass.
function warmup_update!(
    driver::PooledMassPerChainStepDriver,
    iteration::Int,
    accept_statistics::AbstractVector,
    positions,
    mass_weights,
    refind_per_chain,
)
    if driver.adapt_step_size
        @inbounds for c in eachindex(driver.step_sizes)
            driver.step_sizes[c] = _update_step_size!(driver.dual_states[c], accept_statistics[c])
        end
    end

    mass = driver.mass
    schedule = mass.warmup_schedule
    # A mass window closes on THIS iteration exactly when the inner driver's mass
    # block will advance its window index (same guard as the scalar path). Capture
    # it BEFORE the inner update advances `mass_window_index`.
    window_ends_now =
        mass.adapt_mass_matrix &&
        mass.mass_window_index <= length(schedule.slow_window_ends) &&
        iteration > schedule.initial_buffer &&
        iteration == schedule.slow_window_ends[mass.mass_window_index]

    # Pooled shared-mass update. adapt_step_size=false => this runs only the running
    # variance + windowed mass recomputation and never invokes the refind below.
    warmup_update!(mass, iteration, 0.0, positions, mass_weights, nothing)

    # At a window end, re-search each chain's step against the NEW shared mass and
    # restart that chain's dual averaging (the per-chain analog of the scalar
    # driver's window-end reset at warmup.jl's `refind` block).
    if window_ends_now && driver.adapt_step_size && iteration < driver.num_warmup
        metric = _driver_metric(mass)
        @inbounds for c in eachindex(driver.step_sizes)
            driver.step_sizes[c] = refind_per_chain[c](driver.step_sizes[c], metric)
            driver.dual_states[c] = _dual_averaging_state(driver.step_sizes[c], driver.target_accept)
        end
    end

    return driver.step_sizes
end

function warmup_finalize!(driver::PooledMassPerChainStepDriver)
    if driver.adapt_step_size
        @inbounds for c in eachindex(driver.step_sizes)
            driver.step_sizes[c] = _final_step_size(driver.dual_states[c])
        end
    end
    warmup_finalize!(driver.mass)
    return driver.step_sizes
end
