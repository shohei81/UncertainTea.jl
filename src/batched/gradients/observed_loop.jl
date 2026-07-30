# Observed-loop shared-address fast path for the batched gradient walker
# (issue #140): a loop whose body is a single observed scalar choice with an
# iterator-only address resolves the constraint address ONCE per iteration and
# fills all chains with one lookup (`_batched_observed_choice_values!`),
# mirroring the scoring walker's `_score_backend_observed_loop_choice!` fast
# path. The generic per-chain address build + Dict lookup in
# `_batched_choice_numeric_values!` is O(observations x chains); this path is
# O(observations). Values, gradients, and logjoint are bitwise identical to
# the generic walk (validated in
# test/uncertaintea/core/batched_observed_loop_gradient.jl).

# On top of the fast path, issue #141 stages the loop's observations once per
# gradient cache (the constraints are fixed for a cache's lifetime) and, for a
# normal family whose parameter expressions are loop-INVARIANT (they never
# read the iterator slot), hoists the parameter evaluation out of the loop and
# reduces over observations per chain with a single P x B chain-rule
# application at the end: O(observations x parameters x chains) becomes
# O(observations x chains + parameters x chains). The reduction reassociates
# the observation sum, so it is tolerance-equal (~1e-12), not bitwise; when
# the parameter expressions are loop-varying the walk keeps the per-iteration
# fast path (still fed from the staged observation vector, which IS bitwise).

# On top of the hoisted tier, issue #146 fuses the observation reduction
# itself: for an iid exponential-family loop (single observed choice, shared
# constraints, loop-INVARIANT parameter expressions) the per-chain
# O(observations) sum collapses to a closed form over sufficient statistics
# cached once per (step, iterable) alongside the staged observation vector —
# O(observations x chains) becomes O(chains). Normal uses the CENTERED stats
# (n, ybar, S2c = sum((y - ybar)^2)); the naive sum(y^2) - 2*mu*sum(y) + n*mu^2
# form cancels catastrophically when |ybar| >> sigma. Exponential fuses over
# (n, sum(y)); poisson over (n, sum(y), sum(log y!)) with the log-factorial
# data constant computed once at staging. Data that cannot take the closed
# form (non-finite values, negative exponential values, non-count poisson
# values, an empty loop) falls back to the hoisted/per-iteration tiers.

# Test seams: let the validation tests drive the generic walk / the
# non-hoisted loop / the unfused reduction on the same model. Always true in
# production.
const _BATCHED_GRADIENT_OBSERVED_LOOP_FAST_PATH = Ref(true)
const _BATCHED_GRADIENT_OBSERVED_LOOP_HOIST = Ref(true)
const _BATCHED_GRADIENT_OBSERVED_LOOP_SUFFSTATS = Ref(true)

# The family set mirrors the scoring walker's `_score_backend_observed_loop_choice!`
# methods; anything else (or a non-single-observed-choice body) takes the
# generic per-iteration body walk.
_backend_observed_loop_gradient_supported(choice) = false
_backend_observed_loop_gradient_supported(
    ::Union{
        BackendNormalChoicePlanStep,
        BackendLognormalChoicePlanStep,
        BackendExponentialChoicePlanStep,
        BackendGammaChoicePlanStep,
        BackendInverseGammaChoicePlanStep,
        BackendWeibullChoicePlanStep,
        BackendBetaChoicePlanStep,
        BackendStudentTChoicePlanStep,
        BackendLaplaceChoicePlanStep,
        BackendCauchyChoicePlanStep,
        BackendHalfNormalChoicePlanStep,
        BackendHalfCauchyChoicePlanStep,
        BackendUniformChoicePlanStep,
        BackendLogisticChoicePlanStep,
        BackendGumbelChoicePlanStep,
        BackendBernoulliChoicePlanStep,
        BackendBernoulliLogitChoicePlanStep,
        BackendPoissonChoicePlanStep,
        BackendGeometricChoicePlanStep,
        BackendBinomialChoicePlanStep,
        BackendNegativeBinomialChoicePlanStep,
        BackendCategoricalChoicePlanStep,
    },
) = true

function _score_backend_observed_loop_and_gradient!(
    choice,
    step::BackendLoopPlanStep,
    reference_iterable,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    # shared constraints: stage the observation vector once per (step,
    # iterable) on the gradient cache (issue #141). Per-chain constraint
    # vectors keep the per-iteration shared-address lookup.
    observed =
        constraints isa ChoiceMap ?
        _gathered_observed_loop_values!(cache, env, step, choice, reference_iterable, constraints) : nothing

    # fused GLM linear-predictor gradient (issue #150): a bernoulli-logit
    # observation whose eta is the fused linear predictor cannot take the
    # sufficient-statistics/hoisted tiers (eta reads the iterator through the
    # covariate column), but the per-iteration generic walk pays a dense P x B
    # gradient-plane update PER observation. This tier reduces over observations
    # FIRST (per chain), then applies the P x B chain rule once — the same
    # observation-order reassociation the hoisted normal tier uses (tolerance,
    # not bitwise). Requires a loop-invariant intercept (checked) plus a
    # loop-invariant coefficient vector + covariate matrix (guaranteed by the
    # linear-predictor lowering: coef is a latent vector, matrix a model arg).
    if !isnothing(observed) &&
       _BATCHED_GRADIENT_OBSERVED_LOOP_HOIST[] &&
       choice isa BackendBernoulliLogitChoicePlanStep &&
       choice.eta isa BackendLinearPredictorExpr &&
       (isnothing(choice.eta.intercept) || !_backend_expr_reads_slot(choice.eta.intercept, step.iterator_slot))
        _score_backend_observed_loop_bernoullilogit_glm!(choice, step, observed, reference_iterable, totals, gradients, cache, env)
        return totals, gradients
    end

    # sufficient-statistics fusion (issue #146): loop-invariant parameters +
    # shared constraints + closed-form-compatible data replace the whole
    # observation reduction with a few cached numbers
    if !isnothing(observed) && _BATCHED_GRADIENT_OBSERVED_LOOP_SUFFSTATS[]
        stats = _observed_loop_suffstats!(cache, step, choice, observed, reference_iterable)
        if !isnothing(stats)
            _score_backend_observed_loop_gradient_fused!(choice, stats, totals, gradients, cache, env)
            return totals, gradients
        end
    end

    if !isnothing(observed) &&
       _BATCHED_GRADIENT_OBSERVED_LOOP_HOIST[] &&
       _backend_observed_loop_gradient_hoistable(choice, step.iterator_slot)
        _score_backend_observed_loop_gradient_hoisted!(choice, observed, totals, gradients, cache, env)
        return totals, gradients
    end

    # the observed choice has no parameter slot, so its value-gradient seed is
    # identically zero on every iteration: zero the shared scratch once for
    # the whole loop (the continuous per-iteration methods read it; the
    # discrete families take no value seed)
    fill!(_batched_backend_gradient_scratch!(cache, 1), zero(T))
    for (position, item) in enumerate(reference_iterable)
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        if isnothing(observed)
            address = _concrete_address(env, choice.address, 1)
            _batched_observed_choice_values!(env.observed_values, constraints, address)
        else
            fill!(env.observed_values, observed[position])
        end
        _score_backend_observed_loop_choice_gradient!(choice, totals, gradients, cache, env)
    end
    return totals, gradients
end

# Fused bernoulli-logit GLM gradient over the whole observation loop. Per chain,
# with eta_i = intercept + sum_d coef[d]*X[d, col_i] and
# derivative_i = y_i - logistic(eta_i):
#   logjoint sum      = sum_i [y_i*eta_i - log1p(exp(eta_i))]
#   d/d_intercept     = sum_i derivative_i         (chain-ruled through the
#                                                   loop-invariant intercept plane)
#   d/d_coef[d]       = sum_i derivative_i*X[d, col_i]  (seeded on coef's rows)
# The intercept value+gradient is evaluated ONCE (loop-invariant); the covariate
# columns are resolved once from the iterator. A non-0/1 observation scores -Inf
# and contributes no gradient, exactly like `_accumulate_bernoullilogit_gradient!`.
function _score_backend_observed_loop_bernoullilogit_glm!(
    choice::BackendBernoulliLogitChoicePlanStep,
    step::BackendLoopPlanStep,
    observed::Vector{T},
    reference_iterable,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    eta = choice.eta
    intercept_values = _batched_numeric_scratch!(env, 1)
    intercept_gradients = _batched_backend_gradient_scratch!(cache, 1)
    if isnothing(eta.intercept)
        fill!(intercept_values, zero(T))
        fill!(intercept_gradients, zero(T))
    else
        _eval_backend_numeric_expr_and_gradient!(intercept_values, intercept_gradients, cache, env, eta.intercept, 2)
    end

    env.assigned[eta.coef_slot] ||
        throw(BatchedBackendFallback("linear predictor coefficient slot $(eta.coef_slot) is not assigned"))
    env.assigned[eta.matrix_slot] ||
        throw(BatchedBackendFallback("linear predictor covariate slot $(eta.matrix_slot) is not assigned"))

    # resolve the covariate column for each observation once (index depends only
    # on the shared iterator, not on the chain)
    observation_count = length(observed)
    columns = Vector{Int}(undef, observation_count)
    for (position, item) in enumerate(reference_iterable)
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        columns[position] = _eval_backend_index_value_expr(env, eta.index, 1)
    end

    coef_storage = env.generic_values[eta.coef_slot]
    matrix_storage = env.generic_values[eta.matrix_slot]
    seed_rows = cache.seed_rows
    coef_length = eta.coef_length
    parameter_count = size(gradients, 1)
    coefficient_accumulator = zeros(T, coef_length)
    coefficient_rows = Vector{Int}(undef, coef_length)
    for d = 1:coef_length
        row = seed_rows[eta.coef_value_index+d-1]
        row > 0 || throw(
            BatchedBackendFallback("linear predictor coefficient row $(eta.coef_value_index + d - 1) has no gradient seed row"),
        )
        coefficient_rows[d] = row
    end

    for batch_index in eachindex(totals)
        # coef/matrix live in the Any-typed generic store; the function barrier
        # specializes the hot O(observations x coef) reduction on their concrete
        # element types (otherwise every element read boxes)
        total_accumulator, intercept_derivative_sum = _glm_chain_reduce!(
            coef_storage[batch_index],
            matrix_storage[batch_index],
            columns,
            observed,
            intercept_values[batch_index],
            coefficient_accumulator,
            coef_length,
        )
        totals[batch_index] += total_accumulator
        @inbounds for parameter_index = 1:parameter_count
            gradients[parameter_index, batch_index] +=
                intercept_derivative_sum * intercept_gradients[parameter_index, batch_index]
        end
        @inbounds for d = 1:coef_length
            gradients[coefficient_rows[d], batch_index] += coefficient_accumulator[d]
        end
    end
    return totals, gradients
end

# Function-barrier hot loop for one chain: reduces the observation sum of the
# bernoulli-logit logjoint and the intercept-derivative + per-coefficient
# gradient accumulators, all in concrete element types.
function _glm_chain_reduce!(
    coef::AbstractVector,
    matrix::AbstractMatrix,
    columns::Vector{Int},
    observed::Vector{T},
    intercept::T,
    coefficient_accumulator::Vector{T},
    coef_length::Int,
) where {T<:AbstractFloat}
    fill!(coefficient_accumulator, zero(T))
    total_accumulator = zero(T)
    intercept_derivative_sum = zero(T)
    @inbounds for position in eachindex(columns, observed)
        column = columns[position]
        e = intercept
        for d = 1:coef_length
            e += T(coef[d]) * T(matrix[d, column])
        end
        y = observed[position]
        total_accumulator += _backend_bernoullilogit_logpdf(e, y)
        support = _bernoulli_value(y)
        isnothing(support) && continue
        derivative = (support ? one(T) : zero(T)) - _bernoullilogit_logistic(e)
        intercept_derivative_sum += derivative
        for d = 1:coef_length
            coefficient_accumulator[d] += derivative * T(matrix[d, column])
        end
    end
    return total_accumulator, intercept_derivative_sum
end

# Gather the loop's observed values into a vector staged on the gradient
# cache. The cache's constraints are immutable for its lifetime and the
# address depends only on the iterator, so the entry is keyed by the loop step
# and revalidated against the current iterable (a marginalize branch or
# argument-driven bound could change it between calls). The staged values are
# produced by the same normalize + convert pipeline as
# `_batched_observed_choice_values!`, so replaying them is bitwise identical
# to the per-iteration lookup.
function _gathered_observed_loop_values!(
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    step::BackendLoopPlanStep,
    choice,
    reference_iterable,
    constraints::ChoiceMap,
) where {T<:AbstractFloat}
    entry = get(cache.observed_loop_values, step, nothing)
    if entry isa Tuple{Any,Vector{T}} && entry[1] == reference_iterable
        return entry[2]
    end
    values = Vector{T}(undef, length(reference_iterable))
    for (position, item) in enumerate(reference_iterable)
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        address = _concrete_address(env, choice.address, 1)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("backend plan requires a provided value for choice $(address)"))
        values[position] = _batched_backend_observed_value(constrained_value, T)
    end
    cache.observed_loop_values[step] = (reference_iterable, values)
    return values
end

# --- loop-invariant hoisting + per-chain observation reduction (issue #141) ----

# Does a backend expression read the given slot? Conservative on anything the
# walker does not recognize.
_backend_expr_reads_slot(::BackendLiteralExpr, slot::Int) = false
_backend_expr_reads_slot(expr::BackendSlotExpr, slot::Int) = expr.slot == slot
_backend_expr_reads_slot(expr::BackendPrimitiveExpr, slot::Int) =
    any(argument -> _backend_expr_reads_slot(argument, slot), expr.arguments)
_backend_expr_reads_slot(expr::BackendTupleExpr, slot::Int) =
    any(argument -> _backend_expr_reads_slot(argument, slot), expr.arguments)
_backend_expr_reads_slot(expr::BackendBlockExpr, slot::Int) =
    any(argument -> _backend_expr_reads_slot(argument, slot), expr.arguments)
_backend_expr_reads_slot(expr, slot::Int) = true

# The loop body is a single observed choice with no binding slot, so the only
# binding that changes across iterations is the iterator: a parameter
# expression that never reads it is loop-invariant.
_backend_observed_loop_gradient_hoistable(choice, iterator_slot::Int) = false
function _backend_observed_loop_gradient_hoistable(choice::BackendNormalChoicePlanStep, iterator_slot::Int)
    return !_backend_expr_reads_slot(choice.mu, iterator_slot) &&
           !_backend_expr_reads_slot(choice.sigma, iterator_slot)
end

# Loop-invariant normal observations: evaluate mu/sigma (values and gradients)
# ONCE, reduce (sum logpdf, sum dmu, sum dsigma) over the staged observations
# per chain, then apply the parameter x chain rule once. The observed value
# gradient seed is zero, so no dvalue term appears. The per-observation terms
# are the exact `_backend_normal_logpdf` / `_accumulate_normal_gradient!`
# formulas (including NaN totals for a non-positive sigma, issue #98); only
# the summation order differs.
function _score_backend_observed_loop_gradient_hoisted!(
    choice::BackendNormalChoicePlanStep,
    observed::Vector{T},
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    observation_count = length(observed)
    half_log_2pi = T(log(2 * pi) / 2)
    @inbounds for batch_index in eachindex(totals)
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        inv_sigma = 1 / sigma
        log_sigma = sigma > zero(T) ? log(sigma) : T(NaN)
        total_accumulator = zero(T)
        dmu_accumulator = zero(T)
        dsigma_accumulator = zero(T)
        @simd for position = 1:observation_count
            z = (observed[position] - mu) * inv_sigma
            total_accumulator += -log_sigma - half_log_2pi - z * z / 2
            dmu_accumulator += z * inv_sigma
            dsigma_accumulator += (z * z - 1) * inv_sigma
        end
        totals[batch_index] += total_accumulator
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dmu_accumulator * mu_gradients[parameter_index, batch_index] +
                dsigma_accumulator * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

# --- sufficient-statistics fusion (issue #146) ---------------------------------
#
# The fused tier scores the WHOLE loop per chain from a handful of numbers
# computed once per (step, iterable) at first use and staged on the gradient
# cache (the cache's constraints are fixed for its lifetime, like the staged
# observation vector). The closed forms below are the per-observation
# `_accumulate_*_gradient!` formulas summed analytically, so they agree with
# the per-iteration tiers to reassociation (tolerance, not bitwise); the
# out-of-support parameter behavior matches (NaN totals for a non-positive
# normal sigma / exponential rate, issue #98; ArgumentError for a non-positive
# poisson lambda). Data the closed form cannot represent exactly (non-finite
# values, negative exponential values, non-count poisson values, an empty
# loop) is detected once at staging and recorded as `nothing`, so the walk
# falls back to the hoisted/per-iteration tiers without rescanning per call.

struct _NormalObservationStats{T<:AbstractFloat}
    count::Int
    mean::T
    # compensation term sum(y - ybar): zero in exact arithmetic but not in
    # floats (ybar carries ~1 ulp of the DATA scale, which n amplifies when
    # |ybar| >> sigma); carrying it makes the moment identities below exact
    # algebra about the stored float mean
    residual::T
    # centered second moment sum((y - ybar)^2): with delta = ybar - mu,
    #   sum(y - mu)   = n*delta + residual
    #   sum((y-mu)^2) = S2c + delta*(2*residual + n*delta)
    # so the fusion is cancellation-free even when |ybar| >> sigma, unlike
    # the naive power-sum form
    centered_sum_squares::T
end

struct _ExponentialObservationStats{T<:AbstractFloat}
    count::Int
    total::T
end

struct _PoissonObservationStats{T<:AbstractFloat}
    count::Int
    total::T
    # sum(log y!) is a data constant, computed at staging with the same
    # `_logfactorial_like` accumulation the per-observation logpdf uses
    log_factorial_total::T
end

# Fusable = every parameter expression is loop-invariant. Families without a
# fused scorer report false and keep their existing tiers.
_backend_observed_loop_gradient_fusable(choice, iterator_slot::Int) = false
_backend_observed_loop_gradient_fusable(choice::BackendNormalChoicePlanStep, iterator_slot::Int) =
    !_backend_expr_reads_slot(choice.mu, iterator_slot) && !_backend_expr_reads_slot(choice.sigma, iterator_slot)
_backend_observed_loop_gradient_fusable(choice::BackendExponentialChoicePlanStep, iterator_slot::Int) =
    !_backend_expr_reads_slot(choice.rate, iterator_slot)
_backend_observed_loop_gradient_fusable(choice::BackendPoissonChoicePlanStep, iterator_slot::Int) =
    !_backend_expr_reads_slot(choice.lambda, iterator_slot)

function _observed_loop_suffstats!(
    cache::BatchedBackendGradientCache,
    step::BackendLoopPlanStep,
    choice,
    observed::Vector{T},
    reference_iterable,
) where {T<:AbstractFloat}
    _backend_observed_loop_gradient_fusable(choice, step.iterator_slot) || return nothing
    entry = get(cache.observed_loop_stats, step, nothing)
    if entry isa Tuple{Any,Any} && entry[1] == reference_iterable
        return entry[2]
    end
    stats = _observed_loop_compute_suffstats(choice, observed)
    cache.observed_loop_stats[step] = (reference_iterable, stats)
    return stats
end

function _observed_loop_compute_suffstats(::BackendNormalChoicePlanStep, observed::Vector{T}) where {T<:AbstractFloat}
    count = length(observed)
    (count > 0 && all(isfinite, observed)) || return nothing
    mean = sum(observed) / count
    residual = sum(value -> value - mean, observed)
    centered_sum_squares = sum(value -> abs2(value - mean), observed)
    return _NormalObservationStats{T}(count, mean, residual, centered_sum_squares)
end

function _observed_loop_compute_suffstats(::BackendExponentialChoicePlanStep, observed::Vector{T}) where {T<:AbstractFloat}
    count = length(observed)
    count > 0 || return nothing
    # a negative observation scores -Inf and contributes no gradient in the
    # per-observation form; keep those loops on the unfused tiers
    all(value -> isfinite(value) && value >= zero(T), observed) || return nothing
    return _ExponentialObservationStats{T}(count, sum(observed))
end

function _observed_loop_compute_suffstats(::BackendPoissonChoicePlanStep, observed::Vector{T}) where {T<:AbstractFloat}
    count = length(observed)
    count > 0 || return nothing
    total = zero(T)
    log_factorial_total = zero(T)
    for value in observed
        observation_count = _poisson_count(value)
        # a non-count observation scores -Inf and contributes no gradient in
        # the per-observation form; keep those loops on the unfused tiers
        isnothing(observation_count) && return nothing
        total += T(observation_count)
        log_factorial_total += _logfactorial_like(one(T), observation_count)
    end
    return _PoissonObservationStats{T}(count, total, log_factorial_total)
end

# Fused normal: per chain, with delta = ybar - mu and the residual-compensated
# moments above (sum(z) = (n*delta + residual)/sigma, sum(z^2) =
# (S2c + delta*(2*residual + n*delta))/sigma^2)
#   logpdf sum = -n*(log sigma + log(2*pi)/2) - sum(z^2)/2
#   d/dmu     = sum(z)/sigma
#   d/dsigma  = (sum(z^2) - n) / sigma
# Scratch depths mirror the hoisted tier.
function _score_backend_observed_loop_gradient_fused!(
    choice::BackendNormalChoicePlanStep,
    stats::_NormalObservationStats{T},
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    observation_count = T(stats.count)
    half_log_2pi = T(log(2 * pi) / 2)
    @inbounds for batch_index in eachindex(totals)
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        inv_sigma = 1 / sigma
        log_sigma = sigma > zero(T) ? log(sigma) : T(NaN)
        delta = stats.mean - mu
        deviation_sum = observation_count * delta + stats.residual
        squared_z_sum =
            (stats.centered_sum_squares + delta * (2 * stats.residual + observation_count * delta)) *
            inv_sigma *
            inv_sigma
        totals[batch_index] += -observation_count * (log_sigma + half_log_2pi) - squared_z_sum / 2
        dmu = deviation_sum * inv_sigma * inv_sigma
        dsigma = (squared_z_sum - observation_count) * inv_sigma
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

# Fused exponential: per chain
#   logpdf sum = n*log(rate) - rate*sum(y)
#   d/drate    = n/rate - sum(y)
# A non-positive rate scores NaN per observation (issue #98), so the fused
# total is NaN as well.
function _score_backend_observed_loop_gradient_fused!(
    choice::BackendExponentialChoicePlanStep,
    stats::_ExponentialObservationStats{T},
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    rate_values = _batched_numeric_scratch!(env, 1)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 2)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, choice.rate, 3)
    observation_count = T(stats.count)
    @inbounds for batch_index in eachindex(totals)
        rate = rate_values[batch_index]
        log_rate = rate > zero(T) ? log(rate) : T(NaN)
        totals[batch_index] += observation_count * log_rate - rate * stats.total
        drate = observation_count / rate - stats.total
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += drate * rate_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

# Fused poisson: per chain
#   logpdf sum = sum(y)*log(lambda) - n*lambda - sum(log y!)
#   d/dlambda  = sum(y)/lambda - n
# A non-positive lambda throws, matching `_backend_poisson_logpdf`.
function _score_backend_observed_loop_gradient_fused!(
    choice::BackendPoissonChoicePlanStep,
    stats::_PoissonObservationStats{T},
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    lambda_values = _batched_numeric_scratch!(env, 1)
    lambda_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(lambda_values, lambda_gradients, cache, env, choice.lambda, 2)
    observation_count = T(stats.count)
    @inbounds for batch_index in eachindex(totals)
        lambda = lambda_values[batch_index]
        lambda > zero(T) || throw(ArgumentError("poisson requires lambda > 0"))
        totals[batch_index] += stats.total * log(lambda) - observation_count * lambda - stats.log_factorial_total
        dlambda = stats.total / lambda - observation_count
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += dlambda * lambda_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

# --- per-family per-iteration accumulate ---------------------------------------
#
# Each method mirrors its `_score_backend_step_and_gradient!` counterpart with
# the same scratch depths, minus the per-chain value gather (the driver has
# already filled `env.observed_values`) and the binding-slot assignment (the
# loop-observed contract guarantees `binding_slot === nothing`). The
# value-gradient seed at gradient scratch depth 1 is pre-zeroed by the driver.

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    return _accumulate_normal_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendLognormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    return _accumulate_lognormal_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendExponentialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    rate_values = _batched_numeric_scratch!(env, 1)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 2)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, choice.rate, 3)
    return _accumulate_exponential_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        rate_values,
        rate_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendGammaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    rate_values = _batched_numeric_scratch!(env, 2)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, choice.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, choice.rate, 5)
    return _accumulate_gamma_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        shape_values,
        shape_gradients,
        rate_values,
        rate_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendInverseGammaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, choice.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_inversegamma_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        shape_values,
        shape_gradients,
        scale_values,
        scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendWeibullChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, choice.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_weibull_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        shape_values,
        shape_gradients,
        scale_values,
        scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendBetaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    alpha_values = _batched_numeric_scratch!(env, 1)
    alpha_gradients = _batched_backend_gradient_scratch!(cache, 2)
    beta_values = _batched_numeric_scratch!(env, 2)
    beta_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(alpha_values, alpha_gradients, cache, env, choice.alpha, 4)
    _eval_backend_numeric_expr_and_gradient!(beta_values, beta_gradients, cache, env, choice.beta, 5)
    return _accumulate_beta_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        alpha_values,
        alpha_gradients,
        beta_values,
        beta_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendStudentTChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    nu_values = _batched_numeric_scratch!(env, 1)
    nu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    mu_values = _batched_numeric_scratch!(env, 2)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 3)
    sigma_values = _batched_numeric_scratch!(env, 3)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 4)
    _eval_backend_numeric_expr_and_gradient!(nu_values, nu_gradients, cache, env, choice.nu, 5)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 6)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 7)
    return _accumulate_studentt_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        nu_values,
        nu_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendLaplaceChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_laplace_gradient!(
        totals,
        gradients,
        env.observed_values,
        value_gradients,
        mu_values,
        mu_gradients,
        scale_values,
        scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendCauchyChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 5)
    return _accumulate_cauchy_gradient!(
        totals, gradients, env.observed_values, value_gradients, mu_values, mu_gradients, sigma_values, sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendHalfNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    sigma_values = _batched_numeric_scratch!(env, 1)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 2)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, choice.sigma, 3)
    return _accumulate_halfnormal_gradient!(
        totals, gradients, env.observed_values, value_gradients, sigma_values, sigma_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendHalfCauchyChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    scale_values = _batched_numeric_scratch!(env, 1)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 2)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 3)
    return _accumulate_halfcauchy_gradient!(
        totals, gradients, env.observed_values, value_gradients, scale_values, scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendUniformChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    lower_values = _batched_numeric_scratch!(env, 1)
    lower_gradients = _batched_backend_gradient_scratch!(cache, 2)
    upper_values = _batched_numeric_scratch!(env, 2)
    upper_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(lower_values, lower_gradients, cache, env, choice.lower, 4)
    _eval_backend_numeric_expr_and_gradient!(upper_values, upper_gradients, cache, env, choice.upper, 5)
    return _accumulate_uniform_gradient!(
        totals, gradients, env.observed_values, value_gradients, lower_values, lower_gradients, upper_values, upper_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendLogisticChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_logistic_gradient!(
        totals, gradients, env.observed_values, value_gradients, mu_values, mu_gradients, scale_values, scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendGumbelChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, choice.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, choice.scale, 5)
    return _accumulate_gumbel_gradient!(
        totals, gradients, env.observed_values, value_gradients, mu_values, mu_gradients, scale_values, scale_gradients,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendBernoulliChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 2)
    return _accumulate_bernoulli_gradient!(totals, gradients, probability_values, probability_gradients, env.observed_values)
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendBernoulliLogitChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    eta_values = _batched_numeric_scratch!(env, 1)
    eta_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(eta_values, eta_gradients, cache, env, choice.eta, 2)
    return _accumulate_bernoullilogit_gradient!(totals, gradients, eta_values, eta_gradients, env.observed_values)
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendPoissonChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    lambda_values = _batched_numeric_scratch!(env, 1)
    lambda_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(lambda_values, lambda_gradients, cache, env, choice.lambda, 2)
    return _accumulate_poisson_gradient!(totals, gradients, lambda_values, lambda_gradients, env.observed_values)
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendGeometricChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 2)
    return _accumulate_geometric_gradient!(totals, gradients, probability_values, probability_gradients, env.observed_values)
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    trials_values = _batched_index_scratch!(env, 1)
    probability_values = _batched_numeric_scratch!(env, 1)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 1)
    _eval_backend_index_value_expr!(trials_values, env, choice.trials, 2)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 2)
    return _accumulate_binomial_gradient!(
        totals,
        gradients,
        trials_values,
        probability_values,
        probability_gradients,
        env.observed_values,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendNegativeBinomialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    successes_values = _batched_numeric_scratch!(env, 1)
    successes_gradients = _batched_backend_gradient_scratch!(cache, 1)
    probability_values = _batched_numeric_scratch!(env, 2)
    probability_gradients = _batched_backend_gradient_scratch!(cache, 2)
    _eval_backend_numeric_expr_and_gradient!(successes_values, successes_gradients, cache, env, choice.successes, 3)
    _eval_backend_numeric_expr_and_gradient!(probability_values, probability_gradients, cache, env, choice.probability, 4)
    return _accumulate_negativebinomial_gradient!(
        totals,
        gradients,
        successes_values,
        successes_gradients,
        probability_values,
        probability_gradients,
        env.observed_values,
    )
end

function _score_backend_observed_loop_choice_gradient!(
    choice::BackendCategoricalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
) where {T<:AbstractFloat}
    probability_values = ntuple(index -> _batched_numeric_scratch!(env, index), length(choice.probabilities))
    probability_gradients = ntuple(index -> _batched_backend_gradient_scratch!(cache, index), length(choice.probabilities))
    for (index, probability) in enumerate(choice.probabilities)
        _eval_backend_numeric_expr_and_gradient!(
            probability_values[index],
            probability_gradients[index],
            cache,
            env,
            probability,
            length(choice.probabilities) + index,
        )
    end
    return _accumulate_categorical_gradient!(totals, gradients, probability_values, probability_gradients, env.observed_values)
end
