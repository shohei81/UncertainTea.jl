# Backend-native scalar and batched scoring: structured families (mvnormal, mvnormaldense, dirichlet, mixture, broadcast/iid vector machinery).

function _backend_mvnormal_observed_value(value, expected_length::Int, ::Type{T}=Float64) where {T<:Real}
    values = value isa Tuple ? collect(value) : value
    values isa AbstractVector || throw(ArgumentError("mvnormal expects a vector or tuple value"))
    length(values) == expected_length ||
        throw(ArgumentError("mvnormal expects a value of length $expected_length, got $(length(values))"))
    return T[convert(T, float(item)) for item in values]
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment,
    slot::Int,
    values::AbstractMatrix,
)
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index = 1:env.batch_size
        storage[batch_index] = collect(view(values, :, batch_index))
    end
    env.assigned[slot] = true
    return env
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment,
    slot::Int,
    values::AbstractVector{<:AbstractVector},
)
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index = 1:env.batch_size
        storage[batch_index] = [component_values[batch_index] for component_values in values]
    end
    env.assigned[slot] = true
    return env
end

function _backend_mvnormal_logpdf(mu::Tuple, sigma::Tuple, x)
    (length(mu) == length(sigma) && length(mu) == length(x)) ||
        throw(ArgumentError("mvnormal requires matching mean, scale, and value lengths"))
    total = _backend_normal_logpdf(mu[1], sigma[1], x[1])
    for index = 2:length(mu)
        total += _backend_normal_logpdf(mu[index], sigma[index], x[index])
    end
    return total
end

function _score_backend_step!(
    step::BackendMvNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    mu = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.mu)...)
    sigma = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.sigma)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mvnormal_logpdf(mu, sigma, value)
end

function _score_backend_step!(
    step::BackendMvNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)

    mu_values = [_batched_numeric_scratch!(env, step.value_length + index) for index = 1:step.value_length]
    sigma_values = [_batched_numeric_scratch!(env, 2 * step.value_length + index) for index = 1:step.value_length]
    for component_index = 1:step.value_length
        _eval_backend_numeric_expr!(mu_values[component_index], env, step.mu[component_index], 3 * step.value_length + 1)
        _eval_backend_numeric_expr!(sigma_values[component_index], env, step.sigma[component_index], 3 * step.value_length + 2)
        for batch_index = 1:env.batch_size
            totals[batch_index] += _backend_normal_logpdf(
                mu_values[component_index][batch_index],
                sigma_values[component_index][batch_index],
                choice_values[component_index][batch_index],
            )
        end
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end

# --- diagonal / dense multivariate Student-t ----------------------------------

# Shared multivariate-t log density from the Cholesky log-determinant
# `log_det = sum_i log(scale diagonal)` and the Mahalanobis quadratic `q`.
function _backend_mvstudentt_density(nu, d::Int, log_det, quadratic)
    half = (nu + d) / 2
    return loggamma(half) - loggamma(nu / 2) - (d / 2) * log(nu * float(pi)) - log_det -
           half * log1p(quadratic / nu)
end

function _backend_mvstudentt_logpdf(nu, mu::Tuple, sigma::Tuple, x)
    d = length(mu)
    (length(sigma) == d && length(x) == d) ||
        throw(ArgumentError("mvstudentt requires matching mean, scale, and value lengths"))
    z1 = (x[1] - mu[1]) / sigma[1]
    quadratic = z1 * z1
    log_det = log(sigma[1])
    for index = 2:d
        z = (x[index] - mu[index]) / sigma[index]
        quadratic += z * z
        log_det += log(sigma[index])
    end
    return _backend_mvstudentt_density(nu, d, log_det, quadratic)
end

function _backend_mvstudenttdense_logpdf(nu, mu, Lmat, x)
    d = length(mu)
    (length(x) == d && size(Lmat, 1) == d) || return oftype(float(x[1]), -Inf)
    z1 = (x[1] - mu[1]) / Lmat[1, 1]
    solved = Vector{typeof(z1)}(undef, d)
    solved[1] = z1
    log_det = log(Lmat[1, 1])
    quadratic = z1 * z1
    for row = 2:d
        residual = x[row] - mu[row]
        for col = 1:(row-1)
            residual -= Lmat[row, col] * solved[col]
        end
        z = residual / Lmat[row, row]
        solved[row] = z
        log_det += log(Lmat[row, row])
        quadratic += z * z
    end
    return _backend_mvstudentt_density(nu, d, log_det, quadratic)
end

function _score_backend_step!(
    step::BackendMvStudentTChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    nu = _eval_backend_numeric_expr(env, step.nu)
    mu = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.mu)...)
    sigma = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.sigma)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mvstudentt_logpdf(nu, mu, sigma, value)
end

function _score_backend_step!(
    step::BackendMvStudentTChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:d]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)
    mu_values = [_batched_numeric_scratch!(env, d + index) for index = 1:d]
    sigma_values = [_batched_numeric_scratch!(env, 2 * d + index) for index = 1:d]
    nu_values = _batched_numeric_scratch!(env, 3 * d + 1)
    _eval_backend_numeric_expr!(nu_values, env, step.nu, 3 * d + 2)
    for index = 1:d
        _eval_backend_numeric_expr!(mu_values[index], env, step.mu[index], 3 * d + 3)
        _eval_backend_numeric_expr!(sigma_values[index], env, step.sigma[index], 3 * d + 4)
    end
    for batch_index = 1:env.batch_size
        mu = ntuple(index -> mu_values[index][batch_index], d)
        sigma = ntuple(index -> sigma_values[index][batch_index], d)
        x = ntuple(index -> choice_values[index][batch_index], d)
        totals[batch_index] += _backend_mvstudentt_logpdf(nu_values[batch_index], mu, sigma, x)
    end
    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end

function _score_backend_step!(
    step::BackendMvStudentTDenseChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    d = step.value_length
    nu = _eval_backend_numeric_expr(env, step.nu)
    mu = [_eval_backend_numeric_expr(env, expr) for expr in step.mu]
    Lmat = _backend_mvnormaldense_scale_matrix(_environment_value(env, step.scale_tril.slot), d)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mvstudenttdense_logpdf(nu, mu, Lmat, value)
end

function _score_backend_step!(
    step::BackendMvStudentTDenseChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:d]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)
    mu_values = [_batched_numeric_scratch!(env, d + index) for index = 1:d]
    nu_values = _batched_numeric_scratch!(env, 2 * d + 1)
    _eval_backend_numeric_expr!(nu_values, env, step.nu, 2 * d + 2)
    for index = 1:d
        _eval_backend_numeric_expr!(mu_values[index], env, step.mu[index], 2 * d + 3)
    end
    env.assigned[step.scale_tril.slot] ||
        throw(BatchedBackendFallback("mvstudenttdense scale_tril slot $(step.scale_tril.slot) is not assigned"))
    scale_storage = env.generic_values[step.scale_tril.slot]
    for batch_index = 1:env.batch_size
        mu = ntuple(index -> mu_values[index][batch_index], d)
        Lmat = _backend_mvnormaldense_scale_matrix(scale_storage[batch_index], d)
        x = ntuple(index -> choice_values[index][batch_index], d)
        totals[batch_index] += _backend_mvstudenttdense_logpdf(nu_values[batch_index], mu, Lmat, x)
    end
    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end

# --- mixture of normal components ---------------------------------------------

function _backend_mixture_normal_logpdf(weights, mus, sigmas, x)
    k = length(weights)
    (length(mus) == k && length(sigmas) == k) ||
        throw(ArgumentError("mixture requires one weight per normal component"))
    # same weight validation as the CPU `MixtureDist` constructor (issue #76)
    total = zero(float(first(weights)))
    for index = 1:k
        w = weights[index]
        w >= zero(w) || throw(ArgumentError("mixture weights must be nonnegative"))
        total += w
    end
    abs(total - one(total)) <= oftype(total, 1e-8) ||
        throw(ArgumentError("mixture weights must sum to 1 (within 1e-8)"))
    xf = float(x)
    T = typeof(xf + float(weights[1]) + float(mus[1]) + float(sigmas[1]))
    m = T(-Inf)
    for index = 1:k
        term = log(weights[index]) + _backend_normal_logpdf(mus[index], sigmas[index], xf)
        term > m && (m = term)
    end
    isfinite(m) || return T(-Inf)
    total = zero(T)
    for index = 1:k
        term = log(weights[index]) + _backend_normal_logpdf(mus[index], sigmas[index], xf)
        total += exp(term - m)
    end
    return m + log(total)
end

function _score_backend_step!(
    step::BackendMixtureNormalChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_value(step.parameter_slot, params, constraints, address)
    weights = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.weights)...)
    mus = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.mus)...)
    sigmas = tuple((_eval_backend_numeric_expr(env, expr) for expr in step.sigmas)...)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mixture_normal_logpdf(weights, mus, sigmas, value)
end

function _score_backend_step!(
    step::BackendMixtureNormalChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    k = length(step.weights)
    choice_values = env.observed_values
    weight_values = [_batched_numeric_scratch!(env, index) for index = 1:k]
    mu_values = [_batched_numeric_scratch!(env, k + index) for index = 1:k]
    sigma_values = [_batched_numeric_scratch!(env, 2 * k + index) for index = 1:k]
    for index = 1:k
        _eval_backend_numeric_expr!(weight_values[index], env, step.weights[index], 3 * k + 1)
        _eval_backend_numeric_expr!(mu_values[index], env, step.mus[index], 3 * k + 2)
        _eval_backend_numeric_expr!(sigma_values[index], env, step.sigmas[index], 3 * k + 3)
    end
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(choice_values, step.parameter_slot, params, constraints, address_parts)
    for batch_index = 1:env.batch_size
        value = choice_values[batch_index]
        weights = ntuple(index -> weight_values[index][batch_index], k)
        mus = ntuple(index -> mu_values[index][batch_index], k)
        sigmas = ntuple(index -> sigma_values[index][batch_index], k)
        totals[batch_index] += _backend_mixture_normal_logpdf(weights, mus, sigmas, value)
        if !isnothing(step.binding_slot)
            if env.numeric_slots[step.binding_slot]
                env.numeric_values[step.binding_slot, batch_index] = convert(eltype(env.numeric_values), value)
            elseif env.index_slots[step.binding_slot]
                env.index_values[step.binding_slot, batch_index] = Int(value)
            else
                env.generic_values[step.binding_slot][batch_index] = value
            end
        end
    end
    isnothing(step.binding_slot) || (env.assigned[step.binding_slot] = true)
    return totals
end

# --- dense-covariance multivariate normal -------------------------------------

# Log density of mvnormaldense via forward substitution solving L z = x - mu,
# reading only the lower triangle of the d x d factor `Lmat`. Matches the CPU
# `MvNormalDenseDist` reference (returns -Inf on a length mismatch).
function _backend_mvnormaldense_logpdf(mu, Lmat, x)
    d = length(mu)
    (length(x) == d && size(Lmat, 1) == d) || return oftype(float(x[1]), -Inf)
    z1 = (x[1] - mu[1]) / Lmat[1, 1]
    solved = Vector{typeof(z1)}(undef, d)
    solved[1] = z1
    log_det = log(Lmat[1, 1])
    quadratic = z1 * z1
    for row = 2:d
        residual = x[row] - mu[row]
        for col = 1:(row-1)
            residual -= Lmat[row, col] * solved[col]
        end
        z = residual / Lmat[row, row]
        solved[row] = z
        log_det += log(Lmat[row, row])
        quadratic += z * z
    end
    return -log_det - quadratic / 2 - d * log(2 * pi) / 2
end

# Read the whole scale_tril matrix from the generic slot the argument expression
# resolves to, validating it is a d x d matrix.
function _backend_mvnormaldense_scale_matrix(matrix, d::Int)
    matrix isa AbstractMatrix ||
        throw(ArgumentError("mvnormaldense scale_tril must evaluate to a matrix, got $(typeof(matrix))"))
    (size(matrix, 1) == d && size(matrix, 2) == d) ||
        throw(ArgumentError("mvnormaldense scale_tril must be $(d)x$(d), got $(size(matrix))"))
    return matrix
end

function _score_backend_step!(
    step::BackendMvNormalDenseChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    address = _concrete_address(env, step.address)
    value = _backend_choice_vector_value(step.value_index, step.value_length, params, constraints, address)
    d = step.value_length
    mu = [_eval_backend_numeric_expr(env, expr) for expr in step.mu]
    Lmat = _backend_mvnormaldense_scale_matrix(_environment_value(env, step.scale_tril.slot), d)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return _backend_mvnormaldense_logpdf(mu, Lmat, value)
end

function _score_backend_step!(
    step::BackendMvNormalDenseChoicePlanStep,
    totals::AbstractVector,
    env::BatchedPlanEnvironment,
    params::AbstractMatrix,
    constraints,
)
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:d]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)

    mu_values = [_batched_numeric_scratch!(env, d + index) for index = 1:d]
    for index = 1:d
        _eval_backend_numeric_expr!(mu_values[index], env, step.mu[index], 2 * d + 1)
    end

    env.assigned[step.scale_tril.slot] ||
        throw(BatchedBackendFallback("mvnormaldense scale_tril slot $(step.scale_tril.slot) is not assigned"))
    scale_storage = env.generic_values[step.scale_tril.slot]
    for batch_index = 1:env.batch_size
        mu = ntuple(index -> mu_values[index][batch_index], d)
        Lmat = _backend_mvnormaldense_scale_matrix(scale_storage[batch_index], d)
        x = ntuple(index -> choice_values[index][batch_index], d)
        totals[batch_index] += _backend_mvnormaldense_logpdf(mu, Lmat, x)
    end

    isnothing(step.binding_slot) || _assign_backend_choice_vector_value!(env, step.binding_slot, choice_values)
    return totals
end
