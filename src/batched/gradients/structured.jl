# Hand-derived analytic batched logjoint gradients: structured families (mvnormal, mvnormaldense, dirichlet, lkjcholesky, mixture, broadcast/iid vector machinery).

function _accumulate_mvnormal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{<:AbstractVector},
    value_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    sigma_values::AbstractVector{<:AbstractVector},
    sigma_gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        for component_index in eachindex(value_values)
            value = value_values[component_index][batch_index]
            mu = mu_values[component_index][batch_index]
            sigma = sigma_values[component_index][batch_index]
            totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
            z = (value - mu) / sigma
            inv_sigma = 1 / sigma
            dvalue = -z * inv_sigma
            dmu = z * inv_sigma
            dsigma = (z * z - 1) * inv_sigma
            component_value_gradients = value_gradients[component_index]
            component_mu_gradients = mu_gradients[component_index]
            component_sigma_gradients = sigma_gradients[component_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    dvalue * component_value_gradients[parameter_index, batch_index] +
                    dmu * component_mu_gradients[parameter_index, batch_index] +
                    dsigma * component_sigma_gradients[parameter_index, batch_index]
            end
        end
    end
    return totals, gradients
end

function _accumulate_dirichlet_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    parameter_index::Union{Nothing,Int},
    value_values::AbstractVector{<:AbstractVector},
    alpha_values::AbstractVector{<:AbstractVector},
    alpha_gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    value_length = length(value_values)
    choice_gradients = Vector{T}(undef, value_length)
    alpha_derivatives = Vector{T}(undef, value_length)
    for batch_index in eachindex(totals)
        total_alpha = zero(T)
        accumulator = zero(T)
        for component_index = 1:value_length
            alpha = alpha_values[component_index][batch_index]
            alpha > 0 || throw(ArgumentError("dirichlet requires alpha > 0 in every dimension"))
            total_alpha += alpha
            accumulator -= loggamma(alpha)
        end
        accumulator += loggamma(total_alpha)

        total = zero(T)
        valid = true
        weighted_choice_gradient = zero(T)
        for component_index = 1:value_length
            value = value_values[component_index][batch_index]
            value > 0 || begin
                valid = false
                break
            end
            total += value
            alpha = alpha_values[component_index][batch_index]
            accumulator += (alpha - 1) * log(value)
            choice_gradient = (alpha - 1) / value
            choice_gradients[component_index] = choice_gradient
            weighted_choice_gradient += choice_gradient * value
            alpha_derivatives[component_index] = digamma(total_alpha) - digamma(alpha) + log(value)
        end

        if !valid || abs(total - 1) > sqrt(eps(total)) * value_length * 16
            totals[batch_index] += -T(Inf)
            continue
        end

        totals[batch_index] += accumulator
        for component_index = 1:value_length
            alpha_derivative = alpha_derivatives[component_index]
            component_alpha_gradients = alpha_gradients[component_index]
            for parameter_row in axes(gradients, 1)
                gradients[parameter_row, batch_index] +=
                    alpha_derivative * component_alpha_gradients[parameter_row, batch_index]
            end
        end
        isnothing(parameter_index) && continue
        for component_index = 1:(value_length-1)
            unconstrained_row = parameter_index + component_index - 1
            constrained_value = value_values[component_index][batch_index]
            gradients[unconstrained_row, batch_index] +=
                constrained_value * (choice_gradients[component_index] - weighted_choice_gradient)
        end
    end
    return totals, gradients
end

# The LKJ density over the packed factor is sum_i (d - i + 2 eta - 2) log L[i,i]
# with L[i,i] = prod_{k<i} sqrt(1 - w_ik^2) under the tanh/stick construction, so
# d(logpdf)/dz_ij = -(d - i + 2 eta - 2) * w_ij with w_ij = L[i,j] / sqrt(1 -
# sum_{k<j} L[i,k]^2) recovered from the constrained values (w = tanh(z)). The
# transform's own log-abs-det gradient is added by the CholeskyCorrTransform
# post-pass in gradients/core.jl, mirroring the dirichlet split.
function _accumulate_lkjcholesky_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    d::Int,
    parameter_index::Union{Nothing,Int},
    value_values::AbstractVector{<:AbstractVector},
    eta_values::AbstractVector,
    eta_gradients::AbstractMatrix,
) where {T<:AbstractFloat}
    # precision-scaled support tolerance: see the batched scorer in
    # scoring_vectors.jl (Float32-constructed rows are unit only to f32 rounding)
    tolerance = sqrt(eps(T)) * d * 16
    for batch_index in eachindex(totals)
        eta = eta_values[batch_index]
        eta > 0 || throw(ArgumentError("lkjcholesky requires a concentration eta > 0"))
        accumulator = T(_lkj_log_normalizing_constant(d, eta))
        eta_derivative = T(_lkj_log_normalizing_constant_deta(d, eta))
        valid = true
        for row = 1:d
            diagonal = value_values[_packed_lower_index(d, row, row)][batch_index]
            diagonal > 0 || begin
                valid = false
                break
            end
            sum_sqs = zero(T)
            for col = 1:row
                entry = value_values[_packed_lower_index(d, row, col)][batch_index]
                sum_sqs += entry * entry
            end
            abs(sum_sqs - one(T)) <= tolerance || begin
                valid = false
                break
            end
            if row >= 2
                log_diagonal = log(diagonal)
                accumulator += (d - row + 2 * eta - 2) * log_diagonal
                eta_derivative += 2 * log_diagonal
            end
        end
        if !valid
            totals[batch_index] += -T(Inf)
            continue
        end

        totals[batch_index] += accumulator
        for parameter_row in axes(gradients, 1)
            gradients[parameter_row, batch_index] += eta_derivative * eta_gradients[parameter_row, batch_index]
        end
        isnothing(parameter_index) && continue
        position = 0
        for row = 2:d
            remaining = one(T)
            for col = 1:(row-1)
                entry = value_values[_packed_lower_index(d, row, col)][batch_index]
                if remaining > 0
                    w = entry / sqrt(remaining)
                    gradients[parameter_index+position, batch_index] -= (d - row + 2 * eta - 2) * w
                end
                position += 1
                remaining -= entry * entry
            end
        end
    end
    return totals, gradients
end

function _assign_backend_choice_vector_value!(
    env::BatchedPlanEnvironment{T},
    slot_gradients::Array{T,3},
    slot::Int,
    values::AbstractVector{<:AbstractVector},
    gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    env.generic_slots[slot] || throw(BatchedBackendFallback("mvnormal backend binding slot $slot must be generic"))
    storage = env.generic_values[slot]
    for batch_index = 1:env.batch_size
        storage[batch_index] = [component_values[batch_index] for component_values in values]
    end
    env.assigned[slot] = true
    return env
end

function _score_backend_step_and_gradient!(
    step::BackendMvNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:step.value_length]
    choice_gradients = [_batched_backend_gradient_scratch!(cache, index) for index = 1:step.value_length]
    mu_values = [_batched_numeric_scratch!(env, step.value_length + index) for index = 1:step.value_length]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, step.value_length + index) for index = 1:step.value_length]
    sigma_values = [_batched_numeric_scratch!(env, 2 * step.value_length + index) for index = 1:step.value_length]
    sigma_gradients = [_batched_backend_gradient_scratch!(cache, 2 * step.value_length + index) for index = 1:step.value_length]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index = 1:step.value_length
        _fill_choice_vector_gradient!(choice_gradients[component_index], step.value_index, component_index, cache.seed_rows)
        _eval_backend_numeric_expr_and_gradient!(
            mu_values[component_index],
            mu_gradients[component_index],
            cache,
            env,
            step.mu[component_index],
            3 * step.value_length + 1,
        )
        _eval_backend_numeric_expr_and_gradient!(
            sigma_values[component_index],
            sigma_gradients[component_index],
            cache,
            env,
            step.sigma[component_index],
            3 * step.value_length + 2,
        )
    end

    _accumulate_mvnormal_gradient!(
        totals,
        gradients,
        choice_values,
        choice_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, choice_gradients)
    return totals, gradients
end

# Element-aware value+gradient evaluation of a broadcast argument for a fixed data
# element, reusing the scalar numeric-gradient primitive kernels. Vector generic slots
# contribute their element and carry zero gradient (data); numeric latents carry their
# accumulated slot gradient.
function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendLiteralExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    fill!(values, T(_require_numeric_value(env, expr.value, "batched broadcast argument")))
    fill!(gradients, zero(T))
    return values, gradients
end

function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendSlotExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    env.assigned[expr.slot] || throw(BatchedBackendFallback("environment slot $(expr.slot) is not assigned"))
    if env.numeric_slots[expr.slot]
        copyto!(values, view(env.numeric_values, expr.slot, :))
        _copy_slot_gradient!(gradients, cache.slot_gradients, expr.slot)
        return values, gradients
    elseif env.index_slots[expr.slot]
        for batch_index in eachindex(values)
            values[batch_index] = T(env.index_values[expr.slot, batch_index])
        end
        fill!(gradients, zero(T))
        return values, gradients
    end
    storage = env.generic_values[expr.slot]
    for batch_index in eachindex(values)
        value = storage[batch_index]
        scalar = value isa AbstractVector ? value[element_index] : value
        values[batch_index] = T(_require_numeric_value(env, scalar, "batched broadcast slot"))
    end
    fill!(gradients, zero(T))
    return values, gradients
end

function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendPrimitiveExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    expr.op in BACKEND_GRADIENT_SUPPORTED_PRIMITIVES ||
        _backend_numeric_error(env, "batched broadcast gradient does not support primitive `$(expr.op)`")
    isempty(expr.arguments) && _backend_numeric_error(env, "batched broadcast gradient primitive requires arguments")

    _eval_backend_broadcast_numeric_and_gradient!(values, gradients, cache, env, first(expr.arguments), element_index, depth + 1)
    if length(expr.arguments) == 1
        return _apply_backend_numeric_gradient_unary!(values, gradients, env, expr.op)
    elseif expr.op === :clamp
        length(expr.arguments) == 3 ||
            _backend_numeric_error(env, "batched broadcast gradient `clamp` expects exactly 3 arguments")
        middle_values = _batched_numeric_scratch!(env, depth)
        middle_gradients = _batched_backend_gradient_scratch!(cache, depth)
        rhs_values = _batched_numeric_scratch!(env, depth + 1)
        rhs_gradients = _batched_backend_gradient_scratch!(cache, depth + 1)
        _eval_backend_broadcast_numeric_and_gradient!(
            middle_values,
            middle_gradients,
            cache,
            env,
            expr.arguments[2],
            element_index,
            depth + 2,
        )
        _eval_backend_broadcast_numeric_and_gradient!(
            rhs_values,
            rhs_gradients,
            cache,
            env,
            expr.arguments[3],
            element_index,
            depth + 2,
        )
        return _apply_backend_numeric_gradient_ternary!(
            values, gradients, env, expr.op, middle_values, middle_gradients, rhs_values, rhs_gradients,
        )
    end

    temp_values = _batched_numeric_scratch!(env, depth)
    temp_gradients = _batched_backend_gradient_scratch!(cache, depth)
    for argument in Base.tail(expr.arguments)
        _eval_backend_broadcast_numeric_and_gradient!(temp_values, temp_gradients, cache, env, argument, element_index, depth + 1)
        _apply_backend_numeric_gradient_binary!(values, gradients, env, expr.op, temp_values, temp_gradients)
    end
    return values, gradients
end

function _eval_backend_broadcast_numeric_and_gradient!(
    values::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    expr::BackendBlockExpr,
    element_index::Int,
    depth::Int,
) where {T<:AbstractFloat}
    for argument in expr.arguments
        _eval_backend_broadcast_numeric_and_gradient!(values, gradients, cache, env, argument, element_index, depth)
    end
    return values, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBroadcastNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    isnothing(step.binding_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support broadcast normal bindings"))
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    observed = _batched_broadcast_observed_values(env, address_parts, constraints)
    n = _broadcast_uniform_length(observed)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 1)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 2)
    for element_index = 1:n
        _eval_backend_broadcast_numeric_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, element_index, 3)
        _eval_backend_broadcast_numeric_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, element_index, 5)
        for batch_index in eachindex(totals)
            value = observed[batch_index][element_index]
            mu = mu_values[batch_index]
            sigma = sigma_values[batch_index]
            totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
            z = (value - mu) / sigma
            inv_sigma = 1 / sigma
            dmu = z * inv_sigma
            dsigma = (z * z - 1) * inv_sigma
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    dmu * mu_gradients[parameter_index, batch_index] +
                    dsigma * sigma_gradients[parameter_index, batch_index]
            end
        end
    end
    return totals, gradients
end

# Family-generic broadcast scalar observation gradient (issue #287): the same
# per-element loop as broadcast normal, but with a tuple of argument channels
# and the family's single-source partials (issue #285). Argument k uses numeric
# scratch k, gradient scratch k, and nested-evaluation depth 2k + 1 (the same
# depth spacing the two-argument normal path uses).
function _score_backend_step_and_gradient!(
    step::BackendBroadcastScalarChoicePlanStep{F},
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {F,T<:AbstractFloat}
    isnothing(step.binding_slot) ||
        throw(BatchedBackendFallback("batched backend gradient does not support broadcast $(F) bindings"))
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    observed = _batched_broadcast_observed_values(env, address_parts, constraints)
    n = _broadcast_uniform_length(observed)
    arg_count = length(step.arguments)
    arg_values = ntuple(k -> _batched_numeric_scratch!(env, k), arg_count)
    arg_gradients = ntuple(k -> _batched_backend_gradient_scratch!(cache, k), arg_count)
    for element_index = 1:n
        for k = 1:arg_count
            _eval_backend_broadcast_numeric_and_gradient!(
                arg_values[k], arg_gradients[k], cache, env, step.arguments[k], element_index, 2 * k + 1,
            )
        end
        for batch_index in eachindex(totals)
            value = observed[batch_index][element_index]
            args = ntuple(k -> arg_values[k][batch_index], arg_count)
            totals[batch_index] += _broadcast_element_logpdf(Val(F), value, args...)
            partials = _broadcast_element_partials(Val(F), value, args...)
            isnothing(partials) && continue
            for parameter_index in axes(gradients, 1)
                accumulator = zero(T)
                for k = 1:arg_count
                    accumulator += partials[k] * arg_gradients[k][parameter_index, batch_index]
                end
                gradients[parameter_index, batch_index] += accumulator
            end
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendDirichletChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:step.value_length]
    alpha_values = [_batched_numeric_scratch!(env, step.value_length + index) for index = 1:step.value_length]
    alpha_gradients = [
        _batched_backend_gradient_scratch!(cache, step.value_length + index) for index = 1:step.value_length
    ]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    for component_index = 1:step.value_length
        _eval_backend_numeric_expr_and_gradient!(
            alpha_values[component_index],
            alpha_gradients[component_index],
            cache,
            env,
            step.alpha[component_index],
            2 * step.value_length + 1,
        )
    end

    _accumulate_dirichlet_gradient!(
        totals,
        gradients,
        step.parameter_index,
        choice_values,
        alpha_values,
        alpha_gradients,
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, alpha_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLKJCholeskyChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:step.value_length]
    eta_values = _batched_numeric_scratch!(env, step.value_length + 1)
    eta_gradients = _batched_backend_gradient_scratch!(cache, step.value_length + 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, step.value_length, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(eta_values, eta_gradients, cache, env, step.eta, step.value_length + 2)

    _accumulate_lkjcholesky_gradient!(
        totals,
        gradients,
        step.d,
        step.parameter_index,
        choice_values,
        eta_values,
        eta_gradients,
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, Matrix{T}[])
    return totals, gradients
end

# --- diagonal / dense multivariate Student-t gradient -------------------------

# The multivariate-t density couples all d dimensions through the single
# quadratic q, so the gradient is a q-dependent scalar reweighting r =
# (nu+d)/(nu+q) of the mvnormal partials, plus a degrees-of-freedom channel.
# nu's derivative (digamma normalizer + the log1p tail term) chains through the
# nu expression, mirroring lkjcholesky's eta channel.
function _mvstudentt_nu_derivative(nu, d::Int, q)
    half = (nu + d) / 2
    return oftype(q, 0.5) * digamma(half) - oftype(q, 0.5) * digamma(nu / 2) - d / (2 * nu) -
           oftype(q, 0.5) * log1p(q / nu) + (nu + d) * q / (2 * nu * (nu + q))
end

function _accumulate_mvstudentt_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    d::Int,
    value_values::AbstractVector{<:AbstractVector},
    value_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    sigma_values::AbstractVector{<:AbstractVector},
    sigma_gradients::AbstractVector{<:AbstractMatrix},
    nu_values::AbstractVector,
    nu_gradients::AbstractMatrix,
) where {T<:AbstractFloat}
    z = Vector{T}(undef, d)
    inv_sigma = Vector{T}(undef, d)
    for batch_index in eachindex(totals)
        nu = nu_values[batch_index]
        quadratic = zero(T)
        log_det = zero(T)
        for c = 1:d
            s = sigma_values[c][batch_index]
            zc = (value_values[c][batch_index] - mu_values[c][batch_index]) / s
            z[c] = zc
            inv_sigma[c] = one(T) / s
            quadratic += zc * zc
            log_det += log(s)
        end
        totals[batch_index] += _backend_mvstudentt_density(nu, d, log_det, quadratic)
        r = (nu + d) / (nu + quadratic)
        dnu = _mvstudentt_nu_derivative(nu, d, quadratic)
        for parameter_row in axes(gradients, 1)
            gradients[parameter_row, batch_index] += dnu * nu_gradients[parameter_row, batch_index]
        end
        for c = 1:d
            zc = z[c]
            is = inv_sigma[c]
            dvalue = -r * zc * is
            dmu = r * zc * is
            dsigma = (r * zc * zc - one(T)) * is
            cvg = value_gradients[c]
            cmg = mu_gradients[c]
            csg = sigma_gradients[c]
            for parameter_row in axes(gradients, 1)
                gradients[parameter_row, batch_index] +=
                    dvalue * cvg[parameter_row, batch_index] +
                    dmu * cmg[parameter_row, batch_index] +
                    dsigma * csg[parameter_row, batch_index]
            end
        end
    end
    return totals, gradients
end

function _accumulate_mvstudenttdense_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    d::Int,
    value_values::AbstractVector{<:AbstractVector},
    value_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    scale_storage::AbstractVector,
    nu_values::AbstractVector,
    nu_gradients::AbstractMatrix,
) where {T<:AbstractFloat}
    z = Vector{T}(undef, d)
    w = Vector{T}(undef, d)
    for batch_index in eachindex(totals)
        Lmat = _backend_mvnormaldense_scale_matrix(scale_storage[batch_index], d)
        mu = ntuple(index -> mu_values[index][batch_index], d)
        x = ntuple(index -> value_values[index][batch_index], d)
        nu = nu_values[batch_index]
        quadratic = zero(T)
        log_det = zero(T)
        for row = 1:d
            residual = x[row] - mu[row]
            for col = 1:(row-1)
                residual -= Lmat[row, col] * z[col]
            end
            z[row] = residual / Lmat[row, row]
            quadratic += z[row] * z[row]
            log_det += log(Lmat[row, row])
        end
        totals[batch_index] += _backend_mvstudentt_density(nu, d, log_det, quadratic)
        # back substitution Lᵀ w = z
        for row = d:-1:1
            accumulator = z[row]
            for col = (row+1):d
                accumulator -= Lmat[col, row] * w[col]
            end
            w[row] = accumulator / Lmat[row, row]
        end
        r = (nu + d) / (nu + quadratic)
        dnu = _mvstudentt_nu_derivative(nu, d, quadratic)
        for parameter_row in axes(gradients, 1)
            gradients[parameter_row, batch_index] += dnu * nu_gradients[parameter_row, batch_index]
        end
        for c = 1:d
            dmu = r * w[c]
            dvalue = -r * w[c]
            cmg = mu_gradients[c]
            cvg = value_gradients[c]
            for parameter_row in axes(gradients, 1)
                gradients[parameter_row, batch_index] +=
                    dmu * cmg[parameter_row, batch_index] + dvalue * cvg[parameter_row, batch_index]
            end
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendMvStudentTChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:d]
    choice_gradients = [_batched_backend_gradient_scratch!(cache, index) for index = 1:d]
    mu_values = [_batched_numeric_scratch!(env, d + index) for index = 1:d]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, d + index) for index = 1:d]
    sigma_values = [_batched_numeric_scratch!(env, 2 * d + index) for index = 1:d]
    sigma_gradients = [_batched_backend_gradient_scratch!(cache, 2 * d + index) for index = 1:d]
    nu_values = _batched_numeric_scratch!(env, 3 * d + 1)
    nu_gradients = _batched_backend_gradient_scratch!(cache, 3 * d + 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(nu_values, nu_gradients, cache, env, step.nu, 3 * d + 2)
    for component_index = 1:d
        _fill_choice_vector_gradient!(choice_gradients[component_index], step.value_index, component_index, cache.seed_rows)
        _eval_backend_numeric_expr_and_gradient!(
            mu_values[component_index], mu_gradients[component_index], cache, env, step.mu[component_index], 3 * d + 3,
        )
        _eval_backend_numeric_expr_and_gradient!(
            sigma_values[component_index], sigma_gradients[component_index], cache, env, step.sigma[component_index], 3 * d + 4,
        )
    end

    _accumulate_mvstudentt_gradient!(
        totals, gradients, d, choice_values, choice_gradients, mu_values, mu_gradients,
        sigma_values, sigma_gradients, nu_values, nu_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, choice_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendMvStudentTDenseChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:d]
    choice_gradients = [_batched_backend_gradient_scratch!(cache, index) for index = 1:d]
    mu_values = [_batched_numeric_scratch!(env, d + index) for index = 1:d]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, d + index) for index = 1:d]
    nu_values = _batched_numeric_scratch!(env, 2 * d + 1)
    nu_gradients = _batched_backend_gradient_scratch!(cache, 2 * d + 1)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)
    _eval_backend_numeric_expr_and_gradient!(nu_values, nu_gradients, cache, env, step.nu, 2 * d + 2)
    for component_index = 1:d
        _fill_choice_vector_gradient!(choice_gradients[component_index], step.value_index, component_index, cache.seed_rows)
        _eval_backend_numeric_expr_and_gradient!(
            mu_values[component_index], mu_gradients[component_index], cache, env, step.mu[component_index], 2 * d + 3,
        )
    end
    env.assigned[step.scale_tril.slot] ||
        throw(BatchedBackendFallback("mvstudenttdense scale_tril slot $(step.scale_tril.slot) is not assigned"))
    _accumulate_mvstudenttdense_gradient!(
        totals, gradients, d, choice_values, choice_gradients, mu_values, mu_gradients,
        env.generic_values[step.scale_tril.slot], nu_values, nu_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, choice_gradients)
    return totals, gradients
end

# --- mixture-of-normals gradient ----------------------------------------------

function _accumulate_mixture_normal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    weight_values::AbstractVector{<:AbstractVector},
    weight_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    sigma_values::AbstractVector{<:AbstractVector},
    sigma_gradients::AbstractVector{<:AbstractMatrix},
) where {T<:AbstractFloat}
    k = length(weight_values)
    terms = Vector{T}(undef, k)
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        weights = ntuple(index -> weight_values[index][batch_index], k)
        mus = ntuple(index -> mu_values[index][batch_index], k)
        sigmas = ntuple(index -> sigma_values[index][batch_index], k)
        logpdf = _backend_mixture_normal_logpdf(weights, mus, sigmas, value)
        totals[batch_index] += logpdf
        isfinite(logpdf) || continue
        dvalue = zero(T)
        for index = 1:k
            lp = _backend_normal_logpdf(mus[index], sigmas[index], value)
            terms[index] = lp
        end
        for index = 1:k
            mu = mus[index]
            sigma = sigmas[index]
            lp = terms[index]
            responsibility = exp(log(weights[index]) + lp - logpdf)
            weight_derivative = exp(lp - logpdf)
            z = (value - mu) / sigma
            inv_sigma = 1 / sigma
            dvalue += responsibility * (-z * inv_sigma)
            dmu = responsibility * (z * inv_sigma)
            dsigma = responsibility * ((z * z - one(T)) * inv_sigma)
            component_weight_gradients = weight_gradients[index]
            component_mu_gradients = mu_gradients[index]
            component_sigma_gradients = sigma_gradients[index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    weight_derivative * component_weight_gradients[parameter_index, batch_index] +
                    dmu * component_mu_gradients[parameter_index, batch_index] +
                    dsigma * component_sigma_gradients[parameter_index, batch_index]
            end
        end
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] += dvalue * value_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendMixtureNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    k = length(step.weights)
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    weight_values = [_batched_numeric_scratch!(env, index) for index = 1:k]
    weight_gradients = [_batched_backend_gradient_scratch!(cache, 1 + index) for index = 1:k]
    mu_values = [_batched_numeric_scratch!(env, k + index) for index = 1:k]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, k + 1 + index) for index = 1:k]
    sigma_values = [_batched_numeric_scratch!(env, 2 * k + index) for index = 1:k]
    sigma_gradients = [_batched_backend_gradient_scratch!(cache, 2 * k + 1 + index) for index = 1:k]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    for index = 1:k
        _eval_backend_numeric_expr_and_gradient!(
            weight_values[index],
            weight_gradients[index],
            cache,
            env,
            step.weights[index],
            3 * k + 2,
        )
        _eval_backend_numeric_expr_and_gradient!(mu_values[index], mu_gradients[index], cache, env, step.mus[index], 3 * k + 2)
        _eval_backend_numeric_expr_and_gradient!(
            sigma_values[index],
            sigma_gradients[index],
            cache,
            env,
            step.sigmas[index],
            3 * k + 2,
        )
    end
    _accumulate_mixture_normal_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        weight_values,
        weight_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

# --- dense multivariate normal gradient ---------------------------------------

# Solve L z = residual (forward) then Lᵀ w = z (back) so that w = L⁻ᵀ z, giving
# d logpdf / d mu_i = w_i and d logpdf / d x_i = -w_i (the scale factor is a
# constant matrix, so it contributes no gradient).
function _accumulate_mvnormaldense_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    d::Int,
    choice_values::AbstractVector{<:AbstractVector},
    choice_gradients::AbstractVector{<:AbstractMatrix},
    mu_values::AbstractVector{<:AbstractVector},
    mu_gradients::AbstractVector{<:AbstractMatrix},
    scale_storage::AbstractVector,
) where {T<:AbstractFloat}
    z = Vector{T}(undef, d)
    w = Vector{T}(undef, d)
    for batch_index in eachindex(totals)
        Lmat = _backend_mvnormaldense_scale_matrix(scale_storage[batch_index], d)
        mu = ntuple(index -> mu_values[index][batch_index], d)
        x = ntuple(index -> choice_values[index][batch_index], d)
        totals[batch_index] += _backend_mvnormaldense_logpdf(mu, Lmat, x)
        # forward substitution: L z = x - mu
        for row = 1:d
            residual = x[row] - mu[row]
            for col = 1:(row-1)
                residual -= Lmat[row, col] * z[col]
            end
            z[row] = residual / Lmat[row, row]
        end
        # back substitution: Lᵀ w = z
        for row = d:-1:1
            accumulator = z[row]
            for col = (row+1):d
                accumulator -= Lmat[col, row] * w[col]
            end
            w[row] = accumulator / Lmat[row, row]
        end
        for component_index = 1:d
            dmu = w[component_index]
            dvalue = -w[component_index]
            component_mu_gradients = mu_gradients[component_index]
            component_value_gradients = choice_gradients[component_index]
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] +=
                    dmu * component_mu_gradients[parameter_index, batch_index] +
                    dvalue * component_value_gradients[parameter_index, batch_index]
            end
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendMvNormalDenseChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    d = step.value_length
    choice_values = [_batched_numeric_scratch!(env, index) for index = 1:d]
    choice_gradients = [_batched_backend_gradient_scratch!(cache, index) for index = 1:d]
    mu_values = [_batched_numeric_scratch!(env, d + index) for index = 1:d]
    mu_gradients = [_batched_backend_gradient_scratch!(cache, d + index) for index = 1:d]
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_vector_values!(choice_values, step.value_index, d, params, constraints, address_parts)
    for component_index = 1:d
        _fill_choice_vector_gradient!(choice_gradients[component_index], step.value_index, component_index, cache.seed_rows)
        _eval_backend_numeric_expr_and_gradient!(
            mu_values[component_index],
            mu_gradients[component_index],
            cache,
            env,
            step.mu[component_index],
            2 * d + 1,
        )
    end

    env.assigned[step.scale_tril.slot] ||
        throw(BatchedBackendFallback("mvnormaldense scale_tril slot $(step.scale_tril.slot) is not assigned"))
    _accumulate_mvnormaldense_gradient!(
        totals,
        gradients,
        d,
        choice_values,
        choice_gradients,
        mu_values,
        mu_gradients,
        env.generic_values[step.scale_tril.slot],
    )

    isnothing(step.binding_slot) ||
        _assign_backend_choice_vector_value!(env, cache.slot_gradients, step.binding_slot, choice_values, choice_gradients)
    return totals, gradients
end
