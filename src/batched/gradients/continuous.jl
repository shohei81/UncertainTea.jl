# Hand-derived analytic batched logjoint gradients: continuous scalar families (normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, studentt).

# Off-support value with a live derivative seed (issue #343): a latent flowing
# through a saturating transform can land EXACTLY on the support boundary
# (sigmoid(theta) rounds to 1.0 for theta >~ 36.74, exp(theta) underflows to
# 0.0 below ~-745). The logpdf contribution is then -Inf, but the accumulate
# loops used to skip the partials entirely, leaving only the finite transform
# Jacobian term in the unconstrained gradient -- a silently wrong FINITE
# gradient that leapfrog gradient guards never reject. Whenever the partials
# are skipped for a value carrying derivative information, poison the affected
# gradient rows with NaN so the -Inf logjoint and the gradient reject
# together. Observed off-support values have an all-zero value-gradient seed
# and keep the previous skip semantics (finite parameter partials dropped,
# whole evaluation already scored -Inf).
function _poison_offsupport_value_gradient!(
    gradients::AbstractMatrix{T},
    value_gradients::AbstractMatrix{T},
    batch_index::Integer,
) where {T<:AbstractFloat}
    for parameter_index in axes(gradients, 1)
        iszero(value_gradients[parameter_index, batch_index]) ||
            (gradients[parameter_index, batch_index] = T(NaN))
    end
    return nothing
end

function _accumulate_normal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_normal_logpdf(mu, sigma, value)
        dvalue, dmu, dsigma = _normal_logpdf_partials(mu, sigma, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_lognormal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_lognormal_logpdf(mu, sigma, value)
        if !(value > 0)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dmu, dsigma = _lognormal_logpdf_partials(mu, sigma, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_exponential_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    rate_values::AbstractVector{T},
    rate_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_exponential_logpdf(rate, value)
        if !(value >= 0)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, drate = _exponential_logpdf_partials(rate, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                drate * rate_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_gamma_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    shape_values::AbstractVector{T},
    shape_gradients::AbstractMatrix{T},
    rate_values::AbstractVector{T},
    rate_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        shape = shape_values[batch_index]
        rate = rate_values[batch_index]
        totals[batch_index] += _backend_gamma_logpdf(shape, rate, value)
        # subnormal values are the exp-underflow boundary (issue #345): the
        # kernel scores them -Inf, so poison alongside the exact-0.0 case
        if !(value > 0) || issubnormal(value)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dshape, drate = _gamma_logpdf_partials(shape, rate, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dshape * shape_gradients[parameter_index, batch_index] +
                drate * rate_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_inversegamma_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    shape_values::AbstractVector{T},
    shape_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_inversegamma_logpdf(shape, scale, value)
        if !(value > 0)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dshape, dscale = _inversegamma_logpdf_partials(shape, scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dshape * shape_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_weibull_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    shape_values::AbstractVector{T},
    shape_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_weibull_logpdf(shape, scale, value)
        if !(value > 0)
            # At the x == 0, shape == 1 boundary the logpdf is -log(scale)
            # (see logpdf(::WeibullDist, x)), so the scale partial is finite
            # while the value/shape channels stay zero as in the CPU
            # ForwardDiff path (issue #86).
            if value == 0 && shape == 1
                dscale_boundary = -1 / scale
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] +=
                        dscale_boundary * scale_gradients[parameter_index, batch_index]
                end
            else
                # non-finite boundary/off-support score: reject via the
                # gradient too (issue #343)
                _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            end
            continue
        end
        dvalue, dshape, dscale = _weibull_logpdf_partials(shape, scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dshape * shape_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_beta_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    alpha_values::AbstractVector{T},
    alpha_gradients::AbstractMatrix{T},
    beta_values::AbstractVector{T},
    beta_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        alpha = alpha_values[batch_index]
        beta_parameter = beta_values[batch_index]
        totals[batch_index] += _backend_beta_logpdf(alpha, beta_parameter, value)
        if !(0 < value < 1)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dalpha, dbeta = _beta_logpdf_partials(alpha, beta_parameter, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dalpha * alpha_gradients[parameter_index, batch_index] +
                dbeta * beta_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_studentt_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    nu_values::AbstractVector{T},
    nu_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        nu = nu_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_studentt_logpdf(nu, mu, sigma, value)
        dvalue, dnu, dmu, dsigma = _studentt_logpdf_partials(nu, mu, sigma, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dnu * nu_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 5)
    _accumulate_normal_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendNoncenteredNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    z_values = env.observed_values
    z_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    theta_values = _batched_numeric_scratch!(env, 3)
    theta_gradients = _batched_backend_gradient_scratch!(cache, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(z_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(z_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 5)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 6)
    for batch_index in eachindex(totals)
        z = z_values[batch_index]
        sigma = sigma_values[batch_index]
        (isfinite(sigma) && sigma > 0) || throw(
            BatchedBackendFallback("noncentered normal requires a finite positive scale, got $sigma"),
        )
        totals[batch_index] += _backend_normal_logpdf(zero(z), one(z), z)
        theta_values[batch_index] = mu_values[batch_index] + sigma * z
    end
    # d logpdf(N(0,1), z)/dz = -z through the slot seed of z
    for batch_index in eachindex(totals), parameter_index in axes(gradients, 1)
        z_grad = z_gradients[parameter_index, batch_index]
        iszero(z_grad) || (gradients[parameter_index, batch_index] -= z_values[batch_index] * z_grad)
        theta_gradients[parameter_index, batch_index] =
            mu_gradients[parameter_index, batch_index] +
            z_values[batch_index] * sigma_gradients[parameter_index, batch_index] +
            sigma_values[batch_index] * z_grad
    end
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, theta_values, theta_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLognormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 5)
    _accumulate_lognormal_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendExponentialChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    rate_values = _batched_numeric_scratch!(env, 1)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, step.rate, 3)
    _accumulate_exponential_gradient!(totals, gradients, value_values, value_gradients, rate_values, rate_gradients)
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendGammaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    rate_values = _batched_numeric_scratch!(env, 2)
    rate_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, step.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(rate_values, rate_gradients, cache, env, step.rate, 5)
    _accumulate_gamma_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        shape_values,
        shape_gradients,
        rate_values,
        rate_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendInverseGammaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, step.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_inversegamma_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        shape_values,
        shape_gradients,
        scale_values,
        scale_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendWeibullChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, step.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_weibull_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        shape_values,
        shape_gradients,
        scale_values,
        scale_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendBetaChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    alpha_values = _batched_numeric_scratch!(env, 1)
    alpha_gradients = _batched_backend_gradient_scratch!(cache, 2)
    beta_values = _batched_numeric_scratch!(env, 2)
    beta_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(alpha_values, alpha_gradients, cache, env, step.alpha, 4)
    _eval_backend_numeric_expr_and_gradient!(beta_values, beta_gradients, cache, env, step.beta, 5)
    _accumulate_beta_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        alpha_values,
        alpha_gradients,
        beta_values,
        beta_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendStudentTChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    nu_values = _batched_numeric_scratch!(env, 1)
    nu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    mu_values = _batched_numeric_scratch!(env, 2)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 3)
    sigma_values = _batched_numeric_scratch!(env, 3)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(nu_values, nu_gradients, cache, env, step.nu, 5)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 6)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 7)
    _accumulate_studentt_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        nu_values,
        nu_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end
function _accumulate_laplace_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_laplace_logpdf(mu, scale, value)
        dvalue, dmu, dscale = _laplace_logpdf_partials(mu, scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLaplaceChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_laplace_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        scale_values,
        scale_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

# ---- scalar prior families (issue #229): cauchy, halfnormal, halfcauchy, uniform, logistic, gumbel ----

function _accumulate_cauchy_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_cauchy_logpdf(mu, sigma, value)
        dvalue, dmu, dsigma = _cauchy_logpdf_partials(mu, sigma, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_halfnormal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        sigma = sigma_values[batch_index]
        totals[batch_index] += _backend_halfnormal_logpdf(sigma, value)
        if value < 0
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dsigma = _halfnormal_logpdf_partials(sigma, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_halfcauchy_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_halfcauchy_logpdf(scale, value)
        if value < 0
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dscale = _halfcauchy_logpdf_partials(scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_uniform_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    lower_values::AbstractVector{T},
    lower_gradients::AbstractMatrix{T},
    upper_values::AbstractVector{T},
    upper_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        lower = lower_values[batch_index]
        upper = upper_values[batch_index]
        totals[batch_index] += _backend_uniform_logpdf(lower, upper, value)
        if !(lower <= value <= upper)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        # d/dvalue is 0 on the open interval; the bound partials are the only
        # nonzero channels (relevant only for dynamic-bound observations).
        dlower, dupper = _uniform_logpdf_partials(lower, upper, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dlower * lower_gradients[parameter_index, batch_index] +
                dupper * upper_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_logistic_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_logistic_logpdf(mu, scale, value)
        dvalue, dmu, dscale = _logistic_logpdf_partials(mu, scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_gumbel_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_gumbel_logpdf(mu, scale, value)
        dvalue, dmu, dscale = _gumbel_logpdf_partials(mu, scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendCauchyChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 5)
    _accumulate_cauchy_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendHalfNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    sigma_values = _batched_numeric_scratch!(env, 1)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 3)
    _accumulate_halfnormal_gradient!(totals, gradients, value_values, value_gradients, sigma_values, sigma_gradients)
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendHalfCauchyChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    scale_values = _batched_numeric_scratch!(env, 1)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 3)
    _accumulate_halfcauchy_gradient!(totals, gradients, value_values, value_gradients, scale_values, scale_gradients)
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendUniformChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    lower_values = _batched_numeric_scratch!(env, 1)
    lower_gradients = _batched_backend_gradient_scratch!(cache, 2)
    upper_values = _batched_numeric_scratch!(env, 2)
    upper_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(lower_values, lower_gradients, cache, env, step.lower, 4)
    _eval_backend_numeric_expr_and_gradient!(upper_values, upper_gradients, cache, env, step.upper, 5)
    _accumulate_uniform_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        lower_values,
        lower_gradients,
        upper_values,
        upper_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendLogisticChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_logistic_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        scale_values,
        scale_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendGumbelChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_gumbel_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        scale_values,
        scale_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

# ---- positive-support / heavy-tail families (issue #230): pareto, frechet, rayleigh, inversegaussian ----

function _accumulate_pareto_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    xm_values::AbstractVector{T},
    xm_gradients::AbstractMatrix{T},
    alpha_values::AbstractVector{T},
    alpha_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        xm = xm_values[batch_index]
        alpha = alpha_values[batch_index]
        totals[batch_index] += _backend_pareto_logpdf(xm, alpha, value)
        if !(xm > 0 && alpha > 0 && value >= xm)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dxm, dalpha = _pareto_logpdf_partials(xm, alpha, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dxm * xm_gradients[parameter_index, batch_index] +
                dalpha * alpha_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_frechet_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    shape_values::AbstractVector{T},
    shape_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_frechet_logpdf(shape, scale, value)
        if !(shape > 0 && scale > 0 && value > 0)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dshape, dscale = _frechet_logpdf_partials(shape, scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dshape * shape_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_rayleigh_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_rayleigh_logpdf(scale, value)
        if !(scale > 0 && value > 0)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dscale = _rayleigh_logpdf_partials(scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _accumulate_inversegaussian_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    lambda_values::AbstractVector{T},
    lambda_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        lambda = lambda_values[batch_index]
        totals[batch_index] += _backend_inversegaussian_logpdf(mu, lambda, value)
        if !(mu > 0 && lambda > 0 && value > 0)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        dvalue, dmu, dlambda = _inversegaussian_logpdf_partials(mu, lambda, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dlambda * lambda_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendParetoChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    xm_values = _batched_numeric_scratch!(env, 1)
    xm_gradients = _batched_backend_gradient_scratch!(cache, 2)
    alpha_values = _batched_numeric_scratch!(env, 2)
    alpha_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(xm_values, xm_gradients, cache, env, step.xm, 4)
    _eval_backend_numeric_expr_and_gradient!(alpha_values, alpha_gradients, cache, env, step.alpha, 5)
    _accumulate_pareto_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        xm_values,
        xm_gradients,
        alpha_values,
        alpha_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendFrechetChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    shape_values = _batched_numeric_scratch!(env, 1)
    shape_gradients = _batched_backend_gradient_scratch!(cache, 2)
    scale_values = _batched_numeric_scratch!(env, 2)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(shape_values, shape_gradients, cache, env, step.shape, 4)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 5)
    _accumulate_frechet_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        shape_values,
        shape_gradients,
        scale_values,
        scale_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendRayleighChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    scale_values = _batched_numeric_scratch!(env, 1)
    scale_gradients = _batched_backend_gradient_scratch!(cache, 2)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(scale_values, scale_gradients, cache, env, step.scale, 3)
    _accumulate_rayleigh_gradient!(totals, gradients, value_values, value_gradients, scale_values, scale_gradients)
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendInverseGaussianChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    value_values = env.observed_values
    value_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    lambda_values = _batched_numeric_scratch!(env, 2)
    lambda_gradients = _batched_backend_gradient_scratch!(cache, 3)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 4)
    _eval_backend_numeric_expr_and_gradient!(lambda_values, lambda_gradients, cache, env, step.lambda, 5)
    _accumulate_inversegaussian_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        lambda_values,
        lambda_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end
