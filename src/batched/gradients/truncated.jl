# Hand-derived analytic batched logjoint gradients: truncated families (truncatednormal, truncatedstudentt).

# --- truncated normal gradient ------------------------------------------------

function _accumulate_truncatednormal_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
    lower_values::AbstractVector{T},
    lower_gradients::AbstractMatrix{T},
    upper_values::AbstractVector{T},
    upper_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        lower = lower_values[batch_index]
        upper = upper_values[batch_index]
        totals[batch_index] += _backend_truncatednormal_logpdf(mu, sigma, lower, upper, value)
        if value < lower || value > upper
            # off-support latent (dynamic bounds): reject via the gradient too
            # (issue #343)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        inv_sigma = 1 / sigma
        zx = (value - mu) * inv_sigma
        za = (lower - mu) * inv_sigma
        zb = (upper - mu) * inv_sigma
        # pdf/Z ratios in log space (issue #43): with a finite log Z deep in a
        # tail, the split pdf * exp(-log Z) overflows/underflows at Float32
        # into Inf * 0 = NaN; the fused exponent stays finite
        log_z = _log_normal_cdf_diff(za, zb)
        ratio_a = isinf(za) ? zero(T) : exp(-za * za / 2 - T(0.9189385332046727) - log_z)
        ratio_b = isinf(zb) ? zero(T) : exp(-zb * zb / 2 - T(0.9189385332046727) - log_z)
        za_ratio_a = isinf(za) ? zero(T) : za * ratio_a
        zb_ratio_b = isinf(zb) ? zero(T) : zb * ratio_b
        dvalue = -zx * inv_sigma
        dmu = zx * inv_sigma - (ratio_a - ratio_b) * inv_sigma
        dsigma = (zx * zx - one(T)) * inv_sigma - (za_ratio_a - zb_ratio_b) * inv_sigma
        dlower = ratio_a * inv_sigma
        dupper = -ratio_b * inv_sigma
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index] +
                dlower * lower_gradients[parameter_index, batch_index] +
                dupper * upper_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendTruncatedNormalChoicePlanStep,
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
    lower_values = _batched_numeric_scratch!(env, 3)
    lower_gradients = _batched_backend_gradient_scratch!(cache, 4)
    upper_values = _batched_numeric_scratch!(env, 4)
    upper_gradients = _batched_backend_gradient_scratch!(cache, 5)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 6)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 7)
    _eval_backend_numeric_expr_and_gradient!(lower_values, lower_gradients, cache, env, step.lower, 8)
    _eval_backend_numeric_expr_and_gradient!(upper_values, upper_gradients, cache, env, step.upper, 9)
    _accumulate_truncatednormal_gradient!(
        totals,
        gradients,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
        lower_values,
        lower_gradients,
        upper_values,
        upper_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end

# --- truncated Student-t gradient ---------------------------------------------

# Backend support is restricted (at lowering) to a constant nu, so the analytic
# gradient omits the intractable d/dnu term (the incomplete-beta nu-derivative);
# nu carries no latent dependence, so the omitted term is genuinely zero. The
# remaining partials mirror the truncated-normal template with the Student-t pdf
# `p` and CDF replacing the normal ones. With z = (value-mu)/sigma the base
# kernel derivative is k = d/dz [-(nu+1)/2 log1p(z^2/nu)] = -(nu+1) z / (nu+z^2).
function _accumulate_truncatedstudentt_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    nu_values::AbstractVector{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    mu_values::AbstractVector{T},
    mu_gradients::AbstractMatrix{T},
    sigma_values::AbstractVector{T},
    sigma_gradients::AbstractMatrix{T},
    lower_values::AbstractVector{T},
    lower_gradients::AbstractMatrix{T},
    upper_values::AbstractVector{T},
    upper_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        nu = nu_values[batch_index]
        value = value_values[batch_index]
        mu = mu_values[batch_index]
        sigma = sigma_values[batch_index]
        lower = lower_values[batch_index]
        upper = upper_values[batch_index]
        totals[batch_index] += _backend_truncatedstudentt_logpdf(nu, mu, sigma, lower, upper, value)
        if value < lower || value > upper
            # off-support latent (dynamic bounds): reject via the gradient too
            # (issue #343)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        inv_sigma = 1 / sigma
        zx = (value - mu) * inv_sigma
        za = (lower - mu) * inv_sigma
        zb = (upper - mu) * inv_sigma
        # each normalizer term is the ratio pdf/Z formed IN LOG SPACE
        # (issue #43): the plain cdf difference cancels to zero for light
        # tails, and even with a finite log Z the split exp(-log Z) * pdf
        # overflows/underflows at Float32 into Inf * 0 = NaN
        log_z = _t_log_normalizer(nu, za, zb)
        ratio_a = isinf(za) ? zero(T) : exp(_std_t_log_pdf(za, nu) - log_z)
        ratio_b = isinf(zb) ? zero(T) : exp(_std_t_log_pdf(zb, nu) - log_z)
        # guard the z * ratio products too: an infinite bound has ratio 0, and
        # Inf * 0 would reintroduce the NaN the ratios just eliminated
        za_ratio_a = isinf(za) ? zero(T) : za * ratio_a
        zb_ratio_b = isinf(zb) ? zero(T) : zb * ratio_b
        # k = d(base logpdf)/dz for the Student-t kernel.
        k = -(nu + one(T)) * zx / (nu + zx * zx)
        dvalue = k * inv_sigma
        dmu = -k * inv_sigma + (ratio_b - ratio_a) * inv_sigma
        dsigma = -inv_sigma - k * zx * inv_sigma + (zb_ratio_b - za_ratio_a) * inv_sigma
        dlower = ratio_a * inv_sigma
        dupper = -ratio_b * inv_sigma
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dmu * mu_gradients[parameter_index, batch_index] +
                dsigma * sigma_gradients[parameter_index, batch_index] +
                dlower * lower_gradients[parameter_index, batch_index] +
                dupper * upper_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

function _score_backend_step_and_gradient!(
    step::BackendTruncatedStudentTChoicePlanStep,
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
    mu_values = _batched_numeric_scratch!(env, 2)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 3)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    lower_values = _batched_numeric_scratch!(env, 4)
    lower_gradients = _batched_backend_gradient_scratch!(cache, 4)
    upper_values = _batched_numeric_scratch!(env, 5)
    upper_gradients = _batched_backend_gradient_scratch!(cache, 5)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
    # nu is a lowering-guaranteed constant; only its value is needed (gradient zero).
    _eval_backend_numeric_expr!(nu_values, env, step.nu, 6)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 7)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 8)
    _eval_backend_numeric_expr_and_gradient!(lower_values, lower_gradients, cache, env, step.lower, 9)
    _eval_backend_numeric_expr_and_gradient!(upper_values, upper_gradients, cache, env, step.upper, 10)
    _accumulate_truncatedstudentt_gradient!(
        totals,
        gradients,
        nu_values,
        value_values,
        value_gradients,
        mu_values,
        mu_gradients,
        sigma_values,
        sigma_gradients,
        lower_values,
        lower_gradients,
        upper_values,
        upper_gradients,
    )
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
    return totals, gradients
end
