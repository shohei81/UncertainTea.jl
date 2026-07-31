# One fused KernelAbstractions kernel that computes the gradient of the UNCONSTRAINED
# logjoint by scalar forward-mode differentiation. The `ndrange` is 2D:
# `(parameter_index, batch_index)`. Thread `(p, b)` re-walks the SAME device plan as
# the logjoint kernel, but in `DeviceDual` numbers seeded so that only unconstrained
# parameter row `p` of column `b` carries derivative 1. Its accumulated total is a dual
# whose `.deriv` is d(logjoint)/d(param p) and whose `.value` is the logjoint itself.
#
# Cost is O(P * plan) per column -- the same FLOP class as the CPU manual forward
# accumulation in `manual_gradient_scoring.jl`, whose per-distribution derivative
# FORMULAS this differentiation reproduces exactly (forward-mode through the same
# logpdfs and the same unconstrained transform + log-abs-det).
#
# Per-thread slot scratch lives in a 3D `DeviceDual` buffer laid out as
# `slots[slot, parameter_index, batch_index]`; each `(p, b)` thread owns the
# `slots[:, p, b]` column and never races another thread. `params`/`observed` are the
# same plain-`T` buffers the logjoint kernel uses, indexed by the batch column `b`.
#
# `math.jl`'s logpdfs and `_device_transform` are reused verbatim: they are written
# `where {T}` with no `<:Real` bound, so `T` binds to `DeviceDual{...}` and the whole
# density (including the transform log-abs-det) differentiates through the duals. The
# per-choice arguments are `promote`d so the three positional arguments share the one
# `DeviceDual{...}` type the logpdf signatures require.

# ---- dual expression evaluation ------------------------------------------------

@inline _device_grad_eval(e::DeviceLiteralExpr, slots, pidx, b) = e.value
@inline _device_grad_eval(e::DeviceSlotExpr, slots, pidx, b) = @inbounds slots[e.slot, pidx, b]
@inline _device_grad_eval(e::DevicePrimitiveExpr{Op}, slots, pidx, b) where {Op} =
    _device_apply(Val(Op), _device_grad_eval_args(e.args, slots, pidx, b)...)

@inline _device_grad_eval_args(::Tuple{}, slots, pidx, b) = ()
@inline _device_grad_eval_args(t::Tuple, slots, pidx, b) =
    (_device_grad_eval(first(t), slots, pidx, b), _device_grad_eval_args(Base.tail(t), slots, pidx, b)...)

@inline function _device_grad_store_binding!(slots, binding_slot::Int32, value, pidx, b)
    if binding_slot > Int32(0)
        @inbounds slots[binding_slot, pidx, b] = value
    end
    return nothing
end

# Returns (value_dual, logabsdet_dual, new_cursor). Latent values seed derivative 1 on
# the differentiation target row `pidx`; observed values are constants (derivative 0).
@inline function _device_grad_choice_value(step, params, observed, pidx, b, cursor::Int32, ::Type{TD}) where {TD}
    if step.value_source > Int32(0)
        raw = @inbounds params[step.value_source, b]
        u = _seed_latent(TD, raw, step.value_source, pidx)
        c, lad = _device_transform(step.transform, u)
        return (c, lad, cursor)
    else
        v = _obsval(observed, cursor, b)
        return (_seed_obs(TD, v), _seed_obs(TD, zero(_device_dual_basetype(TD))), cursor + Int32(1))
    end
end

# ---- per-step dual scoring ------------------------------------------------------

@inline function _device_grad_score_step(
    step::DeviceNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, v = promote(mu, sigma, value)
    return (_device_normal_logpdf(m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceNoncenteredNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    raw = @inbounds params[step.value_source, b]
    z = _seed_latent(eltype(slots), raw, step.value_source, pidx)
    m, s, zz = promote(mu, sigma, z)
    _device_grad_store_binding!(slots, step.binding_slot, m + s * zz, pidx, b)
    return (_device_normal_logpdf(zero(zz), one(zz), zz), cursor)
end

@inline function _device_grad_score_step(
    step::DeviceLognormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, v = promote(mu, sigma, value)
    return (_device_lognormal_logpdf(m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceExponentialChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    rate = _device_grad_eval(step.rate, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    r, v = promote(rate, value)
    return (_device_exponential_logpdf(r, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceGammaChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    shape = _device_grad_eval(step.shape, slots, pidx, b)
    rate = _device_grad_eval(step.rate, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    sh, r, v = promote(shape, rate, value)
    return (_device_gamma_logpdf(sh, r, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceLaplaceChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    loc = _device_grad_eval(step.loc, slots, pidx, b)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    l, s, v = promote(loc, scale, value)
    return (_device_laplace_logpdf(l, s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceCauchyChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, v = promote(mu, sigma, value)
    return (_device_cauchy_logpdf(m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceHalfNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    s, v = promote(sigma, value)
    return (_device_halfnormal_logpdf(s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceHalfCauchyChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    s, v = promote(scale, value)
    return (_device_halfcauchy_logpdf(s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceLogisticChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, v = promote(mu, scale, value)
    return (_device_logistic_logpdf(m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceGumbelChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, v = promote(mu, scale, value)
    return (_device_gumbel_logpdf(m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceFrechetChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    shape = _device_grad_eval(step.shape, slots, pidx, b)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    sh, sc, v = promote(shape, scale, value)
    return (_device_frechet_logpdf(sh, sc, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceRayleighChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    s, v = promote(scale, value)
    return (_device_rayleigh_logpdf(s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceInverseGaussianChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    lambda = _device_grad_eval(step.lambda, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, l, v = promote(mu, lambda, value)
    return (_device_inversegaussian_logpdf(m, l, v) + lad, cur)
end

@inline function _device_grad_score_step(step::DeviceBetaChoiceStep, slots, params, observed, observed_int, tc, ls, pidx, b, cursor)
    alpha = _device_grad_eval(step.alpha, slots, pidx, b)
    beta = _device_grad_eval(step.beta, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    al, be, v = promote(alpha, beta, value)
    return (_device_beta_logpdf(al, be, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceStudentTChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    nu = _device_grad_eval(step.nu, slots, pidx, b)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    n, m, s, v = promote(nu, mu, sigma, value)
    return (_device_studentt_logpdf(n, m, s, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceInverseGammaChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    shape = _device_grad_eval(step.shape, slots, pidx, b)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    sh, sc, v = promote(shape, scale, value)
    return (_device_inversegamma_logpdf(sh, sc, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceWeibullChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    shape = _device_grad_eval(step.shape, slots, pidx, b)
    scale = _device_grad_eval(step.scale, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    sh, sc, v = promote(shape, scale, value)
    return (_device_weibull_logpdf(sh, sc, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceBinomialChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    p = _device_grad_eval(step.probability, slots, pidx, b)
    # trials `n` is an exact integer staged as a leading observation row (issue
    # #71); the -1 sentinel means the host could not resolve it (a deterministic
    # binding), so fall back to the in-kernel float evaluation.
    n_staged = _obsint(observed_int, cursor, b)
    n = n_staged >= 0 ? n_staged : _device_count_int(_device_dual_value(_device_grad_eval(step.trials, slots, pidx, b)))
    cur = cursor + Int32(1)
    if step.value_source > Int32(0)
        value, lad, _ = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
        _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
        k = _device_count_int(_device_dual_value(value))
        return (_device_binomial_logpdf(n, k, p) + lad, cur)
    end
    k = _obsint(observed_int, cur, b)
    value = _seed_obs(eltype(slots), _obsval(observed, cur, b))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    return (_device_binomial_logpdf(n, k, p), cur + Int32(1))
end

@inline function _device_grad_score_step(
    step::DeviceGeometricChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    p = _device_grad_eval(step.probability, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    pp, v = promote(p, value)
    return (_device_geometric_logpdf(pp, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceNegativeBinomialChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    successes = _device_grad_eval(step.successes, slots, pidx, b)
    p = _device_grad_eval(step.probability, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    r, pp, v = promote(successes, p, value)
    return (_device_negativebinomial_logpdf(r, pp, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceCategoricalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    probabilities = _device_grad_eval_args(step.probabilities, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    # `_device_categorical_logpdf` promotes the heterogeneous probability tuple
    # internally, so only the value dual drives the working type here.
    return (_device_categorical_logpdf(probabilities, value) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceBernoulliChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    p = _device_grad_eval(step.probability, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    pp, v = promote(p, value)
    return (_device_bernoulli_logpdf(pp, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceBernoulliLogitChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    eta = _device_grad_eval(step.eta, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    e, v = promote(eta, value)
    return (_device_bernoullilogit_logpdf(e, v) + lad, cur)
end

# Fused GLM linear predictor in duals: `eta = intercept + sum_d coef[d]*X[d]`.
# Each coefficient dual seeds derivative 1 iff its unconstrained row is the
# thread's differentiation target `pidx`, so forward-mode reproduces the CPU
# analytic `d_eta/d_coef[d] = X[d, index]` seed; the intercept dual carries
# `d_eta/d_intercept = 1` through its slot read. The covariate column is constant
# data (zero derivative). The whole density (value + gradient) differentiates
# through `_device_bernoullilogit_logpdf`, giving `d/d_eta = y - logistic(eta)`.
@inline function _device_grad_score_step(
    step::DeviceBernoulliLogitGLMChoiceStep{D},
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
) where {D}
    TD = eltype(slots)
    intercept = _device_grad_eval(step.intercept, slots, pidx, b)
    terms = ntuple(Val(D)) do i
        row = step.coef_value_source + Int32(i - 1)
        raw = @inbounds params[row, b]
        coef = _seed_latent(TD, raw, row, pidx)
        x = _obsval(observed, cursor + Int32(i - 1), b)
        coef * x
    end
    eta = intercept + _device_tuple_sum(terms)
    cur = cursor + Int32(D)
    y = _seed_obs(TD, _obsval(observed, cur, b))
    _device_grad_store_binding!(slots, step.binding_slot, y, pidx, b)
    e, v = promote(eta, y)
    return (_device_bernoullilogit_logpdf(e, v), cur + Int32(1))
end

# Runtime-length fused GLM gradient with the in-kernel ANALYTIC form (issue #221).
# Rather than seeding a dual per coefficient and differentiating the density
# through D dual FMAs (the unrolled `{D}` method above), compute `eta` once by a
# runtime value loop, form the scalar `derivative = d/d_eta log p = y -
# logistic(eta)`, and fan it to the parameter derivatives: `d/d_coef[d] =
# derivative * X[d]` (a one-hot at the coefficient's row) plus `derivative *
# d_eta/d_intercept` through the intercept dual. This is the exact analytic form
# the host backend uses (#150) and removes both the D-wide value unroll and the
# per-coefficient dual arithmetic from the gradient kernel, so any D compiles.
@inline function _device_grad_score_step(
    step::DeviceBernoulliLogitGLMChoiceStepDyn,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    TD = eltype(slots)
    T = _device_dual_basetype(TD)
    intercept = convert(TD, _device_grad_eval(step.intercept, slots, pidx, b))
    eta = intercept.value
    @inbounds for i = Int32(1):step.coef_length
        eta +=
            T(params[step.coef_value_source+i-Int32(1), b]) * T(_obsval(observed, cursor + i - Int32(1), b))
    end
    cur = cursor + step.coef_length
    yval = T(_obsval(observed, cur, b))
    _device_grad_store_binding!(slots, step.binding_slot, _seed_obs(TD, yval), pidx, b)
    support = (yval == zero(T)) | (yval == one(T))
    derivative = ifelse(support, yval - _device_logistic(eta), zero(T))
    logpdf = _device_bernoullilogit_logpdf(eta, yval)
    return (_glm_analytic_result(TD, logpdf, derivative, intercept, step, observed, cursor, b, pidx), cur + Int32(1))
end

# Scalar walk: only the covariate at the differentiation target row `pidx` (if it
# is one of the coefficient rows) adds to `d_eta/d_pidx`; the intercept dual
# carries the rest.
@inline function _glm_analytic_result(
    ::Type{DeviceDual{T}},
    logpdf::T,
    derivative::T,
    intercept::DeviceDual{T},
    step,
    observed,
    cursor::Int32,
    b,
    pidx,
) where {T}
    d = Int32(pidx) - step.coef_value_source + Int32(1)
    in_range = (d >= Int32(1)) & (d <= step.coef_length)
    safe = ifelse(in_range, d, Int32(1))
    x = T(_obsval(observed, cursor + safe - Int32(1), b))
    deta = intercept.deriv + ifelse(in_range, x, zero(T))
    return DeviceDual{T}(logpdf, derivative * deta)
end

# Wide walk: fan the covariate into each partial slot that is a coefficient row.
# The masked read stays in-bounds for non-coefficient slots (safe index + zero
# mask), so the P-wide tuple builds without a per-coefficient dual chain.
@inline function _glm_analytic_result(
    ::Type{DeviceGradN{N,T}},
    logpdf::T,
    derivative::T,
    intercept::DeviceGradN{N,T},
    step,
    observed,
    cursor::Int32,
    b,
    pidx,
) where {N,T}
    base = intercept.partials
    partials = ntuple(Val(N)) do i
        d = Int32(i) - step.coef_value_source + Int32(1)
        in_range = (d >= Int32(1)) & (d <= step.coef_length)
        safe = ifelse(in_range, d, Int32(1))
        x = T(_obsval(observed, cursor + safe - Int32(1), b))
        @inbounds (base[i] + ifelse(in_range, x, zero(T))) * derivative
    end
    return DeviceGradN{N,T}(logpdf, partials)
end

# Runtime-length iid diagonal-normal prior gradient (issue #221): a runtime loop
# over the D latent rows, seeding each as it is read, so the dual channel widens
# per row without a compile-time `Val(D)` unroll. `mu`/`sigma` flow through as
# duals (constant for a literal iid prior), matching DeviceMvNormalChoiceStep.
@inline function _device_grad_score_step(
    step::DeviceDiagNormalChoiceStepDyn,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    TD = eltype(slots)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    v1 = _seed_latent(TD, (@inbounds params[step.value_source, b]), step.value_source, pidx)
    total = _device_normal_logpdf(promote(mu, sigma, v1)...)
    @inbounds for i = Int32(2):step.dimension
        row = step.value_source + i - Int32(1)
        v = _seed_latent(TD, params[row, b], row, pidx)
        total += _device_normal_logpdf(promote(mu, sigma, v)...)
    end
    return (total, cursor)
end

# Broadcast (vectorized) normal observation in duals (issue #134): the exact
# gradient-kernel analog of the logjoint handler. mu/sigma scalar leaves read the
# dual `slots` buffer (a scalar latent binding carries its seed from that latent's
# choice step, so `d/d(param)` flows through), covariate leaves read the constant
# observation column (zero derivative), and `y` is a `_seed_obs` constant. The
# whole per-element `normal` logpdf differentiates through the duals, reproducing
# the host analytic broadcast gradient (`dmu = z/sigma`, `dsigma = (z^2-1)/sigma`
# times the argument gradients) family by family.
@inline _device_grad_bcast_eval(e::DeviceLiteralExpr, slots, observed, elem_base, pidx, b) = e.value
@inline _device_grad_bcast_eval(e::DeviceSlotExpr, slots, observed, elem_base, pidx, b) =
    @inbounds slots[e.slot, pidx, b]
@inline _device_grad_bcast_eval(e::DeviceObservedColumnExpr, slots, observed, elem_base, pidx, b) =
    _obsval(observed, elem_base + e.offset, b)
@inline _device_grad_bcast_eval(e::DevicePrimitiveExpr{Op}, slots, observed, elem_base, pidx, b) where {Op} =
    _device_apply(Val(Op), _device_grad_bcast_eval_args(e.args, slots, observed, elem_base, pidx, b)...)

@inline _device_grad_bcast_eval_args(::Tuple{}, slots, observed, elem_base, pidx, b) = ()
@inline _device_grad_bcast_eval_args(t::Tuple, slots, observed, elem_base, pidx, b) = (
    _device_grad_bcast_eval(first(t), slots, observed, elem_base, pidx, b),
    _device_grad_bcast_eval_args(Base.tail(t), slots, observed, elem_base, pidx, b)...,
)

@inline function _device_grad_score_step(
    step::DeviceBroadcastNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    TD = eltype(slots)
    m = @inbounds tc[step.count_id]
    y_offset = step.stride - Int32(1)
    total = zero(TD)
    cur = cursor
    for _ = Int32(1):m
        mu = _device_grad_bcast_eval(step.mu, slots, observed, cur, pidx, b)
        sigma = _device_grad_bcast_eval(step.sigma, slots, observed, cur, pidx, b)
        y = _seed_obs(TD, _obsval(observed, cur + y_offset, b))
        mm, ss, vv = promote(mu, sigma, y)
        total += _device_normal_logpdf(mm, ss, vv)
        cur += step.stride
    end
    return (total, cur)
end

@inline function _device_grad_score_step(
    step::DevicePoissonChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    lambda = _device_grad_eval(step.lambda, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    lam, v = promote(lambda, value)
    return (_device_poisson_logpdf(lam, v) + lad, cur)
end

# Vector choice value in duals: latent component i seeds derivative 1 iff its
# unconstrained row is the thread's differentiation target.
@inline function _device_grad_vector_choice_value(
    step,
    params,
    observed,
    pidx,
    b,
    cursor::Int32,
    ::Type{TD},
    ::Val{D},
) where {TD,D}
    if step.value_source > Int32(0)
        value = ntuple(Val(D)) do i
            row = step.value_source + Int32(i - 1)
            raw = @inbounds params[row, b]
            _seed_latent(TD, raw, row, pidx)
        end
        return (value, cursor)
    end
    value = ntuple(Val(D)) do i
        _seed_obs(TD, _obsval(observed, cursor + Int32(i - 1), b))
    end
    return (value, cursor + Int32(D))
end

@inline function _device_grad_score_step(
    step::DeviceDirichletChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    TD = eltype(slots)
    alpha = _device_grad_eval_args(step.alpha, slots, pidx, b)
    if step.value_source > Int32(0)
        z = ntuple(Val(length(step.alpha) - 1)) do i
            row = step.value_source + Int32(i - 1)
            raw = @inbounds params[row, b]
            _seed_latent(TD, raw, row, pidx)
        end
        # duals flow through the register softmax, reproducing the CPU analytic
        # simplex Jacobian and log-abs-det derivative
        value, lad = _device_simplex_constrain(z)
        return (_device_dirichlet_logpdf(alpha, value) + lad, cursor)
    end
    value = ntuple(Val(length(step.alpha))) do i
        _seed_obs(TD, _obsval(observed, cursor + Int32(i - 1), b))
    end
    return (_device_dirichlet_logpdf(alpha, value), cursor + Int32(length(step.alpha)))
end

@inline function _device_grad_score_step(
    step::DeviceLKJCholeskyChoiceStep{D},
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
) where {D}
    TD = eltype(slots)
    eta = _device_grad_eval(step.eta, slots, pidx, b)
    if step.value_source > Int32(0)
        z = ntuple(Val((D * (D - 1)) ÷ 2)) do i
            row = step.value_source + Int32(i - 1)
            raw = @inbounds params[row, b]
            _seed_latent(TD, raw, row, pidx)
        end
        # duals flow through the register tanh/stick constrain, reproducing
        # the CPU analytic z-space gradient and log-abs-det derivative
        value, lad = _device_cholesky_corr_constrain(z, Val(D))
        return (_device_lkjcholesky_logpdf(eta, value, Val(D)) + lad, cursor)
    end
    value = ntuple(Val((D * (D + 1)) ÷ 2)) do i
        _seed_obs(TD, _obsval(observed, cursor + Int32(i - 1), b))
    end
    return (_device_lkjcholesky_logpdf(eta, value, Val(D)), cursor + Int32((D * (D + 1)) ÷ 2))
end

@inline function _device_grad_score_step(
    step::DeviceMvNormalDenseChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    TD = eltype(slots)
    mu = _device_grad_eval_args(step.mu, slots, pidx, b)
    # the factor is constant data (zero derivative), matching the CPU contract
    scale_packed = ntuple(
        i -> _seed_obs(TD, _obsval(observed, cursor + Int32(i - 1), b)),
        Val((length(step.mu) * (length(step.mu) + 1)) ÷ 2),
    )
    cur = cursor + Int32((length(step.mu) * (length(step.mu) + 1)) ÷ 2)
    value, cur2 =
        _device_grad_vector_choice_value(step, params, observed, pidx, b, cur, eltype(slots), Val(length(step.mu)))
    return (_device_mvnormaldense_logpdf(scale_packed, mu, value), cur2)
end

@inline function _device_grad_score_step(
    step::DeviceMvNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval_args(step.mu, slots, pidx, b)
    sigma = _device_grad_eval_args(step.sigma, slots, pidx, b)
    value, cur =
        _device_grad_vector_choice_value(step, params, observed, pidx, b, cursor, eltype(slots), Val(length(step.mu)))
    return (_device_mvnormal_logpdf_fold(mu, sigma, value), cur)
end

@inline function _device_grad_score_step(
    step::DeviceMvStudentTChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    nu = _device_grad_eval(step.nu, slots, pidx, b)
    mu = _device_grad_eval_args(step.mu, slots, pidx, b)
    sigma = _device_grad_eval_args(step.sigma, slots, pidx, b)
    value, cur =
        _device_grad_vector_choice_value(step, params, observed, pidx, b, cursor, eltype(slots), Val(length(step.mu)))
    return (_device_mvstudentt_diag_logpdf(nu, mu, sigma, value), cur)
end

@inline function _device_grad_score_step(
    step::DeviceMvStudentTDenseChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    TD = eltype(slots)
    nu = _device_grad_eval(step.nu, slots, pidx, b)
    mu = _device_grad_eval_args(step.mu, slots, pidx, b)
    scale_packed = ntuple(
        i -> _seed_obs(TD, _obsval(observed, cursor + Int32(i - 1), b)),
        Val((length(step.mu) * (length(step.mu) + 1)) ÷ 2),
    )
    cur = cursor + Int32((length(step.mu) * (length(step.mu) + 1)) ÷ 2)
    value, cur2 =
        _device_grad_vector_choice_value(step, params, observed, pidx, b, cur, eltype(slots), Val(length(step.mu)))
    return (_device_mvstudentt_dense_logpdf(nu, scale_packed, mu, value), cur2)
end

@inline function _device_grad_score_step(
    step::DeviceTruncatedNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    lower = _device_grad_eval(step.lower, slots, pidx, b)
    upper = _device_grad_eval(step.upper, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    m, s, lo, up, v = promote(mu, sigma, lower, upper, value)
    return (_device_truncatednormal_logpdf(m, s, lo, up, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceTruncatedStudentTChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    nu = _device_grad_eval(step.nu, slots, pidx, b)
    mu = _device_grad_eval(step.mu, slots, pidx, b)
    sigma = _device_grad_eval(step.sigma, slots, pidx, b)
    lower = _device_grad_eval(step.lower, slots, pidx, b)
    upper = _device_grad_eval(step.upper, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    # nu is a literal: promoting it yields a zero-derivative dual, so the omitted
    # d/dnu term stays genuinely zero (the t-CDF dual ignores the nu channel).
    n, m, s, lo, up, v = promote(nu, mu, sigma, lower, upper, value)
    return (_device_truncatedstudentt_logpdf(n, m, s, lo, up, v) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceMixtureNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    weights = _device_grad_eval_args(step.weights, slots, pidx, b)
    mus = _device_grad_eval_args(step.mus, slots, pidx, b)
    sigmas = _device_grad_eval_args(step.sigmas, slots, pidx, b)
    value, lad, cur = _device_grad_choice_value(step, params, observed, pidx, b, cursor, eltype(slots))
    _device_grad_store_binding!(slots, step.binding_slot, value, pidx, b)
    # the log-sum-exp promotes per component internally
    return (_device_mixture_normal_logpdf(weights, mus, sigmas, value) + lad, cur)
end

@inline function _device_grad_score_step(
    step::DeviceDeterministicStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    @inbounds slots[step.binding_slot, pidx, b] = _device_grad_eval(step.expr, slots, pidx, b)
    return (zero(eltype(slots)), cursor)
end

@inline function _device_grad_score_step(step::DeviceLoopStep, slots, params, observed, observed_int, tc, ls, pidx, b, cursor)
    TD = eltype(slots)
    count = @inbounds tc[step.loop_id]
    start = @inbounds ls[step.loop_id]
    total = zero(TD)
    cur = cursor
    for t = Int32(0):(count-Int32(1))
        if step.iterator_slot > Int32(0)
            @inbounds slots[step.iterator_slot, pidx, b] = TD(start + t)
        end
        contribution, cur = _device_grad_score_steps(step.body, slots, params, observed, observed_int, tc, ls, pidx, b, cur)
        total += contribution
    end
    return (total, cur)
end

# ---- marginalize=:enumerate in duals (issue #67) -------------------------------
#
# The dual analog of the logjoint scan: duals flow through the branch bodies and
# the log-sum-exp fold, reproducing the CPU responsibility-weighted gradient
# `d logsumexp = sum_v r_v * d(log pmf_v + suffix_v)`. The branch value is a
# ZERO-derivative constant dual (`TD(v)`), so a downstream suffix read of the
# binding picks up a discrete branch constant, matching the CPU cleared
# slot-gradient plane. Impossible/zero-mass branches are excluded by their FULL
# term inside `_device_marg_combine` (the #62 lesson) so a degenerate branch
# derivative cannot contaminate the marginal gradient.
@inline function _device_marg_grad_scan(
    binding::Int32,
    vf::Val,
    probs,
    support::Tuple{A},
    body,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
) where {A}
    TD = eltype(slots)
    v = TD(first(support))
    _device_grad_store_binding!(slots, binding, v, pidx, b)
    lp = _device_marg_logpmf(vf, probs, v)
    suffix, cur = _device_grad_score_steps(body, slots, params, observed, observed_int, tc, ls, pidx, b, cursor)
    return ((lp + suffix,), cur)
end
@inline function _device_marg_grad_scan(
    binding::Int32,
    vf::Val,
    probs,
    support::Tuple,
    body,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
)
    TD = eltype(slots)
    v = TD(first(support))
    _device_grad_store_binding!(slots, binding, v, pidx, b)
    lp = _device_marg_logpmf(vf, probs, v)
    suffix, cur = _device_grad_score_steps(body, slots, params, observed, observed_int, tc, ls, pidx, b, cursor)
    rest, _ = _device_marg_grad_scan(
        binding, vf, probs, Base.tail(support), body, slots, params, observed, observed_int, tc, ls, pidx, b, cursor,
    )
    return ((lp + suffix, rest...), cur)
end

@inline function _device_grad_score_step(
    step::DeviceMarginalizeChoiceStep{F},
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    pidx,
    b,
    cursor,
) where {F}
    selector = _obsval(observed, cursor, b)
    branch_cursor = cursor + Int32(1)
    probs = _device_grad_eval_args(step.probabilities, slots, pidx, b)
    totals, cur = _device_marg_grad_scan(
        step.binding_slot, Val(F), probs, step.support, step.body, slots, params, observed, observed_int, tc, ls, pidx, b,
        branch_cursor,
    )
    return (_device_marg_combine(totals, selector), cur)
end

# ---- recursive dual step-tuple walk --------------------------------------------

@inline _device_grad_score_steps(::Tuple{}, slots, params, observed, observed_int, tc, ls, pidx, b, cursor) =
    (zero(eltype(slots)), cursor)

@inline function _device_grad_score_steps(steps::Tuple, slots, params, observed, observed_int, tc, ls, pidx, b, cursor)
    contribution, cur = _device_grad_score_step(first(steps), slots, params, observed, observed_int, tc, ls, pidx, b, cursor)
    rest, cur2 = _device_grad_score_steps(Base.tail(steps), slots, params, observed, observed_int, tc, ls, pidx, b, cur)
    return (contribution + rest, cur2)
end

# ---- the kernel ----------------------------------------------------------------

@kernel function _device_gradient_kernel!(
    totals,
    gradients,
    plan,
    @Const(params),
    @Const(observed),
    @Const(observed_int),
    slots,
    @Const(trip_counts),
    @Const(loop_starts),
)
    idx = @index(Global, NTuple)
    pidx = idx[1]
    b = idx[2]
    total, _ = _device_grad_score_steps(
        plan.steps, slots, params, observed, observed_int, trip_counts, loop_starts, pidx, b, Int32(1),
    )
    @inbounds gradients[pidx, b] = _device_dual_deriv(total)
    if pidx == 1
        @inbounds totals[b] = _device_dual_value(total)
    end
end
