# The one fused KernelAbstractions logjoint kernel plus the device-side, fully
# inlined interpreter of the device execution plan. One thread == one batch column.
#
# The plan is passed to the kernel by value (it is `isbits`); its step tuple type
# specializes the kernel, so the recursive walk unrolls at compile time. Per-column
# scratch lives in a `slots` matrix (device buffer). Observed values are read from
# `observed[cursor, col]`, with the cursor threaded through the recursion exactly as
# staging assigned rows.

# ---- observation buffer access -------------------------------------------------
#
# Staged observations are shared across the batch whenever the model arguments are
# shared (a single constraint set applied to every chain): the covariate matrix `X`
# and the observed `y` are then identical in every column, so staging keeps a SINGLE
# column and every chain reads it (issue #153). A genuinely per-chain staging keeps
# `C` columns, read at column `b`. The branch is a cheap register compare on the
# staged column count; `observed` / `observed_int` may have different column counts
# (e.g. `observed_int` is a `(1,1)` dummy when no count-family step reads it), so each
# is resolved independently.

@inline _device_obs_col(observed, b) = ifelse(size(observed, 2) == 1, one(b), b)
@inline _obsval(observed, cursor, b) = @inbounds observed[cursor, _device_obs_col(observed, b)]
@inline _obsint(observed_int, cursor, b) = @inbounds observed_int[cursor, _device_obs_col(observed_int, b)]

# ---- expression evaluation -----------------------------------------------------

@inline _device_eval(e::DeviceLiteralExpr, slots, col) = e.value
@inline _device_eval(e::DeviceSlotExpr, slots, col) = @inbounds slots[e.slot, col]
@inline _device_eval(e::DevicePrimitiveExpr{Op}, slots, col) where {Op} =
    _device_apply(Val(Op), _device_eval_args(e.args, slots, col)...)

@inline _device_eval_args(::Tuple{}, slots, col) = ()
@inline _device_eval_args(t::Tuple, slots, col) =
    (_device_eval(first(t), slots, col), _device_eval_args(Base.tail(t), slots, col)...)

@inline _device_apply(::Val{:+}, args...) = +(args...)
@inline _device_apply(::Val{:-}, args...) = -(args...)
@inline _device_apply(::Val{:*}, args...) = *(args...)
@inline _device_apply(::Val{:/}, a, b) = a / b
@inline _device_apply(::Val{:^}, a, b) = a^b
@inline _device_apply(::Val{:exp}, a) = exp(a)
@inline _device_apply(::Val{:log}, a) = log(a)
@inline _device_apply(::Val{:log1p}, a) = log1p(a)
@inline _device_apply(::Val{:sqrt}, a) = sqrt(a)
@inline _device_apply(::Val{:abs}, a) = abs(a)
@inline _device_apply(::Val{:min}, args...) = min(args...)
@inline _device_apply(::Val{:max}, args...) = max(args...)
@inline _device_apply(::Val{:clamp}, a, b, c) = clamp(a, b, c)

# ---- choice value resolution ---------------------------------------------------

@inline function _device_store_binding!(slots, binding_slot::Int32, value, col)
    if binding_slot > Int32(0)
        @inbounds slots[binding_slot, col] = value
    end
    return nothing
end

# Returns (value, logabsdet, new_cursor).
@inline function _device_choice_value(step, params, observed, col, cursor::Int32)
    if step.value_source > Int32(0)
        u = @inbounds params[step.value_source, col]
        c, lad = _device_transform(step.transform, u)
        return (c, lad, cursor)
    else
        v = _obsval(observed, cursor, col)
        return (v, zero(v), cursor + Int32(1))
    end
end

# Vector choice value: a latent reads D consecutive unconstrained rows
# (VectorIdentity: no transform, log-abs-det 0), an observation consumes D
# staged rows and advances the cursor by D.
@inline function _device_vector_choice_value(step, params, observed, col, cursor::Int32, ::Val{D}) where {D}
    if step.value_source > Int32(0)
        value = ntuple(i -> @inbounds(params[step.value_source+Int32(i-1), col]), Val(D))
        return (value, cursor)
    end
    value = ntuple(i -> _obsval(observed, cursor + Int32(i - 1), col), Val(D))
    return (value, cursor + Int32(D))
end

# Compile-time-unrolled sum of per-component normal logpdfs (the diagonal
# mvnormal closed form); per-component promote keeps heterogeneous
# (literal/dual) argument tuples safe, mirroring the categorical fold.
@inline _device_mvnormal_logpdf_fold(mu::Tuple{}, sigma::Tuple{}, value::Tuple{}) = error("empty mvnormal")
@inline _device_mvnormal_logpdf_fold(mu::Tuple{A}, sigma::Tuple{B}, value::Tuple{C}) where {A,B,C} =
    _device_normal_logpdf(promote(mu[1], sigma[1], value[1])...)
@inline function _device_mvnormal_logpdf_fold(mu::Tuple, sigma::Tuple, value::Tuple)
    head = _device_normal_logpdf(promote(first(mu), first(sigma), first(value))...)
    return head + _device_mvnormal_logpdf_fold(Base.tail(mu), Base.tail(sigma), Base.tail(value))
end

# ---- per-step scoring ----------------------------------------------------------

@inline function _device_score_step(step::DeviceNormalChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_normal_logpdf(mu, sigma, value) + lad, cur)
end

@inline function _device_score_step(
    step::DeviceNoncenteredNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    z = @inbounds params[step.value_source, col]
    _device_store_binding!(slots, step.binding_slot, mu + sigma * z, col)
    return (_device_normal_logpdf(zero(z), one(z), z), cursor)
end

@inline function _device_score_step(step::DeviceLognormalChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_lognormal_logpdf(mu, sigma, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceExponentialChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    rate = _device_eval(step.rate, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_exponential_logpdf(rate, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceGammaChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    shape = _device_eval(step.shape, slots, col)
    rate = _device_eval(step.rate, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_gamma_logpdf(shape, rate, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceLaplaceChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    loc = _device_eval(step.loc, slots, col)
    scale = _device_eval(step.scale, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_laplace_logpdf(loc, scale, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceBetaChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    alpha = _device_eval(step.alpha, slots, col)
    beta = _device_eval(step.beta, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_beta_logpdf(alpha, beta, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceStudentTChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    nu = _device_eval(step.nu, slots, col)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_studentt_logpdf(nu, mu, sigma, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceInverseGammaChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    shape = _device_eval(step.shape, slots, col)
    scale = _device_eval(step.scale, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_inversegamma_logpdf(shape, scale, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceWeibullChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    shape = _device_eval(step.shape, slots, col)
    scale = _device_eval(step.scale, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_weibull_logpdf(shape, scale, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceBinomialChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    p = _device_eval(step.probability, slots, col)
    # trials `n` is an exact integer staged as a leading observation row (issue
    # #71); the -1 sentinel means the host could not resolve it (a deterministic
    # binding), so fall back to the in-kernel float evaluation.
    n_staged = _obsint(observed_int, cursor, col)
    n = n_staged >= 0 ? n_staged : _device_count_int(_device_dual_value(_device_eval(step.trials, slots, col)))
    cur = cursor + Int32(1)
    if step.value_source > Int32(0)
        u = @inbounds params[step.value_source, col]
        value, lad = _device_transform(step.transform, u)
        _device_store_binding!(slots, step.binding_slot, value, col)
        k = _device_count_int(_device_dual_value(value))
        return (_device_binomial_logpdf(n, k, p) + lad, cur)
    end
    k = _obsint(observed_int, cur, col)
    value = _obsval(observed, cur, col)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_binomial_logpdf(n, k, p), cur + Int32(1))
end

@inline function _device_score_step(step::DeviceGeometricChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    p = _device_eval(step.probability, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_geometric_logpdf(p, value) + lad, cur)
end

@inline function _device_score_step(
    step::DeviceNegativeBinomialChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
)
    successes = _device_eval(step.successes, slots, col)
    p = _device_eval(step.probability, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_negativebinomial_logpdf(successes, p, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceCategoricalChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    probabilities = _device_eval_args(step.probabilities, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_categorical_logpdf(probabilities, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceBernoulliChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    p = _device_eval(step.probability, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_bernoulli_logpdf(p, value) + lad, cur)
end

@inline function _device_score_step(
    step::DeviceBernoulliLogitChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
)
    eta = _device_eval(step.eta, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    e, v = promote(eta, value)
    return (_device_bernoullilogit_logpdf(e, v) + lad, cur)
end

# Fused GLM linear predictor `eta = intercept + sum_d coef[d] * X[d, index]`
# scored as a logit-Bernoulli observation. The covariate column `X[:, index]`
# was staged into the observation buffer as `D` leading rows (staging.jl), then
# the observed y row: read them cursor-first. The coefficient values come
# straight from the unconstrained params (VectorIdentity: constrained ==
# unconstrained), NOT from a materialized slot.
@inline function _device_score_step(
    step::DeviceBernoulliLogitGLMChoiceStep{D},
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
) where {D}
    intercept = _device_eval(step.intercept, slots, col)
    terms = ntuple(
        i -> @inbounds(params[step.coef_value_source+Int32(i-1), col]) * _obsval(observed, cursor + Int32(i - 1), col),
        Val(D),
    )
    eta = intercept + _device_tuple_sum(terms)
    cur = cursor + Int32(D)
    value = _obsval(observed, cur, col)
    _device_store_binding!(slots, step.binding_slot, value, col)
    e, v = promote(eta, value)
    return (_device_bernoullilogit_logpdf(e, v), cur + Int32(1))
end

# Broadcast (vectorized) normal observation `{:y} ~ normal.(mu_expr, sigma)` --
# the flagship GPU-lowering form (issue #134). A dense per-element fold over the
# vector observation `y[1..M]`: read `M` from the shared trip-count buffer, then
# for each element read its `C` covariate rows and `y` row from the observation
# buffer (`stride == C + 1` rows per element, staged cursor-first) and add the
# per-element `normal(mu_i, sigma_i)` logpdf. Scalar mu/sigma leaves read the
# materialized `slots` buffer (broadcast across every element); covariate leaves
# read `elem_base + offset`, matching the host `_score_backend_step!` broadcast fold.
@inline _device_bcast_eval(e::DeviceLiteralExpr, slots, observed, elem_base, col) = e.value
@inline _device_bcast_eval(e::DeviceSlotExpr, slots, observed, elem_base, col) = @inbounds slots[e.slot, col]
@inline _device_bcast_eval(e::DeviceObservedColumnExpr, slots, observed, elem_base, col) =
    _obsval(observed, elem_base + e.offset, col)
@inline _device_bcast_eval(e::DevicePrimitiveExpr{Op}, slots, observed, elem_base, col) where {Op} =
    _device_apply(Val(Op), _device_bcast_eval_args(e.args, slots, observed, elem_base, col)...)

@inline _device_bcast_eval_args(::Tuple{}, slots, observed, elem_base, col) = ()
@inline _device_bcast_eval_args(t::Tuple, slots, observed, elem_base, col) = (
    _device_bcast_eval(first(t), slots, observed, elem_base, col),
    _device_bcast_eval_args(Base.tail(t), slots, observed, elem_base, col)...,
)

@inline function _device_score_step(
    step::DeviceBroadcastNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
)
    Tt = eltype(slots)
    m = @inbounds tc[step.count_id]
    y_offset = step.stride - Int32(1) # y row follows the C covariate rows
    total = zero(Tt)
    cur = cursor
    for _ = Int32(1):m
        mu = _device_bcast_eval(step.mu, slots, observed, cur, col)
        sigma = _device_bcast_eval(step.sigma, slots, observed, cur, col)
        y = _obsval(observed, cur + y_offset, col)
        total += _device_normal_logpdf(mu, sigma, y)
        cur += step.stride
    end
    # binding deliberately NOT stored (the observed vector has no scalar slot; the
    # lowering audit rejects any downstream read)
    return (total, cur)
end

@inline function _device_score_step(step::DevicePoissonChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    lambda = _device_eval(step.lambda, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_poisson_logpdf(lambda, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceDirichletChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    alpha = _device_eval_args(step.alpha, slots, col)
    if step.value_source > Int32(0)
        z = ntuple(i -> @inbounds(params[step.value_source+Int32(i-1), col]), Val(length(step.alpha) - 1))
        value, lad = _device_simplex_constrain(z)
        return (_device_dirichlet_logpdf(alpha, value) + lad, cursor)
    end
    value = ntuple(i -> _obsval(observed, cursor + Int32(i - 1), col), Val(length(step.alpha)))
    return (_device_dirichlet_logpdf(alpha, value), cursor + Int32(length(step.alpha)))
end

@inline function _device_score_step(
    step::DeviceLKJCholeskyChoiceStep{D},
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
) where {D}
    eta = _device_eval(step.eta, slots, col)
    if step.value_source > Int32(0)
        z = ntuple(i -> @inbounds(params[step.value_source+Int32(i-1), col]), Val((D * (D - 1)) ÷ 2))
        value, lad = _device_cholesky_corr_constrain(z, Val(D))
        return (_device_lkjcholesky_logpdf(eta, value, Val(D)) + lad, cursor)
    end
    value = ntuple(i -> _obsval(observed, cursor + Int32(i - 1), col), Val((D * (D + 1)) ÷ 2))
    return (_device_lkjcholesky_logpdf(eta, value, Val(D)), cursor + Int32((D * (D + 1)) ÷ 2))
end

@inline function _device_score_step(step::DeviceMvNormalDenseChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    mu = _device_eval_args(step.mu, slots, col)
    scale_packed = ntuple(
        i -> _obsval(observed, cursor + Int32(i - 1), col),
        Val((length(step.mu) * (length(step.mu) + 1)) ÷ 2),
    )
    cur = cursor + Int32((length(step.mu) * (length(step.mu) + 1)) ÷ 2)
    value, cur2 = _device_vector_choice_value(step, params, observed, col, cur, Val(length(step.mu)))
    return (_device_mvnormaldense_logpdf(scale_packed, mu, value), cur2)
end

@inline function _device_score_step(step::DeviceMvNormalChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    mu = _device_eval_args(step.mu, slots, col)
    sigma = _device_eval_args(step.sigma, slots, col)
    value, cur = _device_vector_choice_value(step, params, observed, col, cursor, Val(length(step.mu)))
    # binding deliberately NOT stored (vector bindings are unmaterialized; the
    # lowering audit rejects any downstream read)
    return (_device_mvnormal_logpdf_fold(mu, sigma, value), cur)
end

@inline function _device_score_step(
    step::DeviceTruncatedNormalChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    lower = _device_eval(step.lower, slots, col)
    upper = _device_eval(step.upper, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_truncatednormal_logpdf(mu, sigma, lower, upper, value) + lad, cur)
end

@inline function _device_score_step(
    step::DeviceTruncatedStudentTChoiceStep,
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
)
    nu = _device_eval(step.nu, slots, col)
    mu = _device_eval(step.mu, slots, col)
    sigma = _device_eval(step.sigma, slots, col)
    lower = _device_eval(step.lower, slots, col)
    upper = _device_eval(step.upper, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_truncatedstudentt_logpdf(nu, mu, sigma, lower, upper, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceMixtureNormalChoiceStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    weights = _device_eval_args(step.weights, slots, col)
    mus = _device_eval_args(step.mus, slots, col)
    sigmas = _device_eval_args(step.sigmas, slots, col)
    value, lad, cur = _device_choice_value(step, params, observed, col, cursor)
    _device_store_binding!(slots, step.binding_slot, value, col)
    return (_device_mixture_normal_logpdf(weights, mus, sigmas, value) + lad, cur)
end

@inline function _device_score_step(step::DeviceDeterministicStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    @inbounds slots[step.binding_slot, col] = _device_eval(step.expr, slots, col)
    return (zero(eltype(slots)), cursor)
end

@inline function _device_score_step(step::DeviceLoopStep, slots, params, observed, observed_int, tc, ls, col, cursor)
    Tt = eltype(slots)
    count = @inbounds tc[step.loop_id]
    start = @inbounds ls[step.loop_id]
    total = zero(Tt)
    cur = cursor
    for t = Int32(0):(count-Int32(1))
        if step.iterator_slot > Int32(0)
            @inbounds slots[step.iterator_slot, col] = Tt(start + t)
        end
        contribution, cur = _device_score_steps(step.body, slots, params, observed, observed_int, tc, ls, col, cur)
        total += contribution
    end
    return (total, cur)
end

# ---- marginalize=:enumerate (issue #67) ----------------------------------------
#
# Score one branch per compile-time support value, threading a tuple of per-branch
# full terms `log pmf(v) + suffix(v)`. Every branch re-scores the IDENTICAL suffix
# from the SAME cursor (so all advance it equally by construction); the returned
# cursor is one branch's -- they agree. The binding slot is written per branch so
# the suffix reads the branch value.
@inline function _device_marg_scan(
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
    col,
    cursor,
) where {A}
    Tt = eltype(slots)
    v = Tt(first(support))
    _device_store_binding!(slots, binding, v, col)
    lp = _device_marg_logpmf(vf, probs, v)
    suffix, cur = _device_score_steps(body, slots, params, observed, observed_int, tc, ls, col, cursor)
    return ((lp + suffix,), cur)
end
@inline function _device_marg_scan(
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
    col,
    cursor,
)
    Tt = eltype(slots)
    v = Tt(first(support))
    _device_store_binding!(slots, binding, v, col)
    lp = _device_marg_logpmf(vf, probs, v)
    suffix, cur = _device_score_steps(body, slots, params, observed, observed_int, tc, ls, col, cursor)
    rest, _ =
        _device_marg_scan(binding, vf, probs, Base.tail(support), body, slots, params, observed, observed_int, tc, ls, col, cursor)
    return ((lp + suffix, rest...), cur)
end

@inline function _device_score_step(
    step::DeviceMarginalizeChoiceStep{F},
    slots,
    params,
    observed,
    observed_int,
    tc,
    ls,
    col,
    cursor,
) where {F}
    selector = _obsval(observed, cursor, col) # 0 marginalize; v conditions branch v
    branch_cursor = cursor + Int32(1)
    probs = _device_eval_args(step.probabilities, slots, col)
    totals, cur = _device_marg_scan(
        step.binding_slot, Val(F), probs, step.support, step.body, slots, params, observed, observed_int, tc, ls, col,
        branch_cursor,
    )
    return (_device_marg_combine(totals, selector), cur)
end

# ---- recursive step-tuple walk -------------------------------------------------

@inline _device_score_steps(::Tuple{}, slots, params, observed, observed_int, tc, ls, col, cursor) =
    (zero(eltype(slots)), cursor)

@inline function _device_score_steps(steps::Tuple, slots, params, observed, observed_int, tc, ls, col, cursor)
    contribution, cur = _device_score_step(first(steps), slots, params, observed, observed_int, tc, ls, col, cursor)
    rest, cur2 = _device_score_steps(Base.tail(steps), slots, params, observed, observed_int, tc, ls, col, cur)
    return (contribution + rest, cur2)
end

# ---- the kernel ----------------------------------------------------------------

@kernel function _device_logjoint_kernel!(
    totals,
    plan,
    @Const(params),
    @Const(observed),
    @Const(observed_int),
    slots,
    @Const(trip_counts),
    @Const(loop_starts),
)
    col = @index(Global)
    total, _ =
        _device_score_steps(plan.steps, slots, params, observed, observed_int, trip_counts, loop_starts, col, Int32(1))
    @inbounds totals[col] = total
end
