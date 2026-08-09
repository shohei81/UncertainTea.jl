# Single-source scalar family kernels (issue #285, stage 1: continuous families).
#
# Each scalar family declares its math ONCE here, as two free functions of plain
# scalars, and every host path consumes them:
#
#   * `_backend_<family>_logpdf(params..., x)` -- the log density. The historical
#     `_backend_` prefix is retained so the many existing call sites (backend
#     scoring, batched gradients) stay untouched; these are THE reference kernels,
#     not backend-only code. Semantics: NaN for invalid parameters (the backend
#     paths score runtime parameter values that no constructor validated), -Inf
#     off the support, and the exact reference formula on it. Generic in the
#     element type, so ForwardDiff duals flow through.
#
#   * `_<family>_logpdf_partials(params..., x)` -- the hand-derived partial
#     derivatives, returned as a tuple in `(dvalue, dparams...)` channel order
#     (matching the batched-gradient accumulate loops). CONTRACT: inputs are
#     in-support, same-type floats; the CALLER guards the support (and any
#     boundary special cases, e.g. the weibull x == 0, shape == 1 scale channel)
#     exactly as before -- the kernels are pure in-support math.
#
# Consumers: the CPU-reference `logpdf(::Dist, x)` methods delegate to the logpdf
# kernels (src/distributions/continuous.jl); backend scoring calls them per step
# (src/backend/scoring/continuous.jl); the batched analytic gradients call both
# (src/batched/gradients/continuous.jl). The device path keeps its own
# SpecialFunctions-free, Float32-safe implementations (src/device/math.jl) --
# unifying those needs a special-function injection design (issue #285 stage 3).
#
# `test/uncertaintea/core/scalar_kernel_partials.jl` pins every partials kernel
# against ForwardDiff on the corresponding logpdf kernel.

# --- off-support score (issue #343) -------------------------------------------
#
# A latent flowing through a saturating transform can land EXACTLY on the
# support boundary in floating point: sigmoid(theta) rounds to 1.0 for
# theta >~ 36.74 (Float64), exp(theta) underflows to 0.0 below ~-745. The logpdf
# is then -Inf, but a plain `oftype(xx, -Inf)` return drops the ForwardDiff
# partials: the density term contributes ZERO gradient while the transform's
# (finite, exact since issue #105) log-abs-det term survives, so the
# unconstrained gradient comes out finite and silently wrong -- exactly the
# Jacobian derivative -- and gradient guards never fire. Off the support the
# one-sided derivative toward the interior is unbounded, so return -Inf with
# NaN in every partial channel that carries derivative information. Observed
# off-support values (plain floats, or duals with all-zero partials) keep the
# plain -Inf with clean zero partials, exactly as before.
_offsupport_neginf(x::Real) = oftype(x, -Inf)

function _offsupport_neginf(x::ForwardDiff.Dual{Tag,V,N}) where {Tag,V,N}
    parts = ForwardDiff.partials(x)
    poisoned = ForwardDiff.Partials{N,V}(
        ntuple(i -> iszero(parts[i]) ? zero(V) : convert(V, NaN), Val(N)),
    )
    return ForwardDiff.Dual{Tag,V,N}(convert(V, -Inf), poisoned)
end

# A positive value in the subnormal range (issue #345): exp(theta) lands there
# for theta in ~(-745.1, -708.4) before underflowing to exactly 0.0, and the
# few surviving mantissa bits make log(x) -- and hence the gamma density --
# silently wrong by O(1) while the 1/x partials explode. Treat it as the same
# boundary as exact 0.0. Branches on the primal value, so duals flow through.
#
# Issue #367 extends the same guard from gamma to every positive-support family
# whose density carries a log(x) or 1/x value term (lognormal, inversegamma,
# weibull, frechet, rayleigh, inversegaussian): in the subnormal band those
# terms are built from the few surviving mantissa bits (silently wrong values)
# while the 1/x partials overflow (Inf/NaN gradients that only ACCIDENTALLY
# reject). Rejecting the band deliberately also closes the underflow gap on the
# single ForwardDiff path: at exp(theta) == 0.0 the Dual partials underflow
# WITH the value (so `_offsupport_neginf` cannot poison, rejection rests on the
# -Inf value guard alone), but in the subnormal band the partials are still
# nonzero, so a trajectory moving toward underflow now hits poisoned (-Inf,
# NaN) evaluations before the partials vanish -- mirroring the batched path,
# which poisons at both boundaries. Exponential, halfnormal, halfcauchy, and
# pareto need no guard: their densities have no log(x)/1/x value term (pareto's
# lower-bounded transform maps the boundary INTO the support), so the kernels
# are exact through the band and at 0.0 (audited in issue #367).
_positive_subnormal(x::AbstractFloat) = issubnormal(x)
_positive_subnormal(x::Real) = issubnormal(float(x))
_positive_subnormal(x::ForwardDiff.Dual) = _positive_subnormal(ForwardDiff.value(x))

# --- logpdf kernels ----------------------------------------------------------

function _backend_normal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    _primal(sigma_) > zero(_primal(sigma_)) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return -log(sigma_) - log(2 * pi) / 2 - z * z / 2
end

function _backend_lognormal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    _primal(sigma_) > zero(_primal(sigma_)) || return oftype(xx, NaN)
    _primal(xx) > zero(_primal(xx)) || return _offsupport_neginf(xx)
    # exp-subnormal boundary (issues #345/#367): see _positive_subnormal
    _positive_subnormal(xx) && return _offsupport_neginf(xx)
    return _backend_normal_logpdf(mu_, sigma_, log(xx)) - log(xx)
end

function _backend_exponential_logpdf(rate, x)
    xx, rate_ = promote(x, rate)
    _primal(rate_) > zero(_primal(rate_)) || return oftype(xx, NaN)
    _primal(xx) >= zero(_primal(xx)) || return _offsupport_neginf(xx)
    return log(rate_) - rate_ * xx
end

function _backend_gamma_logpdf(shape, rate, x)
    xx, shape_, rate_ = promote(x, shape, rate)
    _primal(shape_) > zero(_primal(shape_)) || return oftype(xx, NaN)
    _primal(rate_) > zero(_primal(rate_)) || return oftype(xx, NaN)
    _primal(xx) > zero(_primal(xx)) || return _offsupport_neginf(xx)
    # exp-subnormal boundary (issue #345): a subnormal value scores as the
    # boundary (-Inf with poisoned partials), not a silently-wrong finite
    # density built from log of a few mantissa bits
    _positive_subnormal(xx) && return _offsupport_neginf(xx)
    return shape_ * log(rate_) - loggamma(shape_) + (shape_ - one(shape_)) * log(xx) - rate_ * xx
end

function _backend_studentt_logpdf(nu, mu, sigma, x)
    xx, nu_, mu_, sigma_ = promote(x, nu, mu, sigma)
    _primal(nu_) > zero(_primal(nu_)) || return oftype(xx, NaN)
    _primal(sigma_) > zero(_primal(sigma_)) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return _studentt_log_constant(nu_) - log(sigma_) -
           (nu_ + one(nu_)) * log1p((z * z) / nu_) / 2
end

function _backend_laplace_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    _primal(scale_) > zero(_primal(scale_)) || return oftype(xx, NaN)
    return -log(2 * scale_) - abs(xx - mu_) / scale_
end

function _backend_inversegamma_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    _primal(shape_) > zero(_primal(shape_)) || return oftype(xx, NaN)
    _primal(scale_) > zero(_primal(scale_)) || return oftype(xx, NaN)
    _primal(xx) > zero(_primal(xx)) || return _offsupport_neginf(xx)
    # exp-subnormal boundary (issues #345/#367): see _positive_subnormal
    _positive_subnormal(xx) && return _offsupport_neginf(xx)
    return shape_ * log(scale_) - loggamma(shape_) -
           (shape_ + one(shape_)) * log(xx) -
           scale_ / xx
end

function _backend_weibull_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    _primal(shape_) > zero(_primal(shape_)) || return oftype(xx, NaN)
    _primal(scale_) > zero(_primal(scale_)) || return oftype(xx, NaN)
    _primal(xx) < zero(_primal(xx)) && return _offsupport_neginf(xx)
    if _primal(xx) == zero(_primal(xx))
        if _primal(shape_) < one(_primal(shape_))
            return oftype(xx, Inf)
        elseif _primal(shape_) == one(_primal(shape_))
            return -log(scale_)
        end
        return _offsupport_neginf(xx)
    end
    # exp-subnormal boundary (issues #345/#367): see _positive_subnormal. Sits
    # AFTER the x == 0 branch, so the issue-#86 exact-zero shape channels stay.
    _positive_subnormal(xx) && return _offsupport_neginf(xx)
    log_ratio = log(xx) - log(scale_)
    return log(shape_) + (shape_ - one(shape_)) * log(xx) -
           shape_ * log(scale_) - exp(shape_ * log_ratio)
end

function _backend_beta_logpdf(alpha, beta_parameter, x)
    xx, alpha_, beta_ = promote(x, alpha, beta_parameter)
    _primal(alpha_) > zero(_primal(alpha_)) || return oftype(xx, NaN)
    _primal(beta_) > zero(_primal(beta_)) || return oftype(xx, NaN)
    zero(_primal(xx)) < _primal(xx) < one(_primal(xx)) || return _offsupport_neginf(xx)
    return loggamma(alpha_ + beta_) - loggamma(alpha_) - loggamma(beta_) +
           (alpha_ - one(alpha_)) * log(xx) +
           (beta_ - one(beta_)) * log1p(-xx)
end

function _backend_cauchy_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    _primal(sigma_) > zero(_primal(sigma_)) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return -log(oftype(xx, pi)) - log(sigma_) - log1p(z * z)
end

function _backend_halfnormal_logpdf(sigma, x)
    xx, sigma_ = promote(x, sigma)
    _primal(sigma_) > zero(_primal(sigma_)) || return oftype(xx, NaN)
    _primal(xx) >= zero(_primal(xx)) || return _offsupport_neginf(xx)
    z = xx / sigma_
    return log(oftype(xx, 2)) - log(sigma_) - log(2 * oftype(xx, pi)) / 2 - z * z / 2
end

function _backend_halfcauchy_logpdf(scale, x)
    xx, scale_ = promote(x, scale)
    _primal(scale_) > zero(_primal(scale_)) || return oftype(xx, NaN)
    _primal(xx) >= zero(_primal(xx)) || return _offsupport_neginf(xx)
    z = xx / scale_
    return log(oftype(xx, 2)) - log(oftype(xx, pi)) - log(scale_) - log1p(z * z)
end

function _backend_uniform_logpdf(lower, upper, x)
    xx, lower_, upper_ = promote(x, lower, upper)
    _primal(upper_) > _primal(lower_) || return oftype(xx, NaN)
    _primal(lower_) <= _primal(xx) <= _primal(upper_) || return _offsupport_neginf(xx)
    return -log(upper_ - lower_)
end

function _backend_logistic_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    _primal(scale_) > zero(_primal(scale_)) || return oftype(xx, NaN)
    z = (xx - mu_) / scale_
    az = abs(z)
    return -log(scale_) - az - 2 * log1p(exp(-az))
end

function _backend_gumbel_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    _primal(scale_) > zero(_primal(scale_)) || return oftype(xx, NaN)
    z = (xx - mu_) / scale_
    return -log(scale_) - z - exp(-z)
end

function _backend_pareto_logpdf(xm, alpha, x)
    xx, xm_, alpha_ = promote(x, xm, alpha)
    (_primal(xm_) > zero(_primal(xm_)) && _primal(alpha_) > zero(_primal(alpha_))) || return oftype(xx, NaN)
    _primal(xx) >= _primal(xm_) || return _offsupport_neginf(xx)
    return log(alpha_) + alpha_ * log(xm_) - (alpha_ + one(alpha_)) * log(xx)
end

function _backend_frechet_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    (_primal(shape_) > zero(_primal(shape_)) && _primal(scale_) > zero(_primal(scale_))) || return oftype(xx, NaN)
    _primal(xx) > zero(_primal(xx)) || return _offsupport_neginf(xx)
    # exp-subnormal boundary (issues #345/#367): see _positive_subnormal
    _positive_subnormal(xx) && return _offsupport_neginf(xx)
    logz = log(xx) - log(scale_)
    return log(shape_) - log(scale_) - (one(shape_) + shape_) * logz - exp(-shape_ * logz)
end

function _backend_rayleigh_logpdf(scale, x)
    xx, scale_ = promote(x, scale)
    _primal(scale_) > zero(_primal(scale_)) || return oftype(xx, NaN)
    _primal(xx) > zero(_primal(xx)) || return _offsupport_neginf(xx)
    # exp-subnormal boundary (issues #345/#367): see _positive_subnormal
    _positive_subnormal(xx) && return _offsupport_neginf(xx)
    return log(xx) - 2 * log(scale_) - xx * xx / (2 * scale_ * scale_)
end

function _backend_inversegaussian_logpdf(mu, lambda, x)
    xx, mu_, lambda_ = promote(x, mu, lambda)
    (_primal(mu_) > zero(_primal(mu_)) && _primal(lambda_) > zero(_primal(lambda_))) || return oftype(xx, NaN)
    _primal(xx) > zero(_primal(xx)) || return _offsupport_neginf(xx)
    # exp-subnormal boundary (issues #345/#367): see _positive_subnormal
    _positive_subnormal(xx) && return _offsupport_neginf(xx)
    d = xx - mu_
    return log(lambda_) / 2 - log(2 * oftype(xx, pi)) / 2 - 3 * log(xx) / 2 -
           lambda_ * d * d / (2 * mu_ * mu_ * xx)
end

# --- analytic partials (in-support; caller guards) ---------------------------

@inline function _normal_logpdf_partials(mu, sigma, value)
    z = (value - mu) / sigma
    inv_sigma = 1 / sigma
    dvalue = -z * inv_sigma
    dmu = z * inv_sigma
    dsigma = (z * z - 1) * inv_sigma
    return dvalue, dmu, dsigma
end

@inline function _lognormal_logpdf_partials(mu, sigma, value)
    log_value = log(value)
    z = (log_value - mu) / sigma
    inv_sigma = 1 / sigma
    dvalue = (-(z * inv_sigma) - 1) / value
    dmu = z * inv_sigma
    dsigma = (z * z - 1) * inv_sigma
    return dvalue, dmu, dsigma
end

@inline function _exponential_logpdf_partials(rate, value)
    dvalue = -rate
    drate = 1 / rate - value
    return dvalue, drate
end

@inline function _gamma_logpdf_partials(shape, rate, value)
    dvalue = (shape - 1) / value - rate
    dshape = log(rate) - digamma(shape) + log(value)
    drate = shape / rate - value
    return dvalue, dshape, drate
end

@inline function _inversegamma_logpdf_partials(shape, scale, value)
    dvalue = -(shape + 1) / value + scale / (value * value)
    dshape = log(scale) - digamma(shape) - log(value)
    dscale = shape / scale - 1 / value
    return dvalue, dshape, dscale
end

@inline function _weibull_logpdf_partials(shape, scale, value)
    log_ratio = log(value) - log(scale)
    ratio_power = exp(shape * log_ratio)
    dvalue = (shape - 1 - shape * ratio_power) / value
    dshape = 1 / shape + log_ratio - ratio_power * log_ratio
    dscale = shape * (ratio_power - 1) / scale
    return dvalue, dshape, dscale
end

@inline function _beta_logpdf_partials(alpha, beta_parameter, value)
    dvalue = (alpha - 1) / value - (beta_parameter - 1) / (1 - value)
    dalpha = digamma(alpha + beta_parameter) - digamma(alpha) + log(value)
    dbeta = digamma(alpha + beta_parameter) - digamma(beta_parameter) + log1p(-value)
    return dvalue, dalpha, dbeta
end

@inline function _studentt_logpdf_partials(nu, mu, sigma, value)
    z = (value - mu) / sigma
    denominator = nu + z * z
    dvalue = -((nu + 1) * z) / (sigma * denominator)
    dmu = -dvalue
    dsigma = nu * (z * z - one(value)) / (sigma * denominator)
    # the digamma-difference part goes through the Float64-widened helper
    # (issue #53): at Float32 the ~1/nu difference of ~log(nu)-sized digammas
    # would disagree with the widened value being differentiated
    dnu =
        _studentt_log_constant_dnu(nu) +
        oftype(value, 0.5) * (-log1p((z * z) / nu) + ((nu + 1) * z * z) / (nu * denominator))
    return dvalue, dnu, dmu, dsigma
end

@inline function _laplace_logpdf_partials(mu, scale, value)
    delta = value - mu
    sign_delta = delta > 0 ? one(delta) : (delta < 0 ? -one(delta) : zero(delta))
    dvalue = -sign_delta / scale
    dmu = sign_delta / scale
    dscale = -1 / scale + abs(delta) / (scale * scale)
    return dvalue, dmu, dscale
end

@inline function _cauchy_logpdf_partials(mu, sigma, value)
    z = (value - mu) / sigma
    denominator = 1 + z * z
    dvalue = -2 * z / (sigma * denominator)
    dmu = -dvalue
    dsigma = (z * z - 1) / (sigma * denominator)
    return dvalue, dmu, dsigma
end

@inline function _halfnormal_logpdf_partials(sigma, value)
    z = value / sigma
    dvalue = -z / sigma
    dsigma = (z * z - 1) / sigma
    return dvalue, dsigma
end

@inline function _halfcauchy_logpdf_partials(scale, value)
    z = value / scale
    denominator = 1 + z * z
    dvalue = -2 * z / (scale * denominator)
    dscale = (z * z - 1) / (scale * denominator)
    return dvalue, dscale
end

# d/dvalue is 0 on the open interval, and the accumulate loop skips the value
# channel entirely, so only the bound partials are returned (relevant only for
# dynamic-bound observations).
@inline function _uniform_logpdf_partials(lower, upper, value)
    inv_width = 1 / (upper - lower)
    dlower = inv_width
    dupper = -inv_width
    return dlower, dupper
end

@inline function _logistic_logpdf_partials(mu, scale, value)
    z = (value - mu) / scale
    s = 1 / (1 + exp(-z)) # sigmoid(z)
    dvalue = (1 - 2 * s) / scale
    dmu = -dvalue
    dscale = (-1 - z * (1 - 2 * s)) / scale
    return dvalue, dmu, dscale
end

@inline function _gumbel_logpdf_partials(mu, scale, value)
    z = (value - mu) / scale
    e = exp(-z)
    dvalue = (e - 1) / scale
    dmu = -dvalue
    dscale = (-1 - z * (e - 1)) / scale
    return dvalue, dmu, dscale
end

@inline function _pareto_logpdf_partials(xm, alpha, value)
    dvalue = -(alpha + 1) / value
    dxm = alpha / xm
    dalpha = 1 / alpha + log(xm) - log(value)
    return dvalue, dxm, dalpha
end

@inline function _frechet_logpdf_partials(shape, scale, value)
    logz = log(value) - log(scale)
    w = exp(-shape * logz)
    dvalue = (-(1 + shape) + shape * w) / value
    dshape = 1 / shape - logz * (1 - w)
    dscale = (shape / scale) * (1 - w)
    return dvalue, dshape, dscale
end

@inline function _rayleigh_logpdf_partials(scale, value)
    dvalue = 1 / value - value / (scale * scale)
    dscale = -2 / scale + value * value / (scale * scale * scale)
    return dvalue, dscale
end

@inline function _inversegaussian_logpdf_partials(mu, lambda, value)
    d = value - mu
    dvalue = -3 / (2 * value) - lambda / (2 * mu * mu) + lambda / (2 * value * value)
    dmu = lambda * d / (mu * mu * mu)
    dlambda = 1 / (2 * lambda) - d * d / (2 * mu * mu * value)
    return dvalue, dmu, dlambda
end

# --- discrete logpdf kernels (issue #285, stage 2) ---------------------------
# Same single-source contract as the continuous kernels above. Unlike those,
# invalid PARAMETERS throw ArgumentError (matching the historical backend
# semantics for discrete families); off-support VALUES return -Inf. The count/
# support classification helpers (_bernoulli_value, _poisson_count,
# _binomial_trials, _categorical_index, _discrete_integer) and the
# precision-careful data-term helpers (_logfactorial_like, _logbinomial_like,
# _betabinomial_logpdf_core, and the issue-#345 extreme-count saddle-point core
# _stirlerr/_bd0/_*_logpdf_saddle) live in distributions/discrete.jl.

function _backend_bernoulli_logpdf(p, x)
    probability = p
    zero(_primal(probability)) <= _primal(probability) <= one(_primal(probability)) ||
        throw(ArgumentError("bernoulli requires 0 <= p <= 1"))
    value = _bernoulli_value(x)
    isnothing(value) && return oftype(float(probability), -Inf)
    return value ? log(probability) : log1p(-probability)
end

# Logit-parameterized Bernoulli (issues #149/#150), stable log-scale form
# `x*eta - log1p(exp(eta))`.
function _backend_bernoullilogit_logpdf(eta, x)
    value = _bernoulli_value(x)
    isnothing(value) && return oftype(float(eta), -Inf)
    log_normalizer = _bernoullilogit_log1p_exp(eta)
    return value ? eta - log_normalizer : -log_normalizer
end

function _backend_poisson_logpdf(lambda, x)
    _primal(lambda) > zero(_primal(lambda)) || throw(ArgumentError("poisson requires lambda > 0"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(lambda), -Inf)
    # extreme counts (issue #345): the naive spelling differences
    # ~count*log(lambda)-sized terms whose rounding swamps (and can flip the
    # sign of) the O(1) result; the saddle-point form never does
    count >= _COUNT_SADDLE_THRESHOLD && return _poisson_logpdf_saddle(lambda, count)
    return count * log(lambda) - lambda - _logfactorial_like(lambda, count)
end

function _backend_geometric_logpdf(probability, x)
    probability_ = float(probability)
    zero(_primal(probability_)) < _primal(probability_) <= one(_primal(probability_)) ||
        throw(ArgumentError("geometric requires 0 < p <= 1"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    if count == 0
        return log(probability_)
    elseif _primal(probability_) == one(_primal(probability_))
        return oftype(probability_, -Inf)
    end
    return log(probability_) + count * log1p(-probability_)
end

function _backend_negativebinomial_logpdf(successes, probability, x)
    successes_, probability_ = promote(successes, probability)
    _primal(successes_) > zero(_primal(successes_)) ||
        throw(ArgumentError("negativebinomial requires successes > 0"))
    zero(_primal(probability_)) < _primal(probability_) <= one(_primal(probability_)) ||
        throw(ArgumentError("negativebinomial requires 0 < p <= 1"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    if count == 0 && _primal(probability_) == one(_primal(probability_))
        return zero(probability_)
    elseif _primal(probability_) == one(_primal(probability_))
        return oftype(probability_, -Inf)
    end
    # extreme counts (issue #345): loggamma(count + r) - log(count!) and the
    # count * log1p(-p) product all carry ~eps * count * log(count) rounding;
    # the saddle-point form stays O(1)-accurate
    count >= _COUNT_SADDLE_THRESHOLD &&
        return _negativebinomial_logpdf_saddle(successes_, probability_, count)
    return loggamma(count + successes_) - loggamma(successes_) - _logfactorial_like(probability_, count) +
           successes_ * log(probability_) + count * log1p(-probability_)
end

function _backend_binomial_logpdf(trials, probability, x)
    probability_ = float(probability)
    zero(_primal(probability_)) <= _primal(probability_) <= one(_primal(probability_)) ||
        throw(ArgumentError("binomial requires 0 <= p <= 1"))
    trial_count = _binomial_trials(trials)
    isnothing(trial_count) && throw(ArgumentError("binomial requires integer trials >= 0"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(probability_, -Inf)
    count <= trial_count || return oftype(probability_, -Inf)
    # extreme trial counts (issue #345): log C(n, k) and k log(p) are
    # ~n log(n)-sized and nearly cancel; only the fused saddle-point form
    # (which pairs them inside bd0 deviances) keeps the O(1) result -- the
    # naive spelling scored binomial(1e16, 0.5) at n/2 as +35 (true -18.65)
    trial_count >= _COUNT_SADDLE_THRESHOLD &&
        return _binomial_logpdf_saddle(trial_count, probability_, count)
    log_combination = _logbinomial_like(probability_, trial_count, count)
    if count == 0 && count == trial_count
        return log_combination
    elseif count == 0
        return log_combination + trial_count * log1p(-probability_)
    elseif count == trial_count
        return log_combination + count * log(probability_)
    end
    return log_combination +
           count * log(probability_) +
           (trial_count - count) * log1p(-probability_)
end

# Beta-Binomial (issue #231): shares `_betabinomial_logpdf_core` with the CPU
# reference; `alpha`/`beta` may be latent-flowing.
function _backend_betabinomial_logpdf(trials, alpha, beta, x)
    alpha_, beta_ = promote(float(alpha), float(beta))
    _primal(alpha_) > zero(_primal(alpha_)) || throw(ArgumentError("betabinomial requires alpha > 0"))
    _primal(beta_) > zero(_primal(beta_)) || throw(ArgumentError("betabinomial requires beta > 0"))
    trial_count = _binomial_trials(trials)
    isnothing(trial_count) && throw(ArgumentError("betabinomial requires integer trials >= 0"))
    count = _poisson_count(x)
    isnothing(count) && return oftype(alpha_, -Inf)
    count <= trial_count || return oftype(alpha_, -Inf)
    return _betabinomial_logpdf_core(trial_count, alpha_, beta_, count)
end

# Discrete-uniform (issue #231): trivial `-log(b - a + 1)` density with no
# continuous parameters.
function _backend_discreteuniform_logpdf(lower, upper, x)
    lo = _discrete_integer(lower)
    hi = _discrete_integer(upper)
    (isnothing(lo) || isnothing(hi)) && throw(ArgumentError("discreteuniform requires integer bounds"))
    lo <= hi || throw(ArgumentError("discreteuniform requires a <= b"))
    value = _discrete_integer(x)
    (isnothing(value) || value < lo || value > hi) && return -Inf
    return -log(float(hi - lo + 1))
end

function _backend_categorical_logpdf(probabilities::Tuple, x)
    length(probabilities) > 0 || throw(ArgumentError("categorical requires at least one probability"))
    total = zero(float(first(probabilities)))
    for probability in probabilities
        probability_ = float(probability)
        zero(_primal(probability_)) <= _primal(probability_) <= one(_primal(probability_)) ||
            throw(ArgumentError("categorical requires 0 <= p <= 1"))
        total += probability_
    end
    tolerance = sqrt(eps(total)) * max(length(probabilities), 1) * 8
    abs(total - one(total)) <= tolerance || throw(ArgumentError("categorical probabilities must sum to 1"))
    index = _categorical_index(x, length(probabilities))
    isnothing(index) && return oftype(total, -Inf)
    return log(float(probabilities[index]))
end

# --- discrete analytic partials (classified inputs; caller classifies) -------
# The caller runs the support/count classification (_bernoulli_value /
# _poisson_count) and skips off-support values exactly as before; these kernels
# take the already-classified inputs.

@inline _bernoulli_logpdf_partials(probability, value) =
    value != 0 ? 1 / probability : -1 / (1 - probability)

# `d/d_eta = x - logistic(eta)`: well-conditioned where the sigmoid-spelling
# gradient (1/p or -1/(1-p)) blows up as p saturates.
@inline _bernoullilogit_logpdf_partials(eta, support::Bool) =
    (support ? one(eta) : zero(eta)) - _bernoullilogit_logistic(eta)

@inline _poisson_logpdf_partials(lambda, count) = count / lambda - 1

@inline _categorical_logpdf_partials(probability) = 1 / probability

@inline function _binomial_logpdf_partials(trials, probability, count)
    if count == 0
        return -trials / (1 - probability)
    elseif count == trials
        return count / probability
    end
    return count / probability - (trials - count) / (1 - probability)
end

# The count == 0 contribution of the -count / (1 - p) term is exactly zero;
# skipping it keeps the gradient finite at p == 1 (issue #77).
@inline function _geometric_logpdf_partials(probability, count)
    derivative = 1 / probability
    if count > 0
        derivative -= count / (1 - probability)
    end
    return derivative
end

@inline function _negativebinomial_logpdf_partials(successes, probability, count)
    dsuccesses = digamma(count + successes) - digamma(successes) + log(probability)
    # as in the geometric case, skip the exactly-zero count == 0 term so p == 1
    # stays finite (issue #77)
    dprobability = successes / probability
    if count > 0
        dprobability -= count / (1 - probability)
    end
    return dsuccesses, dprobability
end

# Beta-Binomial (issue #231): closed-form digamma differences. With s = alpha + beta,
#   d/dalpha = psi(k+a) - psi(n+s) - psi(a) + psi(s),
#   d/dbeta  = psi(n-k+b) - psi(n+s) - psi(b) + psi(s).
@inline function _betabinomial_logpdf_partials(trials, alpha, beta, count)
    s = alpha + beta
    dgamma_ns = digamma(trials + s)
    dgamma_s = digamma(s)
    dalpha = digamma(count + alpha) - dgamma_ns - digamma(alpha) + dgamma_s
    dbeta = digamma(trials - count + beta) - dgamma_ns - digamma(beta) + dgamma_s
    return dalpha, dbeta
end
