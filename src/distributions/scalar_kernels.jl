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

# --- logpdf kernels ----------------------------------------------------------

function _backend_normal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return -log(sigma_) - log(2 * pi) / 2 - z * z / 2
end

function _backend_lognormal_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return _backend_normal_logpdf(mu_, sigma_, log(xx)) - log(xx)
end

function _backend_exponential_logpdf(rate, x)
    xx, rate_ = promote(x, rate)
    rate_ > zero(rate_) || return oftype(xx, NaN)
    xx >= zero(xx) || return oftype(xx, -Inf)
    return log(rate_) - rate_ * xx
end

function _backend_gamma_logpdf(shape, rate, x)
    xx, shape_, rate_ = promote(x, shape, rate)
    shape_ > zero(shape_) || return oftype(xx, NaN)
    rate_ > zero(rate_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return shape_ * log(rate_) - loggamma(shape_) + (shape_ - one(shape_)) * log(xx) - rate_ * xx
end

function _backend_studentt_logpdf(nu, mu, sigma, x)
    xx, nu_, mu_, sigma_ = promote(x, nu, mu, sigma)
    nu_ > zero(nu_) || return oftype(xx, NaN)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return _studentt_log_constant(nu_) - log(sigma_) -
           (nu_ + one(nu_)) * log1p((z * z) / nu_) / 2
end

function _backend_laplace_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    return -log(2 * scale_) - abs(xx - mu_) / scale_
end

function _backend_inversegamma_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    shape_ > zero(shape_) || return oftype(xx, NaN)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return shape_ * log(scale_) - loggamma(shape_) -
           (shape_ + one(shape_)) * log(xx) -
           scale_ / xx
end

function _backend_weibull_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    shape_ > zero(shape_) || return oftype(xx, NaN)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx < zero(xx) && return oftype(xx, -Inf)
    if xx == zero(xx)
        if shape_ < one(shape_)
            return oftype(xx, Inf)
        elseif shape_ == one(shape_)
            return -log(scale_)
        end
        return oftype(xx, -Inf)
    end
    log_ratio = log(xx) - log(scale_)
    return log(shape_) + (shape_ - one(shape_)) * log(xx) -
           shape_ * log(scale_) - exp(shape_ * log_ratio)
end

function _backend_beta_logpdf(alpha, beta_parameter, x)
    xx, alpha_, beta_ = promote(x, alpha, beta_parameter)
    alpha_ > zero(alpha_) || return oftype(xx, NaN)
    beta_ > zero(beta_) || return oftype(xx, NaN)
    zero(xx) < xx < one(xx) || return oftype(xx, -Inf)
    return loggamma(alpha_ + beta_) - loggamma(alpha_) - loggamma(beta_) +
           (alpha_ - one(alpha_)) * log(xx) +
           (beta_ - one(beta_)) * log1p(-xx)
end

function _backend_cauchy_logpdf(mu, sigma, x)
    xx, mu_, sigma_ = promote(x, mu, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    z = (xx - mu_) / sigma_
    return -log(oftype(xx, pi)) - log(sigma_) - log1p(z * z)
end

function _backend_halfnormal_logpdf(sigma, x)
    xx, sigma_ = promote(x, sigma)
    sigma_ > zero(sigma_) || return oftype(xx, NaN)
    xx >= zero(xx) || return oftype(xx, -Inf)
    z = xx / sigma_
    return log(oftype(xx, 2)) - log(sigma_) - log(2 * oftype(xx, pi)) / 2 - z * z / 2
end

function _backend_halfcauchy_logpdf(scale, x)
    xx, scale_ = promote(x, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx >= zero(xx) || return oftype(xx, -Inf)
    z = xx / scale_
    return log(oftype(xx, 2)) - log(oftype(xx, pi)) - log(scale_) - log1p(z * z)
end

function _backend_uniform_logpdf(lower, upper, x)
    xx, lower_, upper_ = promote(x, lower, upper)
    upper_ > lower_ || return oftype(xx, NaN)
    lower_ <= xx <= upper_ || return oftype(xx, -Inf)
    return -log(upper_ - lower_)
end

function _backend_logistic_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    z = (xx - mu_) / scale_
    az = abs(z)
    return -log(scale_) - az - 2 * log1p(exp(-az))
end

function _backend_gumbel_logpdf(mu, scale, x)
    xx, mu_, scale_ = promote(x, mu, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    z = (xx - mu_) / scale_
    return -log(scale_) - z - exp(-z)
end

function _backend_pareto_logpdf(xm, alpha, x)
    xx, xm_, alpha_ = promote(x, xm, alpha)
    (xm_ > zero(xm_) && alpha_ > zero(alpha_)) || return oftype(xx, NaN)
    xx >= xm_ || return oftype(xx, -Inf)
    return log(alpha_) + alpha_ * log(xm_) - (alpha_ + one(alpha_)) * log(xx)
end

function _backend_frechet_logpdf(shape, scale, x)
    xx, shape_, scale_ = promote(x, shape, scale)
    (shape_ > zero(shape_) && scale_ > zero(scale_)) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    logz = log(xx) - log(scale_)
    return log(shape_) - log(scale_) - (one(shape_) + shape_) * logz - exp(-shape_ * logz)
end

function _backend_rayleigh_logpdf(scale, x)
    xx, scale_ = promote(x, scale)
    scale_ > zero(scale_) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
    return log(xx) - 2 * log(scale_) - xx * xx / (2 * scale_ * scale_)
end

function _backend_inversegaussian_logpdf(mu, lambda, x)
    xx, mu_, lambda_ = promote(x, mu, lambda)
    (mu_ > zero(mu_) && lambda_ > zero(lambda_)) || return oftype(xx, NaN)
    xx > zero(xx) || return oftype(xx, -Inf)
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
