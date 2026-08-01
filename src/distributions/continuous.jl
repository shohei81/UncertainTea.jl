# CPU-reference distributions (structs, builders, logpdf, rand): continuous scalar families (normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, studentt).

struct NormalDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function NormalDist(mu::T, sigma::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("normal requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

struct LaplaceDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    scale::T

    function LaplaceDist(mu::T, scale::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("laplace requires scale > 0"))
        new{T}(mu, scale)
    end
end

struct ExponentialDist{T<:Real} <: AbstractTeaDistribution
    rate::T

    function ExponentialDist(rate::T) where {T<:Real}
        rate > zero(T) || throw(ArgumentError("exponential requires rate > 0"))
        new{T}(rate)
    end
end

struct GammaDist{T<:Real} <: AbstractTeaDistribution
    shape::T
    rate::T

    function GammaDist(shape::T, rate::T) where {T<:Real}
        shape > zero(T) || throw(ArgumentError("gamma requires shape > 0"))
        rate > zero(T) || throw(ArgumentError("gamma requires rate > 0"))
        new{T}(shape, rate)
    end
end

struct InverseGammaDist{T<:Real} <: AbstractTeaDistribution
    shape::T
    scale::T

    function InverseGammaDist(shape::T, scale::T) where {T<:Real}
        shape > zero(T) || throw(ArgumentError("inversegamma requires shape > 0"))
        scale > zero(T) || throw(ArgumentError("inversegamma requires scale > 0"))
        new{T}(shape, scale)
    end
end

struct WeibullDist{T<:Real} <: AbstractTeaDistribution
    shape::T
    scale::T

    function WeibullDist(shape::T, scale::T) where {T<:Real}
        shape > zero(T) || throw(ArgumentError("weibull requires shape > 0"))
        scale > zero(T) || throw(ArgumentError("weibull requires scale > 0"))
        new{T}(shape, scale)
    end
end

struct BetaDist{T<:Real} <: AbstractTeaDistribution
    alpha::T
    beta::T

    function BetaDist(alpha::T, beta::T) where {T<:Real}
        alpha > zero(T) || throw(ArgumentError("beta requires alpha > 0"))
        beta > zero(T) || throw(ArgumentError("beta requires beta > 0"))
        new{T}(alpha, beta)
    end
end

struct LogNormalDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function LogNormalDist(mu::T, sigma::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("lognormal requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

struct StudentTDist{T<:Real} <: AbstractTeaDistribution
    nu::T
    mu::T
    sigma::T

    function StudentTDist(nu::T, mu::T, sigma::T) where {T<:Real}
        nu > zero(T) || throw(ArgumentError("studentt requires nu > 0"))
        sigma > zero(T) || throw(ArgumentError("studentt requires sigma > 0"))
        new{T}(nu, mu, sigma)
    end
end

struct CauchyDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T

    function CauchyDist(mu::T, sigma::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("cauchy requires sigma > 0"))
        new{T}(mu, sigma)
    end
end

struct HalfNormalDist{T<:Real} <: AbstractTeaDistribution
    sigma::T

    function HalfNormalDist(sigma::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("halfnormal requires sigma > 0"))
        new{T}(sigma)
    end
end

struct HalfCauchyDist{T<:Real} <: AbstractTeaDistribution
    scale::T

    function HalfCauchyDist(scale::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("halfcauchy requires scale > 0"))
        new{T}(scale)
    end
end

struct UniformDist{T<:Real} <: AbstractTeaDistribution
    lower::T
    upper::T

    function UniformDist(lower::T, upper::T) where {T<:Real}
        upper > lower || throw(ArgumentError("uniform requires upper > lower"))
        new{T}(lower, upper)
    end
end

struct LogisticDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    scale::T

    function LogisticDist(mu::T, scale::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("logistic requires scale > 0"))
        new{T}(mu, scale)
    end
end

struct GumbelDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    scale::T

    function GumbelDist(mu::T, scale::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("gumbel requires scale > 0"))
        new{T}(mu, scale)
    end
end

# Positive-support / heavy-tail families (issue #230).
struct ParetoDist{T<:Real} <: AbstractTeaDistribution
    xm::T
    alpha::T

    function ParetoDist(xm::T, alpha::T) where {T<:Real}
        xm > zero(T) || throw(ArgumentError("pareto requires x_m > 0"))
        alpha > zero(T) || throw(ArgumentError("pareto requires alpha > 0"))
        new{T}(xm, alpha)
    end
end

struct FrechetDist{T<:Real} <: AbstractTeaDistribution
    shape::T
    scale::T

    function FrechetDist(shape::T, scale::T) where {T<:Real}
        shape > zero(T) || throw(ArgumentError("frechet requires shape > 0"))
        scale > zero(T) || throw(ArgumentError("frechet requires scale > 0"))
        new{T}(shape, scale)
    end
end

struct RayleighDist{T<:Real} <: AbstractTeaDistribution
    scale::T

    function RayleighDist(scale::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("rayleigh requires scale > 0"))
        new{T}(scale)
    end
end

struct InverseGaussianDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    lambda::T

    function InverseGaussianDist(mu::T, lambda::T) where {T<:Real}
        mu > zero(T) || throw(ArgumentError("inversegaussian requires mu > 0"))
        lambda > zero(T) || throw(ArgumentError("inversegaussian requires lambda > 0"))
        new{T}(mu, lambda)
    end
end

# Builders normalize parameters through `float` so integer (or other non-float
# real) literals reach the samplers as float storage (issue #73); `float` keeps
# ForwardDiff Duals intact.
function normal(mu, sigma)
    promoted_mu, promoted_sigma = promote(float(mu), float(sigma))
    return NormalDist(promoted_mu, promoted_sigma)
end

function laplace(mu, scale)
    promoted_mu, promoted_scale = promote(float(mu), float(scale))
    return LaplaceDist(promoted_mu, promoted_scale)
end

function exponential(rate)
    return ExponentialDist(float(rate))
end

function gamma(shape, rate)
    promoted_shape, promoted_rate = promote(float(shape), float(rate))
    return GammaDist(promoted_shape, promoted_rate)
end

function inversegamma(shape, scale)
    promoted_shape, promoted_scale = promote(float(shape), float(scale))
    return InverseGammaDist(promoted_shape, promoted_scale)
end

function weibull(shape, scale)
    promoted_shape, promoted_scale = promote(float(shape), float(scale))
    return WeibullDist(promoted_shape, promoted_scale)
end

function beta(alpha, beta_parameter)
    promoted_alpha, promoted_beta = promote(float(alpha), float(beta_parameter))
    return BetaDist(promoted_alpha, promoted_beta)
end

function lognormal(mu, sigma)
    promoted_mu, promoted_sigma = promote(float(mu), float(sigma))
    return LogNormalDist(promoted_mu, promoted_sigma)
end

function studentt(nu, mu, sigma)
    promoted_nu, promoted_mu, promoted_sigma = promote(float(nu), float(mu), float(sigma))
    return StudentTDist(promoted_nu, promoted_mu, promoted_sigma)
end

function cauchy(mu, sigma)
    promoted_mu, promoted_sigma = promote(float(mu), float(sigma))
    return CauchyDist(promoted_mu, promoted_sigma)
end

function halfnormal(sigma)
    return HalfNormalDist(float(sigma))
end

function halfcauchy(scale)
    return HalfCauchyDist(float(scale))
end

function uniform(lower, upper)
    promoted_lower, promoted_upper = promote(float(lower), float(upper))
    return UniformDist(promoted_lower, promoted_upper)
end

function logistic(mu, scale)
    promoted_mu, promoted_scale = promote(float(mu), float(scale))
    return LogisticDist(promoted_mu, promoted_scale)
end

function gumbel(mu, scale)
    promoted_mu, promoted_scale = promote(float(mu), float(scale))
    return GumbelDist(promoted_mu, promoted_scale)
end

function pareto(xm, alpha)
    promoted_xm, promoted_alpha = promote(float(xm), float(alpha))
    return ParetoDist(promoted_xm, promoted_alpha)
end

function frechet(shape, scale)
    promoted_shape, promoted_scale = promote(float(shape), float(scale))
    return FrechetDist(promoted_shape, promoted_scale)
end

function rayleigh(scale)
    return RayleighDist(float(scale))
end

function inversegaussian(mu, lambda)
    promoted_mu, promoted_lambda = promote(float(mu), float(lambda))
    return InverseGaussianDist(promoted_mu, promoted_lambda)
end

function Random.rand(rng::AbstractRNG, dist::NormalDist{T}) where {T<:AbstractFloat}
    return dist.mu + dist.sigma * randn(rng, T)
end

function Random.rand(rng::AbstractRNG, dist::LaplaceDist)
    scale = float(dist.scale)
    threshold = rand(rng, typeof(scale)) - oftype(scale, 0.5)
    noise = threshold < 0 ? log1p(2 * threshold) : -log1p(-2 * threshold)
    return float(dist.mu) + scale * noise
end

function Random.rand(rng::AbstractRNG, dist::ExponentialDist)
    rate = float(dist.rate)
    return randexp(rng, typeof(rate)) / rate
end

function _rand_gamma_marsaglia(rng::AbstractRNG, shape::T, rate::T) where {T<:AbstractFloat}
    if shape < one(T)
        # Marsaglia-Tsang boost gamma(shape+1) * u^(1/shape), evaluated in log
        # space: the direct power underflows to exactly 0.0 for small shape
        # (u^100 at shape=0.01), and exact-zero draws crash downstream
        # simplex/log transforms at HMC init (issue #101). Flooring at
        # floatmin keeps the draw strictly positive.
        boosted = log(_rand_gamma_marsaglia(rng, shape + one(T), rate)) + log(rand(rng, T)) / shape
        return max(exp(boosted), floatmin(T))
    end

    d = shape - T(1 / 3)
    c = inv(sqrt(T(9) * d))
    while true
        x = randn(rng, T)
        v = one(T) + c * x
        v > zero(T) || continue
        v3 = v * v * v
        u = rand(rng, T)
        if u < one(T) - T(0.0331) * (x * x) * (x * x) ||
           log(u) < T(0.5) * x * x + d * (one(T) - v3 + log(v3))
            return d * v3 / rate
        end
    end
end

function Random.rand(rng::AbstractRNG, dist::GammaDist)
    shape = float(dist.shape)
    rate = float(dist.rate)
    return _rand_gamma_marsaglia(rng, shape, rate)
end

function Random.rand(rng::AbstractRNG, dist::InverseGammaDist)
    shape = float(dist.shape)
    scale = float(dist.scale)
    return inv(_rand_gamma_marsaglia(rng, shape, scale))
end

function Random.rand(rng::AbstractRNG, dist::WeibullDist)
    shape = float(dist.shape)
    scale = float(dist.scale)
    return scale * (randexp(rng, typeof(scale)) ^ inv(shape))
end

function Random.rand(rng::AbstractRNG, dist::BetaDist)
    alpha = float(dist.alpha)
    beta_parameter = float(dist.beta)
    x = _rand_gamma_marsaglia(rng, alpha, one(alpha))
    y = _rand_gamma_marsaglia(rng, beta_parameter, one(beta_parameter))
    return x / (x + y)
end

function Random.rand(rng::AbstractRNG, dist::LogNormalDist{T}) where {T<:AbstractFloat}
    return exp(dist.mu + dist.sigma * randn(rng, T))
end

function Random.rand(rng::AbstractRNG, dist::StudentTDist)
    nu = float(dist.nu)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    scale = randn(rng, typeof(mu)) / sqrt(_rand_gamma_marsaglia(rng, nu / 2, nu / 2))
    return mu + sigma * scale
end

function Random.rand(rng::AbstractRNG, dist::CauchyDist)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    u = rand(rng, typeof(mu))
    return mu + sigma * tan(oftype(mu, pi) * (u - oftype(mu, 0.5)))
end

function Random.rand(rng::AbstractRNG, dist::HalfNormalDist)
    sigma = float(dist.sigma)
    return abs(sigma * randn(rng, typeof(sigma)))
end

function Random.rand(rng::AbstractRNG, dist::HalfCauchyDist)
    scale = float(dist.scale)
    u = rand(rng, typeof(scale))
    return abs(scale * tan(oftype(scale, pi) * (u - oftype(scale, 0.5))))
end

function Random.rand(rng::AbstractRNG, dist::UniformDist)
    lower = float(dist.lower)
    upper = float(dist.upper)
    return lower + (upper - lower) * rand(rng, typeof(lower))
end

function Random.rand(rng::AbstractRNG, dist::LogisticDist)
    mu = float(dist.mu)
    scale = float(dist.scale)
    u = rand(rng, typeof(mu))
    return mu + scale * (log(u) - log1p(-u))
end

function Random.rand(rng::AbstractRNG, dist::GumbelDist)
    mu = float(dist.mu)
    scale = float(dist.scale)
    u = rand(rng, typeof(mu))
    return mu - scale * log(-log(u))
end

function Random.rand(rng::AbstractRNG, dist::ParetoDist)
    xm = float(dist.xm)
    alpha = float(dist.alpha)
    u = rand(rng, typeof(xm))
    # Inverse CDF: F(x) = 1 - (xm/x)^alpha, so x = xm * u^(-1/alpha).
    return xm * u^(-inv(alpha))
end

function Random.rand(rng::AbstractRNG, dist::FrechetDist)
    shape = float(dist.shape)
    scale = float(dist.scale)
    u = rand(rng, typeof(scale))
    # Inverse CDF: F(x) = exp(-(x/s)^(-alpha)), so x = s * (-log u)^(-1/alpha).
    return scale * (-log(u))^(-inv(shape))
end

function Random.rand(rng::AbstractRNG, dist::RayleighDist)
    scale = float(dist.scale)
    u = rand(rng, typeof(scale))
    # Inverse CDF: F(x) = 1 - exp(-x^2/(2 s^2)), so x = s * sqrt(-2 log u).
    return scale * sqrt(-2 * log(u))
end

function Random.rand(rng::AbstractRNG, dist::InverseGaussianDist)
    mu = float(dist.mu)
    lambda = float(dist.lambda)
    # Michael-Schucany-Haas transform: one randn + one uniform.
    nu = randn(rng, typeof(mu))
    y = nu * nu
    x =
        mu + (mu * mu * y) / (2 * lambda) -
        (mu / (2 * lambda)) * sqrt(4 * mu * lambda * y + mu * mu * y * y)
    z = rand(rng, typeof(mu))
    return z <= mu / (mu + x) ? x : mu * mu / x
end

function _std_t_cdf(z, nu)
    zz = float(z)
    # Guard infinite arguments before arithmetic so ForwardDiff Duals carrying an
    # Inf value (with NaN partials) collapse to a finite constant.
    isinf(zz) && return zz > zero(zz) ? one(zz) : zero(zz)
    zz == zero(zz) && return oftype(zz, 0.5)
    x = nu / (nu + zz * zz)
    regularized = beta_inc(nu / 2, one(nu) / 2, x)[1]
    return zz > zero(zz) ? one(zz) - regularized / 2 : regularized / 2
end

# Lentz continued fraction for the regularized incomplete beta (the Numerical
# Recipes form, mirroring `_device_betacf`); called with plain reals only --
# the ForwardDiff overloads of `_std_t_cdf`/`_std_t_log_cdf` differentiate via
# the analytic d/dz = t-pdf(z) instead.
function _beta_inc_cf(a::T, b::T, x::T) where {T<:AbstractFloat}
    qab = a + b
    qap = a + one(T)
    qam = a - one(T)
    fpmin = T(1e-30)
    c = one(T)
    d = one(T) - qab * x / qap
    abs(d) < fpmin && (d = fpmin)
    d = one(T) / d
    h = d
    for m = 1:200
        mf = T(m)
        m2 = 2 * mf
        aa = mf * (b - mf) * x / ((qam + m2) * (a + m2))
        d = one(T) + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = one(T) + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = one(T) / d
        h *= d * c
        aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
        d = one(T) + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = one(T) + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = one(T) / d
        delta = d * c
        h *= delta
        abs(delta - one(T)) <= T(4) * eps(T) && break
    end
    return h
end

# log I_x(a, b) computed IN LOG SPACE on the direct branch (x small), so a tail
# probability that underflows as a plain value (light Student-t tails at large
# nu -- issue #97) still yields a finite log. The symmetric branch is the
# complement of a small quantity and is safe as log1p. `x` and `y` are the
# caller-computed pair with x + y == 1 in exact arithmetic, keeping each
# accurate near its own zero. CPU port of the device
# `_device_log_beta_inc_reg_parts`.
function _log_beta_inc_reg(a::T, b::T, x::T, y::T) where {T<:AbstractFloat}
    x <= zero(T) && return T(-Inf)
    y <= zero(T) && return zero(T)
    logx = x < T(0.5) ? log(x) : log1p(-y)
    logy = y < T(0.5) ? log(y) : log1p(-x)
    log_front = a * logx + b * logy + loggamma(a + b) - loggamma(a) - loggamma(b)
    if x < (a + one(T)) / (a + b + 2)
        return log_front + log(_beta_inc_cf(a, b, x)) - log(a)
    end
    return log1p(-exp(log_front) * _beta_inc_cf(b, a, y) / b)
end

# log(T_cdf(z; nu)) computed WITHOUT the `1 - regularized/2` cancellation and
# with the incomplete beta itself formed in log space (issue #97): the
# regularized incomplete beta is small in the lower tail, so its LOG stays
# finite past the point where the plain value underflows to 0 (light tails at
# large nu, z beyond ~39 at Float64); the upper side goes through log1p. This
# is the CPU port of the device `_device_std_t_log_cdf` (issues #43/#97).
function _std_t_log_cdf(z, nu)
    zz = float(z)
    isinf(zz) && return zz > zero(zz) ? zero(zz) : oftype(zz, -Inf)
    zz == zero(zz) && return -log(oftype(zz, 2))
    # compute in at least Float64: a Float32 tail loses the incomplete beta's
    # exponent range long before its log leaves the Float32 range, so widen,
    # take the log, and narrow the result
    W = promote_type(typeof(zz), Float64)
    zw = W(zz)
    nuw = W(nu)
    denominator = nuw + zw * zw
    x = nuw / denominator
    y = zw * zw / denominator # accurate complement of x
    log_regularized = _log_beta_inc_reg(nuw / 2, W(0.5), x, y)
    zw < zero(zw) && return oftype(zz, log_regularized - log(W(2)))
    return oftype(zz, log1p(-exp(log_regularized) / 2))
end

# log(T_cdf(zb) - T_cdf(za)): one-sided normalizers use the symmetry
# S(z) = cdf(-z) so a tail probability is computed directly rather than as
# 1 - cdf (which cancels at Float64 for light tails, e.g. nu = 1e5 with a 15
# sigma cutoff -- issue #43); finite intervals difference LOG CDFs through
# expm1, with the log-space midpoint form taking over once the difference
# sinks under sqrt(eps) (its own error is O(s^2) exactly there). Mirrors the
# device `_device_t_log_normalizer`.
function _t_log_normalizer(nu, za, zb)
    zaf, zbf = promote(float(za), float(zb))
    lower_infinite = isinf(zaf) && zaf < zero(zaf)
    upper_infinite = isinf(zbf) && zbf > zero(zbf)
    if lower_infinite && upper_infinite
        return zero(zaf)
    elseif lower_infinite
        return _std_t_log_cdf(zbf, nu)
    elseif upper_infinite
        return _std_t_log_cdf(-zaf, nu)
    end
    if zaf > zero(zaf) # right tail: S(za) - S(zb) = cdf(-za) - cdf(-zb)
        log_big = _std_t_log_cdf(-zaf, nu)
        log_small = _std_t_log_cdf(-zbf, nu)
    else # left tail and straddling: cdf(zb) - cdf(za)
        log_big = _std_t_log_cdf(zbf, nu)
        log_small = _std_t_log_cdf(zaf, nu)
    end
    s = log_small - log_big
    s < -sqrt(eps(float(one(s)))) && return log_big + log(-expm1(s))
    midpoint = (zaf + zbf) / 2
    return _std_t_log_pdf(midpoint, nu) + log(zbf - zaf)
end

# The nu-only Student-t normalizing constant
# loggamma((nu+1)/2) - loggamma(nu/2) - (log(nu) + log(pi))/2, computed in at
# least Float64: at Float32 the two ~nu*log(nu)-sized loggammas lose their
# O(log nu) difference to rounding (~0.03 absolute at nu = 1e5 -- issue #53).
# The z-dependent terms have no such cancellation and stay at input precision.
function _studentt_log_constant(nu)
    nuf = float(nu)
    W = promote_type(typeof(nuf), Float64)
    nuw = W(nuf)
    return oftype(nuf, loggamma((nuw + one(nuw)) / 2) - loggamma(nuw / 2) - (log(nuw) + log(W(pi))) / 2)
end

# Its nu-derivative, (digamma((nu+1)/2) - digamma(nu/2) - 1/nu) / 2, widened
# the same way: the digamma difference is ~1/nu against ~log(nu)-sized terms,
# so the Float32 analytic gradient would otherwise disagree with the
# Float64-widened value the ForwardDiff reference differentiates.
function _studentt_log_constant_dnu(nu)
    nuf = float(nu)
    W = promote_type(typeof(nuf), Float64)
    nuw = W(nuf)
    return oftype(nuf, (digamma((nuw + one(nuw)) / 2) - digamma(nuw / 2) - one(nuw) / nuw) / 2)
end

# Standard (unit-scale, zero-location) Student-t log-density; `_std_t_pdf`
# exponentiates it, and the truncated normalizer (midpoint fallback and the
# gradient's pdf/Z hazard ratios) uses it directly. Computed in at least
# Float64: at Float32 the loggamma((nu+1)/2) - loggamma(nu/2) difference loses
# ~0.05 to rounding for large nu, and the truncated gradient's cancellation
# structure (-k + pdf/Z, two nearly equal hazards) amplifies that into
# multiple-hundred-percent gradient errors.
function _std_t_log_pdf(z, nu)
    zz = float(z)
    W = promote_type(typeof(zz), Float64)
    zw = W(zz)
    nuw = W(nu)
    return oftype(
        zz,
        loggamma((nuw + one(nuw)) / 2) - loggamma(nuw / 2) -
        (log(nuw) + log(W(pi))) / 2 -
        (nuw + one(nuw)) * log1p((zw * zw) / nuw) / 2,
    )
end

# Standard (unit-scale, zero-location) Student-t density with `nu` degrees of
# freedom, guarded so an infinite standardized argument (an unbounded truncation
# side) yields a zero density.
function _std_t_pdf(z, nu)
    zz = float(z)
    isinf(zz) && return zero(zz)
    return exp(_std_t_log_pdf(zz, nu))
end

# Peel a degrees-of-freedom argument down to a plain value. `TruncatedStudentTDist`
# promotes every field to a common type, so a constant `nu` still arrives as a
# ForwardDiff Dual carrying zero partials — that is the tractable case. A `nu`
# with nonzero partials is a genuine latent dependence whose CDF derivative has no
# closed form, so it is rejected rather than silently mis-differentiated.
_constant_nu_value(nu::Real) = nu
function _constant_nu_value(nu::ForwardDiff.Dual)
    all(iszero, ForwardDiff.partials(nu)) || throw(
        ArgumentError(
            "truncatedstudentt gradient with respect to nu (degrees of freedom) is unsupported; nu must be a constant",
        ),
    )
    return _constant_nu_value(ForwardDiff.value(nu))
end

# ForwardDiff rule for the Student-t CDF differentiated through its `z` argument.
# `beta_inc` is not itself dual-differentiable, but d/dz T_cdf(z, nu) is exactly
# the Student-t density, a closed form. This keeps the CPU truncatedstudentt
# logpdf ForwardDiff-differentiable whenever `nu` is a constant (the only
# tractable case: the incomplete beta's nu-derivative has no closed form).
function _std_t_cdf(z::ForwardDiff.Dual{T}, nu::Real) where {T}
    zv = ForwardDiff.value(z)
    # An infinite standardized bound pins the CDF at 0/1 with a flat (zero)
    # derivative, independent of `nu`. Handle it before requiring a constant `nu`:
    # an unbounded truncation side needs no `d/dnu` term, so a latent `nu` stays
    # valid here (e.g. both bounds infinite). Skipping the pdf * partials product
    # also avoids an infinite partial surfacing as 0 * Inf = NaN.
    if isinf(zv)
        value = zv > zero(zv) ? one(zv) : zero(zv)
        return ForwardDiff.Dual{T}(value, zero(ForwardDiff.partials(z)))
    end
    nu_value = _constant_nu_value(nu)
    value = _std_t_cdf(zv, nu_value)
    derivative = _std_t_pdf(zv, nu_value)
    return ForwardDiff.Dual{T}(value, derivative * ForwardDiff.partials(z))
end

# The log-CDF analogue: d/dz log T_cdf(z) = exp(log pdf - log cdf), computed in
# log space so the ratio stays finite where the plain cdf underflows (the same
# rule the device dual path uses).
function _std_t_log_cdf(z::ForwardDiff.Dual{T}, nu::Real) where {T}
    zv = ForwardDiff.value(z)
    if isinf(zv)
        value = zv > zero(zv) ? zero(zv) : oftype(zv, -Inf)
        return ForwardDiff.Dual{T}(value, zero(ForwardDiff.partials(z)))
    end
    nu_value = _constant_nu_value(nu)
    value = _std_t_log_cdf(zv, nu_value)
    derivative = exp(_std_t_log_pdf(zv, nu_value) - value)
    return ForwardDiff.Dual{T}(value, derivative * ForwardDiff.partials(z))
end

# The scalar CPU-reference logpdfs delegate to the single-source kernels in
# distributions/scalar_kernels.jl (issue #285). The kernels are the exact same
# formulas plus parameter-validation NaN branches, which are unreachable here
# because every constructor validates its parameters.
logpdf(dist::NormalDist, x) = _backend_normal_logpdf(dist.mu, dist.sigma, x)
logpdf(dist::LaplaceDist, x) = _backend_laplace_logpdf(dist.mu, dist.scale, x)
logpdf(dist::ExponentialDist, x) = _backend_exponential_logpdf(dist.rate, x)
logpdf(dist::GammaDist, x) = _backend_gamma_logpdf(dist.shape, dist.rate, x)
logpdf(dist::InverseGammaDist, x) = _backend_inversegamma_logpdf(dist.shape, dist.scale, x)
logpdf(dist::WeibullDist, x) = _backend_weibull_logpdf(dist.shape, dist.scale, x)
logpdf(dist::BetaDist, x) = _backend_beta_logpdf(dist.alpha, dist.beta, x)
logpdf(dist::LogNormalDist, x) = _backend_lognormal_logpdf(dist.mu, dist.sigma, x)
logpdf(dist::StudentTDist, x) = _backend_studentt_logpdf(dist.nu, dist.mu, dist.sigma, x)
logpdf(dist::CauchyDist, x) = _backend_cauchy_logpdf(dist.mu, dist.sigma, x)
logpdf(dist::HalfNormalDist, x) = _backend_halfnormal_logpdf(dist.sigma, x)
logpdf(dist::HalfCauchyDist, x) = _backend_halfcauchy_logpdf(dist.scale, x)
logpdf(dist::UniformDist, x) = _backend_uniform_logpdf(dist.lower, dist.upper, x)
logpdf(dist::LogisticDist, x) = _backend_logistic_logpdf(dist.mu, dist.scale, x)
logpdf(dist::GumbelDist, x) = _backend_gumbel_logpdf(dist.mu, dist.scale, x)
logpdf(dist::ParetoDist, x) = _backend_pareto_logpdf(dist.xm, dist.alpha, x)
logpdf(dist::FrechetDist, x) = _backend_frechet_logpdf(dist.shape, dist.scale, x)
logpdf(dist::RayleighDist, x) = _backend_rayleigh_logpdf(dist.scale, x)

logpdf(dist::InverseGaussianDist, x) = _backend_inversegaussian_logpdf(dist.mu, dist.lambda, x)
