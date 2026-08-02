# CPU-reference distributions (structs, builders, logpdf, rand): truncated families (truncatednormal, truncatedstudentt).

struct TruncatedNormalDist{T<:Real} <: AbstractTeaDistribution
    mu::T
    sigma::T
    lower::T
    upper::T

    function TruncatedNormalDist(mu::T, sigma::T, lower::T, upper::T) where {T<:Real}
        sigma > zero(T) || throw(ArgumentError("truncatednormal requires sigma > 0"))
        lower < upper || throw(ArgumentError("truncatednormal requires lower < upper"))
        new{T}(mu, sigma, lower, upper)
    end
end

struct TruncatedStudentTDist{T<:Real} <: AbstractTeaDistribution
    nu::T
    mu::T
    sigma::T
    lower::T
    upper::T

    function TruncatedStudentTDist(nu::T, mu::T, sigma::T, lower::T, upper::T) where {T<:Real}
        nu > zero(T) || throw(ArgumentError("truncatedstudentt requires nu > 0"))
        sigma > zero(T) || throw(ArgumentError("truncatedstudentt requires sigma > 0"))
        lower < upper || throw(ArgumentError("truncatedstudentt requires lower < upper"))
        new{T}(nu, mu, sigma, lower, upper)
    end
end

"""
    truncatednormal(mu, sigma, lower, upper)

Normal distribution with mean `mu` and standard deviation `sigma`, truncated
(and renormalized) to the interval `[lower, upper]`. Requires `lower < upper`.
"""
function truncatednormal(mu, sigma, lower, upper)
    promoted_mu, promoted_sigma, promoted_lower, promoted_upper = promote(mu, sigma, lower, upper)
    return TruncatedNormalDist(promoted_mu, promoted_sigma, promoted_lower, promoted_upper)
end

"""
    truncatedstudentt(nu, mu, sigma, lower, upper)

Student's t distribution with `nu` degrees of freedom, location `mu`, and scale
`sigma`, truncated (and renormalized) to the interval `[lower, upper]`.
Requires `lower < upper`.
"""
function truncatedstudentt(nu, mu, sigma, lower, upper)
    promoted_nu, promoted_mu, promoted_sigma, promoted_lower, promoted_upper =
        promote(nu, mu, sigma, lower, upper)
    return TruncatedStudentTDist(
        promoted_nu,
        promoted_mu,
        promoted_sigma,
        promoted_lower,
        promoted_upper,
    )
end

_std_normal_cdf(z) = erfc(-z / sqrt(oftype(float(z), 2))) / 2
_std_normal_invcdf(u) = sqrt(oftype(float(u), 2)) * erfinv(2 * u - one(u))

# log(erfc(a) - erfc(b)) for 0 < a < b, cancellation- and underflow-free
# (CPU port of the device `_device_log_erfc_diff`, issue #96):
# erfc(a) - erfc(b) = e^{-a^2} erfcx(a) (1 - e^{a^2-b^2} erfcx(b)/erfcx(a)),
# so the tail magnitude lives in the -a^2 log term and the parenthesis is
# -expm1(s) with s < 0. `s` carries ~eps-level absolute error from the erfcx
# ratio, so once |s| sinks under sqrt(eps) the expm1 form has lost half its
# digits -- there the caller-supplied log-space midpoint form takes over (its
# own error is O(s^2), negligible exactly where it is selected).
function _log_erfc_diff(a, b, log_fallback)
    ea = erfcx(a)
    eb = erfcx(b)
    s = (a - b) * (a + b) + log(eb / ea)
    s < -sqrt(eps(float(one(s)))) && return -a * a + log(ea) + log(-expm1(s))
    return log_fallback
end

# log(Phi(zb) - Phi(za)) with each branch computed in log space (CPU port of
# the device `_device_log_normal_cdf_diff`, issue #96): one-sided tails via
# logerfc (erfcx-based past erfc underflow), same-sign two-sided intervals via
# the erfcx difference identity (exact for narrow and far-tail intervals where
# plain erfc differences cancel or underflow), and the straddling branch
# directly (an addition -- it cannot cancel).
function _log_normal_cdf_diff(za, zb)
    root2 = sqrt(oftype(float(za), 2))
    log2 = log(oftype(float(za), 2))
    lower_infinite = isinf(za) && za < zero(za)
    upper_infinite = isinf(zb) && zb > zero(zb)
    # Handle infinite bounds explicitly: Φ(-∞)=0, Φ(+∞)=1. This keeps erfc off
    # arguments whose ForwardDiff partials would otherwise be NaN.
    if lower_infinite && upper_infinite
        return zero(float(za))
    elseif lower_infinite
        return logerfc(-zb / root2) - log2
    elseif upper_infinite
        return logerfc(za / root2) - log2
    end
    # log-space midpoint form for ulp-narrow same-sign intervals (mid^2 is
    # symmetric under the za <-> -zb mirror, so both tails share it); the +log2
    # aligns it with the erfc-difference scale the branches subtract log2 from
    mid = (za + zb) / 2
    log_fallback =
        -mid * mid / 2 - oftype(float(za), 0.9189385332046727) + log(abs(zb - za)) + log2
    if za > zero(za)
        return _log_erfc_diff(za / root2, zb / root2, log_fallback) - log2
    elseif zb < zero(zb)
        return _log_erfc_diff(-zb / root2, -za / root2, log_fallback) - log2
    end
    return log(erf(zb / root2) - erf(za / root2)) - log2
end

# Standard-normal survival function and its inverse: the right tail computed
# DIRECTLY (erfc is small there and keeps relative accuracy, and erfcinv is
# accurate down to subnormal arguments), so tail intervals invert without the
# Φ ~ 1 cancellation that made the old inverse-CDF sampler escape its support
# (issue #72).
_std_normal_ccdf(z) = erfc(z / sqrt(oftype(float(z), 2))) / 2
_std_normal_invccdf(s) = sqrt(oftype(float(s), 2)) * erfcinv(2 * s)

# Standardized truncated-normal draw on [a, b] with 0 <= a < b. The survival
# difference S(a) - S(b) is exact to relative rounding while S(a) is
# representable and the window is not ulp-narrow; past that, rejection
# sampling: Robert (1995) translated-exponential proposal for wide/one-sided
# tails, uniform proposal with the exp((a^2 - x^2)/2) ratio for narrow windows
# (acceptance ~1 exactly where the exponential proposal would reject on x > b).
function _rand_std_truncnormal_right(rng::AbstractRNG, a::T, b::T) where {T<:AbstractFloat}
    lower_survival = _std_normal_ccdf(a)
    upper_survival = isinf(b) ? zero(T) : _std_normal_ccdf(b)
    mass = lower_survival - upper_survival
    if mass > lower_survival * sqrt(eps(T))
        survival = upper_survival + rand(rng, T) * mass
        survival = min(max(survival, floatmin(T)), lower_survival)
        return clamp(_std_normal_invccdf(survival), a, b)
    end
    alpha = (a + sqrt(a * a + T(4))) / 2
    if isinf(b) || (b - a) * alpha > one(T)
        while true
            x = a + randexp(rng, T) / alpha
            x <= b || continue
            diff = x - alpha
            rand(rng, T) <= exp(-diff * diff / 2) && return x
        end
    end
    while true
        x = a + rand(rng, T) * (b - a)
        rand(rng, T) <= exp((a - x) * (a + x) / 2) && return x
    end
end

function _rand_std_truncnormal(rng::AbstractRNG, za::T, zb::T) where {T<:AbstractFloat}
    za >= zero(T) && return _rand_std_truncnormal_right(rng, za, zb)
    zb <= zero(T) && return -_rand_std_truncnormal_right(rng, -zb, -za)
    # straddling zero: both CDFs sit in the well-conditioned center, so the
    # plain inverse-CDF method keeps full accuracy (and the historical
    # single-uniform draw pattern)
    lower_cdf = _std_normal_cdf(za)
    upper_cdf = _std_normal_cdf(zb)
    threshold = lower_cdf + rand(rng, T) * (upper_cdf - lower_cdf)
    threshold = min(max(threshold, floatmin(T)), prevfloat(one(T)))
    return clamp(_std_normal_invcdf(threshold), za, zb)
end

function Random.rand(rng::AbstractRNG, dist::TruncatedNormalDist)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    lower = oftype(mu, dist.lower)
    upper = oftype(mu, dist.upper)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    z = _rand_std_truncnormal(rng, za, zb)
    return clamp(mu + sigma * z, lower, upper)
end

function Random.rand(rng::AbstractRNG, dist::TruncatedStudentTDist)
    nu = float(dist.nu)
    mu = float(dist.mu)
    sigma = float(dist.sigma)
    lower = float(dist.lower)
    upper = float(dist.upper)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    lower_cdf = _std_t_cdf(za, nu)
    upper_cdf = _std_t_cdf(zb, nu)
    mass = upper_cdf - lower_cdf

    if mass < oftype(mass, 1e-6)
        target = lower_cdf + rand(rng, typeof(mu)) * mass
        lo = isfinite(za) ? za : (isfinite(zb) ? zb - oftype(zb, 50) : oftype(nu, -50))
        hi = isfinite(zb) ? zb : (isfinite(za) ? za + oftype(za, 50) : oftype(nu, 50))
        for _ = 1:80
            mid = (lo + hi) / 2
            if _std_t_cdf(mid, nu) < target
                lo = mid
            else
                hi = mid
            end
        end
        return mu + sigma * (lo + hi) / 2
    end

    base = studentt(nu, mu, sigma)
    while true
        draw = rand(rng, base)
        (draw >= lower && draw <= upper) && return draw
    end
end

function logpdf(dist::TruncatedNormalDist, x)
    xx, mu, sigma, lower, upper = promote(x, dist.mu, dist.sigma, dist.lower, dist.upper)
    (xx < lower || xx > upper) && return oftype(xx, -Inf)
    base = logpdf(normal(mu, sigma), xx)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    return base - _log_normal_cdf_diff(za, zb)
end

function logpdf(dist::TruncatedStudentTDist, x)
    xx, nu, mu, sigma, lower, upper =
        promote(x, dist.nu, dist.mu, dist.sigma, dist.lower, dist.upper)
    (xx < lower || xx > upper) && return oftype(xx, -Inf)
    base = logpdf(studentt(nu, mu, sigma), xx)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    return base - _t_log_normalizer(nu, za, zb)
end
