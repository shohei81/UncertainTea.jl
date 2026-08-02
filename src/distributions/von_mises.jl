# von Mises distribution for circular/angular data (issue #293): wind
# directions, phases, time-of-day. Density on angles theta in [-pi, pi):
#
#     p(x | mu, kappa) = exp(kappa * cos(x - mu)) / (2 pi I0(kappa))
#
# `log I0(kappa)` is computed by an AD-generic pair (power series for small
# kappa, asymptotic expansion for large kappa) rather than
# SpecialFunctions.besselix, so ForwardDiff/Enzyme duals flow through latent
# `mu`/`kappa` without needing Bessel derivative rules. CPU-reference only.
# There is no circular/wrapped transform yet, so a latent `mu` samples
# unconstrained (the density is 2pi-periodic in mu; document the
# identifiability caveat) and `kappa > 0` flows through an exp link.

struct VonMisesDist{M,K} <: AbstractTeaDistribution
    mu::M
    kappa::K

    function VonMisesDist(mu, kappa)
        kappa > zero(kappa) || throw(ArgumentError("vonmises requires kappa > 0"))
        return new{typeof(mu),typeof(kappa)}(mu, kappa)
    end
end

"""
    vonmises(mu, kappa)

von Mises distribution on the circle: mean direction `mu` (radians) and
concentration `kappa > 0` (`kappa -> 0` approaches the circular uniform;
large `kappa` approaches `normal(mu, 1/sqrt(kappa))`). Observed values are
scored on any real angle through `cos(x - mu)` (2π-periodic); a latent `mu`
samples unconstrained, so wrap it for reporting. CPU-reference only.
"""
vonmises(mu, kappa) = VonMisesDist(mu, kappa)

# log I0(x) for x >= 0, AD-generic. Adaptive power series
# sum_k (x^2/4)^k / (k!)^2 for x <= 300 (I0(300) ~ 1e128 stays comfortably in
# Float64 range; terms peak near k ~ x/2 and the loop stops once a term stops
# moving the total); asymptotic expansion log I0(x) = x - log(2 pi x)/2 +
# log1p(1/(8x) + 9/(128 x^2) + 225/(3072 x^3)) beyond, where its relative
# error is ~ (8x)^-4 < 1e-13.
@inline function _log_besseli0(x)
    if x <= oftype(x, 300)
        q = x * x / 4
        term = one(q)
        total = one(q)
        for k = 1:400
            term *= q / (k * k)
            new_total = total + term
            new_total == total && break
            total = new_total
        end
        return log(total)
    end
    inv_x = inv(x)
    correction = inv_x * (oftype(x, 1 / 8) + inv_x * (oftype(x, 9 / 128) + inv_x * oftype(x, 225 / 3072)))
    return x - log(2 * oftype(x, pi) * x) / 2 + log1p(correction)
end

function logpdf(dist::VonMisesDist, x)
    xx, mu, kappa = promote(x, dist.mu, dist.kappa)
    return kappa * cos(xx - mu) - log(2 * oftype(xx, pi)) - _log_besseli0(kappa)
end

# Backstop for the rejection loops below (issue #342). Both loops accept with
# probability >= ~0.5 per iteration in their regimes, so 10_000 consecutive
# rejections (probability <= 2^-10000) can only mean a numerical pathology;
# error out instead of hanging generate/prior_predictive/SBC silently.
const _VONMISES_MAX_REJECTIONS = 10_000

# Best-Fisher (1979) rejection sampler, the standard von Mises draw; the result
# is wrapped into [-pi, pi). Extreme-kappa guards (issue #342): the core
# sampler's setup breaks down at both ends of the kappa range —
#
#   * small kappa: `tau - sqrt(2 tau)` cancels catastrophically once kappa^2
#     approaches eps(1.0) ~ 2.2e-16; for kappa <~ 1.5e-8 it rounds to 0, so
#     rho = 0, r = Inf, f = NaN and neither accept test ever fires (hang).
#   * large kappa: `4 kappa^2` overflows for kappa >~ 6.7e153, so tau = Inf,
#     rho = NaN and again the loop never accepts.
#
# Each regime gets an explicit branch; the core sampler is untouched in
# between (Best-Fisher is exact for ANY rho in (0, 1) — the accept test
# c * exp(1 - c) <= 1 holds identically — so the residual ~1e-4 relative
# rounding of rho at the 1e-6 crossover only nudges efficiency, not
# correctness).
function Random.rand(rng::AbstractRNG, dist::VonMisesDist)
    mu = float(dist.mu)
    kappa = float(dist.kappa)
    if kappa < 1e-6
        # Near-uniform regime: exact rejection sampling with a circular-uniform
        # proposal. Target density is proportional to exp(kappa * cos(delta))
        # for delta = theta - mu; with envelope constant exp(kappa) the accept
        # probability exp(kappa * (cos(delta) - 1)) lies in [exp(-2 kappa), 1],
        # i.e. >= 1 - 2e-6 here, so the draw is exactly von Mises and the loop
        # essentially never iterates (no small-kappa pathology possible).
        for _ = 1:_VONMISES_MAX_REJECTIONS
            delta = (2 * rand(rng) - 1) * pi
            if rand(rng) <= exp(kappa * (cos(delta) - 1))
                return mod2pi(mu + delta + pi) - pi
            end
        end
        error(
            "vonmises rand: small-kappa rejection sampler failed to accept after " *
            "$(_VONMISES_MAX_REJECTIONS) iterations (kappa = $kappa)",
        )
    elseif kappa > 1e8
        # Sharp-peak regime: von Mises is a wrapped normal(mu, 1/sqrt(kappa))
        # to double precision. With sigma = 1/sqrt(kappa) < 1e-4, the density
        # ratio exp(-kappa (1 - cos t)) / exp(-kappa t^2 / 2) =
        # exp(kappa t^4 / 24 + O(kappa t^6)) deviates from 1 by < 3e-7 even at
        # |t| = 5 sigma, and the wrapped mass beyond pi is < erfc(pi / sigma)
        # ~ 0, so a single wrapped Gaussian draw is exact to double precision.
        theta = mu + randn(rng) / sqrt(kappa)
        return mod2pi(theta + pi) - pi
    end
    tau = 1 + sqrt(1 + 4 * kappa * kappa)
    rho = (tau - sqrt(2 * tau)) / (2 * kappa)
    r = (1 + rho * rho) / (2 * rho)
    for _ = 1:_VONMISES_MAX_REJECTIONS
        u1, u2, u3 = rand(rng), rand(rng), rand(rng)
        z = cos(pi * u1)
        f = (1 + r * z) / (r + z)
        c = kappa * (r - f)
        if c * (2 - c) - u2 > 0 || log(c / u2) + 1 - c >= 0
            theta = mu + sign(u3 - 0.5) * acos(clamp(f, -1.0, 1.0))
            return mod2pi(theta + pi) - pi
        end
    end
    error("vonmises rand: Best-Fisher sampler failed to accept after " *
          "$(_VONMISES_MAX_REJECTIONS) iterations (kappa = $kappa)")
end
