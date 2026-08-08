# Zero-inflated count distributions (issue #292): excess-zero count data
# (ecology, insurance, health-care utilization) mixes a point mass at zero with
# a count component. With inflation probability `p`:
#
#     P(y = 0) = p + (1 - p) * P_count(0)
#     P(y = k) = (1 - p) * P_count(k)          (k >= 1)
#
# scored in log space via logaddexp for the zero branch. Both families reuse
# the single-source count kernels (issue #285); all parameters may be
# latent-flowing. CPU-reference only (honestly backend/device unsupported).

struct ZeroInflatedPoissonDist{P,L} <: AbstractTeaDistribution
    p::P             # inflation (structural-zero) probability
    lambda::L

    function ZeroInflatedPoissonDist(p, lambda)
        zero(_primal(p)) <= _primal(p) <= one(_primal(p)) ||
            throw(ArgumentError("zeroinflatedpoisson requires 0 <= p <= 1"))
        _primal(lambda) > zero(_primal(lambda)) ||
            throw(ArgumentError("zeroinflatedpoisson requires lambda > 0"))
        return new{typeof(p),typeof(lambda)}(p, lambda)
    end
end

struct ZeroInflatedNegativeBinomialDist{P,S,Q} <: AbstractTeaDistribution
    p::P             # inflation (structural-zero) probability
    successes::S
    probability::Q

    function ZeroInflatedNegativeBinomialDist(p, successes, probability)
        zero(_primal(p)) <= _primal(p) <= one(_primal(p)) ||
            throw(ArgumentError("zeroinflatednegativebinomial requires 0 <= p <= 1"))
        _primal(successes) > zero(_primal(successes)) ||
            throw(ArgumentError("zeroinflatednegativebinomial requires successes > 0"))
        zero(_primal(probability)) < _primal(probability) <= one(_primal(probability)) ||
            throw(ArgumentError("zeroinflatednegativebinomial requires 0 < probability <= 1"))
        return new{typeof(p),typeof(successes),typeof(probability)}(p, successes, probability)
    end
end

"""
    zeroinflatedpoisson(p, lambda)
    zeroinflatednegativebinomial(p, successes, probability)

Zero-inflated count distributions: a structural zero with probability `p`
mixed with a Poisson(`lambda`) / NegativeBinomial(`successes`, `probability`)
count, so `P(y = 0) = p + (1 - p)·P_count(0)` and `P(y = k) = (1 - p)·P_count(k)`
for `k ≥ 1`. All parameters may be latent-flowing (`p` typically via a logistic
link). CPU-reference only.
"""
zeroinflatedpoisson(p, lambda) = ZeroInflatedPoissonDist(p, lambda)
zeroinflatednegativebinomial(p, successes, probability) =
    ZeroInflatedNegativeBinomialDist(p, successes, probability)

# The shared docstring binds only to the first definition; mirror it onto the
# second so `@ref` links and `?` lookups resolve.
@doc (@doc zeroinflatedpoisson) zeroinflatednegativebinomial

# log(exp(a) + exp(b)) without overflow
@inline function _zi_logaddexp(a, b)
    m = max(a, b)
    isfinite(m) || return m
    return m + log(exp(a - m) + exp(b - m))
end

@inline function _zero_inflated_logpdf(p, count_log0, count_logk, count)
    if count == 0
        # log(p + (1 - p) exp(count_log0))
        return _zi_logaddexp(log(p), log1p(-p) + count_log0)
    end
    return log1p(-p) + count_logk
end

function logpdf(dist::ZeroInflatedPoissonDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.lambda), -Inf)
    # p == 0 degenerates to the plain count; p == 1 puts all mass at zero
    _primal(dist.p) == one(_primal(dist.p)) &&
        return count == 0 ? zero(float(dist.lambda)) : oftype(float(dist.lambda), -Inf)
    logk = _backend_poisson_logpdf(dist.lambda, count)
    _primal(dist.p) == zero(_primal(dist.p)) && return logk
    log0 = count == 0 ? logk : _backend_poisson_logpdf(dist.lambda, 0)
    return _zero_inflated_logpdf(dist.p, log0, logk, count)
end

function logpdf(dist::ZeroInflatedNegativeBinomialDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.probability), -Inf)
    _primal(dist.p) == one(_primal(dist.p)) &&
        return count == 0 ? zero(float(dist.probability)) : oftype(float(dist.probability), -Inf)
    logk = _backend_negativebinomial_logpdf(dist.successes, dist.probability, count)
    _primal(dist.p) == zero(_primal(dist.p)) && return logk
    log0 = count == 0 ? logk : _backend_negativebinomial_logpdf(dist.successes, dist.probability, 0)
    return _zero_inflated_logpdf(dist.p, log0, logk, count)
end

function Random.rand(rng::AbstractRNG, dist::ZeroInflatedPoissonDist)
    rand(rng) < dist.p && return 0
    return rand(rng, poisson(dist.lambda))
end

function Random.rand(rng::AbstractRNG, dist::ZeroInflatedNegativeBinomialDist)
    rand(rng) < dist.p && return 0
    return rand(rng, negativebinomial(dist.successes, dist.probability))
end
