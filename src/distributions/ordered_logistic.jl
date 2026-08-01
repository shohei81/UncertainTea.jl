# Ordered logistic (cumulative-logit) distribution for ordinal outcomes
# (issue #291): the staple likelihood for Likert scales, severity grades, and
# ranked categories. An outcome k in 1..K is scored against a latent linear
# predictor `eta` and K-1 strictly increasing cutpoints c_1 < ... < c_{K-1}:
#
#     P(y <= k) = logistic(c_k - eta)
#     P(y = 1)  = logistic(c_1 - eta)
#     P(y = k)  = logistic(c_k - eta) - logistic(c_{k-1} - eta)
#     P(y = K)  = 1 - logistic(c_{K-1} - eta)
#
# computed in log space through the stable `log1p(exp(...))` helpers shared
# with bernoullilogit. CPU-reference only (honestly backend/device
# unsupported). There is no ordered-vector transform yet, so LATENT cutpoints
# should be given an increasing parameterization by hand (a base plus
# log-gap increments), the same identifiability recipe the HMM means use.

struct OrderedLogisticDist{T,C<:AbstractVector} <: AbstractTeaDistribution
    eta::T
    cutpoints::C

    function OrderedLogisticDist(eta, cutpoints::AbstractVector)
        length(cutpoints) >= 1 ||
            throw(ArgumentError("orderedlogistic requires at least one cutpoint"))
        issorted(cutpoints) || throw(ArgumentError("orderedlogistic cutpoints must be increasing"))
        return new{typeof(eta),typeof(cutpoints)}(eta, cutpoints)
    end
end

"""
    orderedlogistic(eta, cutpoints)

Ordered logistic (cumulative-logit) distribution over the ordinal categories
`1..K`, with linear predictor `eta` and `K - 1` strictly increasing
`cutpoints`. `P(y <= k) = logistic(cutpoints[k] - eta)`. `eta` (and the
cutpoints) may be latent-flowing; give latent cutpoints an increasing
parameterization (base + `exp` gaps) since there is no ordered transform yet.
CPU-reference only.
"""
orderedlogistic(eta, cutpoints) = OrderedLogisticDist(eta, cutpoints)

# log P(y <= k) = -log1p(exp(eta - c_k)) and log P(y > k) = -log1p(exp(c_k - eta)),
# via the overflow-free log1p_exp shared with bernoullilogit.
@inline _ordlogit_log_cdf(eta, c) = -_bernoullilogit_log1p_exp(eta - c)
@inline _ordlogit_log_ccdf(eta, c) = -_bernoullilogit_log1p_exp(c - eta)

function logpdf(dist::OrderedLogisticDist, x)
    K = length(dist.cutpoints) + 1
    category = _categorical_index(x, K)
    isnothing(category) && return oftype(float(dist.eta), -Inf)
    eta = dist.eta
    if category == 1
        return _ordlogit_log_cdf(eta, dist.cutpoints[1])
    elseif category == K
        return _ordlogit_log_ccdf(eta, dist.cutpoints[K-1])
    end
    lower = dist.cutpoints[category-1]
    upper = dist.cutpoints[category]
    # log(sigma(u - eta) - sigma(l - eta)) with u > l, in log space:
    # log_cdf(u) + log1p(-exp(log_cdf(l) - log_cdf(u)))
    log_upper = _ordlogit_log_cdf(eta, upper)
    log_lower = _ordlogit_log_cdf(eta, lower)
    return log_upper + log1p(-exp(log_lower - log_upper))
end

function Random.rand(rng::AbstractRNG, dist::OrderedLogisticDist)
    u = rand(rng)
    eta = dist.eta
    for (k, c) in enumerate(dist.cutpoints)
        u <= exp(_ordlogit_log_cdf(eta, c)) && return k
    end
    return length(dist.cutpoints) + 1
end
