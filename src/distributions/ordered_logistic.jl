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
        issorted(cutpoints; by=_primal) ||
            throw(ArgumentError("orderedlogistic cutpoints must be increasing"))
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

# Stable log(1 - exp(d)) for d <= 0 (Maechler 2012): near zero -exp(d) would
# cancel against 1, so difference through expm1 there; in the far tail exp is
# exact and log1p keeps full precision.
@inline _ordlogit_log1m_exp(d) =
    d > -0.6931471805599453 ? log(-expm1(d)) : log1p(-exp(d))

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
    # log(sigma(u - eta) - sigma(l - eta)) with u > l. Differencing the two
    # log-CDFs is only well-conditioned when both CDFs are small (eta at or
    # above the cutpoints); for eta far below both CDFs saturate to 1 and the
    # ratio -> 1 cancels catastrophically (issue #344, -Inf at eta <= -37).
    # Mirror the tail there: P(l < Y <= u) = CCDF(l) - CCDF(u), whose logs
    # stay well separated for eta below the cutpoints. Branch on the midpoint
    # (a plain value comparison, so ForwardDiff Duals flow through either arm).
    if 2 * eta > lower + upper
        log_upper = _ordlogit_log_cdf(eta, upper)
        log_lower = _ordlogit_log_cdf(eta, lower)
        return log_upper + _ordlogit_log1m_exp(log_lower - log_upper)
    else
        log_lower = _ordlogit_log_ccdf(eta, lower)
        log_upper = _ordlogit_log_ccdf(eta, upper)
        return log_lower + _ordlogit_log1m_exp(log_upper - log_lower)
    end
end

function Random.rand(rng::AbstractRNG, dist::OrderedLogisticDist)
    u = rand(rng)
    eta = dist.eta
    for (k, c) in enumerate(dist.cutpoints)
        u <= exp(_ordlogit_log_cdf(eta, c)) && return k
    end
    return length(dist.cutpoints) + 1
end
