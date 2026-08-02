# CPU-reference distributions (structs, builders, logpdf, rand): discrete families (bernoulli, poisson, geometric, binomial, negativebinomial, categorical).

struct BernoulliDist{T<:Real} <: AbstractTeaDistribution
    p::T

    function BernoulliDist(p::T) where {T<:Real}
        zero(T) <= p <= one(T) || throw(ArgumentError("bernoulli requires 0 <= p <= 1"))
        new{T}(p)
    end
end

struct GeometricDist{T<:Real} <: AbstractTeaDistribution
    p::T

    function GeometricDist(p::T) where {T<:Real}
        zero(T) < p <= one(T) || throw(ArgumentError("geometric requires 0 < p <= 1"))
        new{T}(p)
    end
end

# Logit-parameterized Bernoulli (issue #149): the success probability is
# `logistic(eta) = 1/(1+exp(-eta))`, but the log density is scored in the stable
# log-scale form `x*eta - log1p(exp(eta))` so gradients stay finite where the
# probability saturates to 0/1 (|eta| >~ 37 in Float64). The naive
# `bernoulli(1/(1+exp(-eta)))` spelling scores `log(p)`/`log1p(-p)` and gives
# -Inf/NaN there. `eta` is stored through `float` so integer literals and
# ForwardDiff Duals both flow correctly.
struct BernoulliLogitDist{T<:Real} <: AbstractTeaDistribution
    eta::T

    BernoulliLogitDist(eta::T) where {T<:Real} = new{T}(eta)
end

struct BinomialDist{T<:Real} <: AbstractTeaDistribution
    trials::Int
    p::T

    function BinomialDist(trials::Int, p::T) where {T<:Real}
        trials >= 0 || throw(ArgumentError("binomial requires trials >= 0"))
        zero(T) <= p <= one(T) || throw(ArgumentError("binomial requires 0 <= p <= 1"))
        new{T}(trials, p)
    end
end

struct NegativeBinomialDist{T<:Real} <: AbstractTeaDistribution
    successes::T
    p::T

    function NegativeBinomialDist(successes::T, p::T) where {T<:Real}
        successes > zero(T) || throw(ArgumentError("negativebinomial requires successes > 0"))
        zero(T) < p <= one(T) || throw(ArgumentError("negativebinomial requires 0 < p <= 1"))
        new{T}(successes, p)
    end
end

struct CategoricalDist{T<:Real} <: AbstractTeaDistribution
    probabilities::Vector{T}

    function CategoricalDist(probabilities::Vector{T}) where {T<:Real}
        isempty(probabilities) && throw(ArgumentError("categorical requires at least one probability"))
        total = zero(T)
        for probability in probabilities
            zero(T) <= probability <= one(T) || throw(ArgumentError("categorical requires 0 <= p <= 1"))
            total += probability
        end
        tolerance = sqrt(eps(float(total))) * max(length(probabilities), 1) * 8
        abs(total - one(total)) <= tolerance || throw(ArgumentError("categorical probabilities must sum to 1"))
        new{T}(probabilities)
    end
end

struct PoissonDist{T<:Real} <: AbstractTeaDistribution
    lambda::T

    function PoissonDist(lambda::T) where {T<:Real}
        lambda > zero(T) || throw(ArgumentError("poisson requires lambda > 0"))
        new{T}(lambda)
    end
end

# Builders normalize parameters through `float` so integer (or other non-float
# real) literals reach the samplers as float storage (issue #73); `float` keeps
# ForwardDiff Duals intact.
"""
    bernoulli(p)

Bernoulli distribution over `{0, 1}` with success probability `p`.
"""
function bernoulli(p)
    return BernoulliDist(float(p))
end

"""
    geometric(p)

Geometric distribution over `0, 1, 2, ...`: the number of failures before the
first success in Bernoulli trials with success probability `p`.
"""
function geometric(p)
    return GeometricDist(float(p))
end

"""
    bernoullilogit(eta)

Bernoulli distribution parameterized on the logit scale: success probability
`logistic(eta) = 1 / (1 + exp(-eta))`, evaluated in a numerically stable way
(no manual clamping needed).
"""
function bernoullilogit(eta)
    return BernoulliLogitDist(float(eta))
end

# Numerically stable `log(1 + exp(eta))`: the branch keeps the exponentiated
# argument non-positive, so it never overflows and stays finite at |eta| = 90
# where a naive `log(1 + exp(eta))` would overflow to Inf.
_bernoullilogit_log1p_exp(eta) =
    eta > zero(eta) ? eta + log1p(exp(-eta)) : log1p(exp(eta))

# Stable `logistic(eta) = 1/(1 + exp(-eta))`, evaluated so the exponent is never
# positive (no overflow); returns exactly 0/1 only in the true float limit.
function _bernoullilogit_logistic(eta)
    if eta >= zero(eta)
        return inv(one(eta) + exp(-eta))
    end
    expeta = exp(eta)
    return expeta / (one(eta) + expeta)
end

function binomial(trials, p)
    count = _binomial_trials(trials)
    isnothing(count) && throw(ArgumentError("binomial requires integer trials >= 0"))
    return BinomialDist(count, float(p))
end

"""
    negativebinomial(successes, p)

Negative binomial distribution over `0, 1, 2, ...`: the number of failures
before `successes` successes in Bernoulli trials with success probability `p`.
`successes` may be non-integer (the Polya/gamma-Poisson generalization).
"""
function negativebinomial(successes, p)
    promoted_successes, promoted_probability = promote(float(successes), float(p))
    return NegativeBinomialDist(promoted_successes, promoted_probability)
end

"""
    categorical(probabilities)
    categorical(p1, p2, ...)

Categorical distribution over the integer categories `1:K`, where `K` is the
number of probabilities supplied (as a vector or as separate arguments).
"""
function categorical(probabilities::AbstractVector)
    promoted = map(float, collect(probabilities))
    return CategoricalDist(promoted)
end

function categorical(probabilities::Vararg{Real})
    promoted = collect(promote(map(float, probabilities)...))
    return CategoricalDist(promoted)
end

"""
    poisson(lambda)

Poisson distribution over `0, 1, 2, ...` with mean (rate) `lambda`.
"""
function poisson(lambda)
    return PoissonDist(float(lambda))
end

function Random.rand(rng::AbstractRNG, dist::BernoulliDist)
    return rand(rng) < dist.p
end

function Random.rand(rng::AbstractRNG, dist::BernoulliLogitDist)
    return rand(rng) < _bernoullilogit_logistic(float(dist.eta))
end

function Random.rand(rng::AbstractRNG, dist::GeometricDist)
    probability = float(dist.p)
    probability == one(probability) && return 0
    threshold = max(rand(rng, typeof(probability)), floatmin(typeof(probability)))
    return floor(Int, log(threshold) / log1p(-probability))
end

function Random.rand(rng::AbstractRNG, dist::BinomialDist)
    successes = 0
    for _ = 1:dist.trials
        successes += rand(rng) < dist.p
    end
    return successes
end

function Random.rand(rng::AbstractRNG, dist::NegativeBinomialDist)
    probability = float(dist.p)
    probability == one(probability) && return 0
    rate = probability / (1 - probability)
    lambda = _rand_gamma_marsaglia(rng, float(dist.successes), rate)
    return rand(rng, poisson(lambda))
end

function Random.rand(rng::AbstractRNG, dist::CategoricalDist)
    threshold = rand(rng, eltype(dist.probabilities))
    cumulative = zero(threshold)
    for (index, probability) in enumerate(dist.probabilities)
        cumulative += probability
        threshold <= cumulative && return index
    end
    return length(dist.probabilities)
end

# Knuth's product-of-uniforms sampler needs exp(-lambda) to stay above zero, so
# it only serves small rates; above the threshold the transformed rejection
# sampler with squeeze (PTRS, Hoermann 1993) takes over (issue #74).
const _POISSON_KNUTH_MAX_LAMBDA = 30.0

# PTRS: valid for lambda >= 10; the acceptance test compares against the exact
# log-pmf (via loggamma), so draws are unbiased for arbitrarily large rates.
function _rand_poisson_ptrs(rng::AbstractRNG, lambda::Float64)
    b = 0.931 + 2.53 * sqrt(lambda)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2.0)
    log_lambda = log(lambda)
    while true
        u = rand(rng) - 0.5
        v = rand(rng)
        us = 0.5 - abs(u)
        k = floor(Int, (2.0 * a / us + b) * u + lambda + 0.43)
        if us >= 0.07 && v <= v_r
            return k
        end
        if k < 0 || (us < 0.013 && v > us)
            continue
        end
        if log(v) + log(inv_alpha) - log(a / (us * us) + b) <=
           k * log_lambda - lambda - loggamma(k + 1.0)
            return k
        end
    end
end

function Random.rand(rng::AbstractRNG, dist::PoissonDist)
    lambda = float(dist.lambda)
    lambda == zero(lambda) && return 0
    lambda > _POISSON_KNUTH_MAX_LAMBDA && return _rand_poisson_ptrs(rng, Float64(lambda))
    limit = exp(-lambda)
    product = one(lambda)
    count = 0
    while product > limit
        count += 1
        product *= rand(rng, typeof(lambda))
    end
    return count - 1
end

# Bernoulli support: `Bool`, or a real that compares equal to 0 or 1 (issue #85);
# anything else scores -Inf. Shared by the CPU logpdf and the backend scorer.
_bernoulli_value(x::Bool) = x
_bernoulli_value(x::Real) = x == zero(x) ? false : (x == one(x) ? true : nothing)
_bernoulli_value(@nospecialize(x)) = nothing

# The scalar discrete logpdfs delegate to the single-source kernels in
# distributions/scalar_kernels.jl (issue #285, stage 2); the parameter-validation
# throws are unreachable behind validated constructors.
logpdf(dist::BernoulliDist, x) = _backend_bernoulli_logpdf(dist.p, x)

# Support handling mirrors `bernoulli` (Bool or a numeric 0/1); the density is
# the stable log-scale form `x*eta - log1p(exp(eta))`.
logpdf(dist::BernoulliLogitDist, x) = _backend_bernoullilogit_logpdf(dist.eta, x)

logpdf(dist::GeometricDist, x) = _backend_geometric_logpdf(dist.p, x)

function _poisson_count(x)
    if x isa Integer
        return x >= 0 ? Int(x) : nothing
    elseif x isa Real && isfinite(x)
        truncated = trunc(x)
        (x >= zero(x) && x == truncated) || return nothing
        # Integer-valued floats beyond typemax(Int) stay on the float path
        # (issue #345): they are valid probability-mass points and must score
        # (the saddle-point kernels handle them), not throw InexactError.
        truncated <= typemax(Int) || return float(truncated)
        return Int(truncated)
    end
    return nothing
end

_binomial_trials(x) = _poisson_count(x)

function _categorical_index(x, categories::Int)
    if x isa Integer
        return 1 <= x <= categories ? Int(x) : nothing
    elseif x isa Real && isfinite(x)
        truncated = trunc(x)
        return one(x) <= x <= categories && x == truncated ? Int(truncated) : nothing
    end
    return nothing
end

function logpdf(dist::CategoricalDist, x)
    index = _categorical_index(x, length(dist.probabilities))
    isnothing(index) && return oftype(float(dist.probabilities[1]), -Inf)
    return log(dist.probabilities[index])
end

# Log-factorial of the data count `n`. The count is data, so its derivative is
# exactly zero: compute `loggamma(n + 1)` in plain Float64 (O(1), and more
# accurate than summing `log(k)`) and convert to the caller's arithmetic type
# so dual/Float32 callers keep their element type (issue #148). The device path
# already does this (src/device/math.jl).
function _logfactorial_like(value, n::Integer)
    zero_like = log(one(value))
    n < 2 && return zero_like
    return oftype(zero_like, loggamma(n + 1.0))
end

# Integer-valued float counts beyond typemax(Int) (issue #345): Stirling with
# the same `_stirlerr` correction the saddle-point kernels use (loggamma(n + 1)
# would be equally accurate; this spelling avoids the Int conversion the
# `::Integer` method assumes).
function _logfactorial_like(value, n::Real)
    zero_like = log(one(value))
    nf = Float64(n)
    return oftype(zero_like, (nf + 0.5) * log(nf) - nf + log(2 * pi) / 2 + _stirlerr(nf))
end

# --- issue #345: stirlerr/saddle-point core for extreme counts ----------------
#
# For counts past ~1e8 the naive count-family spellings difference loggammas
# (and count * log(lambda)-sized products) of magnitude ~n*log(n); their O(1)
# result sinks under the eps * n * log(n) absolute rounding, and past ~1e15 the
# logpdf even goes POSITIVE (binomial(1e16, 0.5) at n/2 scored +35 against the
# true -18.65). Following R's dbinom/dpois (Loader's saddle-point algorithm),
# split log(n!) into the exactly-computable Stirling part and the small
# `stirlerr` correction, and fold the p/lambda terms into `_bd0` deviance terms
# so no large intermediate is ever differenced. Kernels keep the exact
# small-count path below `_COUNT_SADDLE_THRESHOLD` (bit-identical to the
# historical results); the saddle path is accurate to ~2 ulp for every count
# above it (empirical study against BigFloat, issue #345).
const _COUNT_SADDLE_THRESHOLD = 100_000_000

# stirlerr(n) = log(n!) - [n log n - n + log(2 pi n)/2]. Exact via loggamma
# below 16 (also serves non-integer negativebinomial successes); the Stirling
# series in 1/n^2 above (error < 1e-15 relative for n >= 16).
function _stirlerr(n)
    nf = float(n)
    if nf < 16
        return loggamma(nf + one(nf)) - (nf + oftype(nf, 0.5)) * log(nf) + nf -
               log(2 * oftype(nf, pi)) / 2
    end
    inv_n2 = one(nf) / (nf * nf)
    return evalpoly(inv_n2, (1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188)) / nf
end

# Binomial deviance bd0(x, np) = x log(x/np) + np - x >= 0, stable where x and
# np nearly cancel (R's bd0): for |x - np| < 0.1 (x + np) sum the series in
# v = (x - np)/(x + np) until the primal stops moving; otherwise the direct
# formula has no cancellation. AD-generic: np (and x for negativebinomial
# successes) may carry ForwardDiff duals, whose partials converge at the same
# v^2 rate as the primal.
function _bd0(x, np)
    xf, npf = promote(float(x), float(np))
    if abs(xf - npf) < 0.1 * (xf + npf)
        v = (xf - npf) / (xf + npf)
        s = (xf - npf) * v
        ej = 2 * xf * v
        v2 = v * v
        for j = 1:1000
            ej *= v2
            s_next = s + ej / (2 * j + 1)
            s_next == s && return s_next
            s = s_next
        end
        return s
    end
    return xf * log(xf / npf) + npf - xf
end

# R dpois_raw: log dpois(k; lambda) = -stirlerr(k) - bd0(k, lambda) -
# log(2 pi k)/2 for k > 0.
function _poisson_logpdf_saddle(lambda, count)
    lambdaf = float(lambda)
    kf = Float64(count)
    return oftype(lambdaf, -_stirlerr(kf) - _bd0(kf, lambdaf) - log(2 * pi * kf) / 2)
end

# R dbinom_raw. `trials - count` is formed in the caller's exact (integer)
# arithmetic before the float conversion, and the log(2 pi k (n-k) / n)
# normalizer uses explicit logs -- log1p(-k/n) would hit -Inf when k/n rounds
# to 1 (k = n - 1 at n >= 1e16).
function _binomial_logpdf_saddle(trials, probability, count)
    pf = float(probability)
    if count == 0
        probability == one(probability) && return oftype(pf, -Inf)
        return oftype(pf, Float64(trials) * log1p(-pf))
    elseif count == trials
        probability == zero(probability) && return oftype(pf, -Inf)
        return oftype(pf, Float64(trials) * log(pf))
    end
    (probability == zero(probability) || probability == one(probability)) &&
        return oftype(pf, -Inf)
    nf = Float64(trials)
    kf = Float64(count)
    mf = Float64(trials - count)
    lc = _stirlerr(nf) - _stirlerr(kf) - _stirlerr(mf) -
         _bd0(kf, nf * pf) - _bd0(mf, nf * (one(pf) - pf))
    return oftype(pf, lc - (log(2 * pi) + log(kf) + log(mf) - log(nf)) / 2)
end

# R dnbinom: log dnbinom(k; r, p) = dbinom_raw(x = r, n = k + r, p, 1 - p) +
# log(r / (r + k)). With n p = (r + k) r / (r + k) = r at k = mean, both bd0
# terms stay near zero exactly where the mass concentrates.
function _negativebinomial_logpdf_saddle(successes, probability, count)
    pf = float(probability)
    rf = float(successes)
    kf = Float64(count)
    nf = kf + rf
    lc = _stirlerr(nf) - _stirlerr(rf) - _stirlerr(kf) -
         _bd0(rf, nf * pf) - _bd0(kf, nf * (one(pf) - pf))
    return oftype(pf, lc - (log(2 * pi) + log(rf) + log(kf) - log(nf)) / 2 + log(rf) - log(nf))
end

function _logbinomial_like(value, n::Integer, k::Integer)
    if n >= _COUNT_SADDLE_THRESHOLD
        return _logbinomial_like_saddle(value, n, k)
    end
    return _logfactorial_like(value, n) -
           _logfactorial_like(value, k) -
           _logfactorial_like(value, n - k)
end

# Huge integer-valued float counts (issue #345) always take the saddle form.
_logbinomial_like(value, n::Real, k::Real) = _logbinomial_like_saddle(value, n, k)

# log C(n, k) in the entropy/stirlerr form: k log(n/k) + (n-k) log(n/(n-k)) +
# log(n/(2 pi k (n-k)))/2 + stirlerr(n) - stirlerr(k) - stirlerr(n-k). No
# ~n log(n)-sized intermediate is differenced, so the absolute error stays
# ~eps * n * H(k/n) instead of ~eps * n * log(n). (The full binomial logpdf
# additionally needs the bd0 pairing with the p-terms -- the kernel handles
# that; this helper serves direct log-binomial-coefficient uses.)
function _logbinomial_like_saddle(value, n, k)
    zero_like = log(one(value))
    (k == 0 || k == n) && return zero_like
    nf = Float64(n)
    kf = Float64(k)
    mf = Float64(n - k)
    return oftype(
        zero_like,
        kf * log(nf / kf) + mf * log(nf / mf) +
        (log(nf) - log(2 * pi) - log(kf) - log(mf)) / 2 +
        _stirlerr(nf) - _stirlerr(kf) - _stirlerr(mf),
    )
end

logpdf(dist::BinomialDist, x) = _backend_binomial_logpdf(dist.trials, dist.p, x)

logpdf(dist::NegativeBinomialDist, x) = _backend_negativebinomial_logpdf(dist.successes, dist.p, x)

logpdf(dist::PoissonDist, x) = _backend_poisson_logpdf(dist.lambda, x)

# --- issue #231: betabinomial, multinomial, discreteuniform ------------------

# Log Beta function `log B(a, b) = loggamma(a) + loggamma(b) - loggamma(a+b)`.
# Reuses the O(1) `loggamma` discipline from #148 (never an O(count) loop) and
# stays ForwardDiff-Dual-friendly, so it serves both the CPU reference and the
# backend scorer with latent-flowing `a`/`b`.
_logbeta(a, b) = loggamma(a) + loggamma(b) - loggamma(a + b)

# Signed integer coercion for discreteuniform bounds/values: unlike
# `_poisson_count`, the support may include negative integers, so a value is
# accepted whenever it is integer-valued (Bool excluded upstream by callers).
function _discrete_integer(x)
    if x isa Integer
        return Int(x)
    elseif x isa Real && isfinite(x)
        truncated = trunc(x)
        return x == truncated ? Int(truncated) : nothing
    end
    return nothing
end

# Beta-Binomial: `k ~ betabinomial(n, alpha, beta)` is a binomial whose success
# probability is itself `Beta(alpha, beta)` distributed, the standard
# overdispersed-binomial likelihood. `n` is an integer trial count (data), while
# `alpha`/`beta` may flow from latents (their gradients are digamma differences).
struct BetaBinomialDist{T<:Real} <: AbstractTeaDistribution
    trials::Int
    alpha::T
    beta::T

    function BetaBinomialDist(trials::Int, alpha::T, beta::T) where {T<:Real}
        trials >= 0 || throw(ArgumentError("betabinomial requires trials >= 0"))
        alpha > zero(T) || throw(ArgumentError("betabinomial requires alpha > 0"))
        beta > zero(T) || throw(ArgumentError("betabinomial requires beta > 0"))
        new{T}(trials, alpha, beta)
    end
end

# Compositional count data: `x ~ multinomial(n, p)` over a length-K simplex `p`,
# the natural observation partner of the `dirichlet` prior. `p` flows from a
# latent (its per-component gradients are `x_i / p_i`).
struct MultinomialDist{T<:Real} <: AbstractTeaDistribution
    trials::Int
    probabilities::Vector{T}

    function MultinomialDist(trials::Int, probabilities::Vector{T}) where {T<:Real}
        trials >= 0 || throw(ArgumentError("multinomial requires trials >= 0"))
        isempty(probabilities) && throw(ArgumentError("multinomial requires at least one probability"))
        total = zero(T)
        for probability in probabilities
            zero(T) <= probability <= one(T) || throw(ArgumentError("multinomial requires 0 <= p <= 1"))
            total += probability
        end
        tolerance = sqrt(eps(float(total))) * max(length(probabilities), 1) * 8
        abs(total - one(total)) <= tolerance || throw(ArgumentError("multinomial probabilities must sum to 1"))
        new{T}(trials, probabilities)
    end
end

# Uniform over the integers `a, a+1, ..., b` (inclusive). Trivial density
# `-log(b - a + 1)`; important as an index/changepoint latent prior. No
# continuous parameters, so it contributes no gradient.
struct DiscreteUniformDist <: AbstractTeaDistribution
    a::Int
    b::Int

    function DiscreteUniformDist(a::Int, b::Int)
        a <= b || throw(ArgumentError("discreteuniform requires a <= b"))
        new(a, b)
    end
end

function betabinomial(trials, alpha, beta)
    count = _binomial_trials(trials)
    isnothing(count) && throw(ArgumentError("betabinomial requires integer trials >= 0"))
    promoted_alpha, promoted_beta = promote(float(alpha), float(beta))
    return BetaBinomialDist(count, promoted_alpha, promoted_beta)
end

function multinomial(trials, probabilities::AbstractVector)
    count = _binomial_trials(trials)
    isnothing(count) && throw(ArgumentError("multinomial requires integer trials >= 0"))
    return MultinomialDist(count, map(float, collect(probabilities)))
end

function discreteuniform(a, b)
    lower = _discrete_integer(a)
    upper = _discrete_integer(b)
    (isnothing(lower) || isnothing(upper)) && throw(ArgumentError("discreteuniform requires integer bounds"))
    return DiscreteUniformDist(lower, upper)
end

function Random.rand(rng::AbstractRNG, dist::BetaBinomialDist)
    probability = rand(rng, BetaDist(float(dist.alpha), float(dist.beta)))
    return rand(rng, BinomialDist(dist.trials, probability))
end

# Sequential conditional-binomial sampler: draw component k as
# Binomial(remaining trials, p_k / remaining probability mass).
function Random.rand(rng::AbstractRNG, dist::MultinomialDist)
    k = length(dist.probabilities)
    counts = zeros(Int, k)
    remaining_trials = dist.trials
    remaining_mass = one(float(eltype(dist.probabilities)))
    for index = 1:(k-1)
        remaining_trials == 0 && break
        conditional =
            remaining_mass > zero(remaining_mass) ?
            clamp(float(dist.probabilities[index]) / remaining_mass, 0.0, 1.0) : 0.0
        draw = rand(rng, BinomialDist(remaining_trials, conditional))
        counts[index] = draw
        remaining_trials -= draw
        remaining_mass -= float(dist.probabilities[index])
    end
    counts[k] = remaining_trials
    return counts
end

Random.rand(rng::AbstractRNG, dist::DiscreteUniformDist) = rand(rng, dist.a:dist.b)

# log-pmf core shared with the backend scorer: logC(n,k) + logB(k+a, n-k+b) -
# logB(a, b). `alpha` carries the arithmetic type so `_logbinomial_like`
# converts its (data) combinatorial term to the caller's element type.
function _betabinomial_logpdf_core(trials::Integer, alpha, beta, count::Integer)
    log_combination = _logbinomial_like(alpha, trials, count)
    return log_combination + _logbeta(count + alpha, trials - count + beta) - _logbeta(alpha, beta)
end

logpdf(dist::BetaBinomialDist, x) = _backend_betabinomial_logpdf(dist.trials, dist.alpha, dist.beta, x)

function logpdf(dist::MultinomialDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("multinomial logpdf expects a vector or tuple value"))
    carrier = float(first(dist.probabilities))
    length(values) == length(dist.probabilities) || return oftype(carrier, -Inf)
    accumulator = _logfactorial_like(carrier, dist.trials)
    total = 0
    for (value, probability) in zip(values, dist.probabilities)
        count = _poisson_count(value)
        isnothing(count) && return oftype(carrier, -Inf)
        total += count
        accumulator -= _logfactorial_like(carrier, count)
        if count > 0
            probability > zero(probability) || return oftype(carrier, -Inf)
            accumulator += count * log(probability)
        end
    end
    total == dist.trials || return oftype(carrier, -Inf)
    return accumulator
end

logpdf(dist::DiscreteUniformDist, x) = _backend_discreteuniform_logpdf(dist.a, dist.b, x)
