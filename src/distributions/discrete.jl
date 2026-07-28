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
function bernoulli(p)
    return BernoulliDist(float(p))
end

function geometric(p)
    return GeometricDist(float(p))
end

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

function negativebinomial(successes, p)
    promoted_successes, promoted_probability = promote(float(successes), float(p))
    return NegativeBinomialDist(promoted_successes, promoted_probability)
end

function categorical(probabilities::AbstractVector)
    promoted = map(float, collect(probabilities))
    return CategoricalDist(promoted)
end

function categorical(probabilities::Vararg{Real})
    promoted = collect(promote(map(float, probabilities)...))
    return CategoricalDist(promoted)
end

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

function logpdf(dist::BernoulliDist, x)
    value = _bernoulli_value(x)
    isnothing(value) && return oftype(float(dist.p), -Inf)
    return value ? log(dist.p) : log1p(-dist.p)
end

# Support handling mirrors `bernoulli` (Bool or a numeric 0/1); the density is
# the stable log-scale form `x*eta - log1p(exp(eta))`.
function logpdf(dist::BernoulliLogitDist, x)
    value = _bernoulli_value(x)
    isnothing(value) && return oftype(float(dist.eta), -Inf)
    eta = dist.eta
    log_normalizer = _bernoullilogit_log1p_exp(eta)
    return value ? eta - log_normalizer : -log_normalizer
end

function logpdf(dist::GeometricDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.p), -Inf)
    if count == 0
        return log(dist.p)
    elseif dist.p == one(dist.p)
        return oftype(float(dist.p), -Inf)
    end
    return log(dist.p) + count * log1p(-dist.p)
end

function _poisson_count(x)
    if x isa Integer
        return x >= 0 ? Int(x) : nothing
    elseif x isa Real && isfinite(x)
        truncated = trunc(x)
        return x >= zero(x) && x == truncated ? Int(truncated) : nothing
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

function _logbinomial_like(value, n::Integer, k::Integer)
    return _logfactorial_like(value, n) -
           _logfactorial_like(value, k) -
           _logfactorial_like(value, n - k)
end

function logpdf(dist::BinomialDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.p), -Inf)
    count <= dist.trials || return oftype(float(dist.p), -Inf)
    probability = dist.p
    log_combination = _logbinomial_like(probability, dist.trials, count)
    if count == 0 && count == dist.trials
        return log_combination
    elseif count == 0
        return log_combination + dist.trials * log1p(-probability)
    elseif count == dist.trials
        return log_combination + count * log(probability)
    end
    return log_combination +
           count * log(probability) +
           (dist.trials - count) * log1p(-probability)
end

function logpdf(dist::NegativeBinomialDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.p), -Inf)
    successes, probability = promote(dist.successes, dist.p)
    if count == 0 && probability == one(probability)
        return zero(probability)
    elseif probability == one(probability)
        return oftype(probability, -Inf)
    end
    return loggamma(count + successes) - loggamma(successes) - _logfactorial_like(probability, count) +
           successes * log(probability) + count * log1p(-probability)
end

function logpdf(dist::PoissonDist, x)
    count = _poisson_count(x)
    isnothing(count) && return oftype(float(dist.lambda), -Inf)
    lambda = dist.lambda
    return count * log(lambda) - lambda - _logfactorial_like(lambda, count)
end
