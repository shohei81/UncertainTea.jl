# Hidden Markov model with Gaussian emissions (issue #261). The observation
# SEQUENCE is scored through the forward algorithm, which marginalizes the K^T
# hidden-state paths in O(T*K^2) via the log-space alpha recursion — the same
# marginal Stan hand-writes with `log_sum_exp`. The K^T enumeration a
# per-timestep `marginalize=:enumerate` indicator would need is intractable, so
# the loop-carried Markov chain is exactly the case that needs the forward
# algorithm rather than site enumeration.
#
# First cut: fixed dynamics (`init`, `transition` are model arguments) with latent
# Gaussian emission means + a shared emission sd, so HMC/NUTS infers the emission
# parameters through the differentiable forward pass. CPU-reference only, like the
# other loop/marginal families. Latent transition/initial simplexes and
# non-Gaussian emissions are follow-ups.

struct HiddenMarkovDist{I<:AbstractVector,X<:AbstractMatrix,M<:AbstractVector,S} <: AbstractTeaDistribution
    init::I              # K initial-state probabilities
    transition::X        # K x K row-stochastic transition matrix
    means::M             # K Gaussian emission means (may be latent)
    sigma::S             # shared Gaussian emission sd (may be latent)

    function HiddenMarkovDist(init::AbstractVector, transition::AbstractMatrix, means::AbstractVector, sigma)
        k = length(init)
        k >= 1 || throw(ArgumentError("hmm requires at least one state"))
        size(transition) == (k, k) || throw(
            ArgumentError(
                "hmm transition must be $(k)x$(k) to match the $(k) initial-state probabilities, got $(size(transition))",
            ),
        )
        length(means) == k ||
            throw(ArgumentError("hmm requires one emission mean per state ($(k)), got $(length(means))"))
        return new{typeof(init),typeof(transition),typeof(means),typeof(sigma)}(init, transition, means, sigma)
    end
end

"""
    hmm(init, transition, means, sigma)

Hidden Markov model with Gaussian emissions. `init` is the length-`K` initial
state distribution and `transition` the `K x K` row-stochastic transition matrix
(fixed dynamics, model arguments); `means` are the `K` per-state emission means
and `sigma` the shared emission standard deviation (typically latent). As an
observation `{:y} ~ hmm(init, transition, means, sigma)` scores the length-`T`
observation sequence `y` through the forward algorithm, marginalizing the hidden
state path. CPU-reference only. Use an identifiable (e.g. ordered) `means`
parameterization so the marginal posterior is well-defined.
"""
hmm(init, transition, means, sigma) = HiddenMarkovDist(init, transition, means, sigma)

# Numerically stable log-sum-exp of a vector (generic in the element type so
# ForwardDiff Duals flow through).
function _hmm_logsumexp(values)
    m = maximum(values)
    isfinite(m) || return m
    return m + log(sum(v -> exp(v - m), values))
end

function logpdf(h::HiddenMarkovDist, y::AbstractVector)
    k = length(h.init)
    t_max = length(y)
    t_max >= 1 || throw(ArgumentError("hmm requires a non-empty observation sequence"))
    # No explicit accumulator type: the comprehensions infer it from the actual
    # (possibly ForwardDiff-Dual) `means`/`sigma` values, which stays correct even
    # when the latent-built `means` vector has an `Any` element type.
    log_transition = log.(h.transition)
    log_norm = -log(h.sigma) - oftype(float(h.sigma), 0.9189385332046727) # -log(sigma) - 0.5 log(2pi)
    emission(yt, state) = begin
        z = (yt - h.means[state]) / h.sigma
        log_norm - z * z / 2
    end
    # forward alpha recursion in log space
    log_alpha = [log(h.init[state]) + emission(y[1], state) for state = 1:k]
    for t = 2:t_max
        log_alpha = [
            _hmm_logsumexp([log_alpha[prev] + log_transition[prev, state] for prev = 1:k]) +
            emission(y[t], state) for state = 1:k
        ]
    end
    return _hmm_logsumexp(log_alpha)
end

function Random.rand(rng::AbstractRNG, h::HiddenMarkovDist)
    # a prior draw needs a sequence length; without one, emit an empty sequence
    # (the family is used as an observation, so `rand` is only a contract stub).
    return Float64[]
end
