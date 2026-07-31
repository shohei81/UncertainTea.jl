# # Logistic Regression (fused GLM path)
#
# Logistic regression is the workhorse GLM. UncertainTea recognizes the fused
# linear predictor `alpha + sum(beta .* X[:, i])` inside a `bernoullilogit`
# observation and lowers it to a hand-derived **analytic gradient** on both the
# host batched path and the KernelAbstractions device backend — the same closed
# form Stan and NumPyro use — instead of falling back to autodiff. This example
# builds a small logistic model and samples it with batched NUTS.
#
# The chains and data here are intentionally tiny so the docs build stays fast
# (this runs in CI on every push). Scale everything up for real work.

using Random
using UncertainTea

# ## The model
#
# An intercept `alpha` plus an i.i.d.-normal coefficient prior on `beta` (a
# diagonal `mvnormal`), with a logit-Bernoulli likelihood over each observation's
# covariate column `X[:, i]`.

@tea static function logistic(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ mvnormal((0.0, 0.0, 0.0), (2.5, 2.5, 2.5))
    for i = 1:n
        {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
    end
    return alpha
end

# ## The data
#
# A synthetic design matrix `X` (3 covariates × N observations, column per
# observation) and 0/1 outcomes drawn from a known coefficient vector.

rng = MersenneTwister(1)
N = 80
X = randn(rng, 3, N)
alpha_true, beta_true = 0.3, [0.8, -0.5, 1.1]
p = 1 ./ (1 .+ exp.(-(alpha_true .+ vec(sum(beta_true .* X; dims=1)))))
y = Float64.(rand(rng, N) .< p)
constraints = choicemap(((:y => i, y[i]) for i = 1:N)...)

# ## Sampling
#
# `batched_nuts` samples the model over a dense many-chain layout. Because the
# GLM lowers to the analytic path, the gradient is exact and cheap. Pass
# `backend=MetalBackend()` to run the device-resident path.

chains = batched_nuts(
    logistic,
    (X, N),
    constraints;
    num_chains=4,
    num_samples=300,
    num_warmup=300,
    rng=MersenneTwister(2),
)

# ## Posterior summary
#
# The posterior means should sit near the true `alpha`/`beta` used to generate
# the data.

summary = summarize(chains)

#-

rhat(chains)
