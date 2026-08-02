# # Poisson GLM (broadcast observations)
#
# A Poisson regression written with the vectorized observation syntax: the
# whole count vector is one addressed choice, `{:y} ~ poisson.(exp.(...))`.
# The broadcast form lowers to a single dense plan step with an analytic
# batched gradient — the fast path for GLM-style models — instead of a
# per-element loop.
#
# The data and chains here are intentionally tiny so the docs build stays fast
# (this runs in CI on every push). Scale everything up for real work.

using Random
using UncertainTea

# ## The model
#
# An intercept and a slope on the log-rate scale (the canonical log link).
# Dotted function calls (`exp.(...)`) are supported inside broadcast
# observation arguments.

@tea static function poisson_glm(x, n)
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ poisson.(exp.(a .+ b .* x))
    return a
end

# ## The data
#
# Counts drawn from known coefficients. Integer observation vectors are
# accepted directly — no `Float64.(...)` conversion needed for count data.

rng = MersenneTwister(21)
n = 60
x = collect(range(-1.5, 1.5; length=n))
a_true, b_true = 0.4, 0.9
y = [rand(rng, UncertainTea.PoissonDist(exp(a_true + b_true * xi))) for xi in x]
constraints = choicemap((:y, y))

# ## Prior predictive check
#
# Before conditioning, check that the prior generates counts on a sensible
# scale. `prior_predictive` samples the full joint from the prior and keeps
# the observation addresses (the constraint values are unused — they only fix
# the observed/latent split).

prior_draws = prior_predictive(poisson_glm, (x, n), constraints; num_draws=50, rng=MersenneTwister(22))

# ## Sampling
#
# `batched_nuts` advances all chains together over a dense layout; the
# broadcast likelihood uses the analytic gradient tier.

chains = batched_nuts(
    poisson_glm,
    (x, n),
    constraints;
    num_chains=4,
    num_samples=300,
    num_warmup=300,
    rng=MersenneTwister(23),
)

# ## Posterior summary
#
# The posterior means should sit near the generating `a = 0.4`, `b = 0.9`.

summarize(chains)

#-

rhat(chains)

# ## Where to go from here
#
# - The same spelling works for `bernoulli`, `bernoullilogit` (logistic
#   regression without manual clamping), `exponential`, `studentt`, and
#   `normal` likelihoods — see the
#   [Modeling](../modeling.md) page.
# - `predict` and `psis_loo`/`loo` take the same model/constraints for
#   posterior-predictive checks and model comparison.
