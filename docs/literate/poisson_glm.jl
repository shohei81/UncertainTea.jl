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
using UncertainTea               # the @tea DSL, choicemap, distributions
using UncertainTea.Inference     # batched_nuts
using UncertainTea.Diagnostics   # prior_predictive, summarize, rhat

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

# ## Posterior fit
#
# The observed counts against the posterior-mean rate curve `exp(a + b*x)`,
# with a pointwise 90% band computed directly from the posterior draws via
# `posterior_array`.

using Statistics
using Plots

draws = posterior_array(chains)                 # (samples, chains, params)
pnames = parameter_names(chains)
a_draws = vec(draws[:, :, findfirst(==("a"), pnames)])
b_draws = vec(draws[:, :, findfirst(==("b"), pnames)])

xgrid = range(-1.5, 1.5; length=80)
rate = exp.(a_draws .+ b_draws .* xgrid')       # draws x grid
mid = vec(mean(rate; dims=1))
lo = [quantile(col, 0.05) for col in eachcol(rate)]
hi = [quantile(col, 0.95) for col in eachcol(rate)]

plot(xgrid, mid; ribbon=(mid .- lo, hi .- mid), color=:darkorange, fillalpha=0.25,
    label="posterior mean rate (90% band)", xlabel="x", ylabel="count", size=(700, 400))
scatter!(x, y; color=:black, markersize=3, label="observed counts")

# ## Trace and density plots via MCMCChains
#
# [`UncertainTea.to_mcmcchains`](@ref) converts the result to an `MCMCChains.Chains`, which
# carries StatsPlots recipes for trace, density, autocorrelation, and corner
# plots. Load `MCMCChains` with `import` rather than `using`: it also exports
# `summarize`/`ess`/`rhat`, which would clash with the
# `UncertainTea.Diagnostics` versions used above.

import MCMCChains
using StatsPlots

mc = to_mcmcchains(chains)
plot(mc; size=(700, 400))

# ## Where to go from here
#
# - The same spelling works for `bernoulli`, `bernoullilogit` (logistic
#   regression without manual clamping), `exponential`, `studentt`, and
#   `normal` likelihoods — see the
#   [Modeling](../modeling.md) page.
# - `predict` and `psis_loo`/`loo` take the same model/constraints for
#   posterior-predictive checks and model comparison.
