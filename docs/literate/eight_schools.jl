# # Eight Schools (Non-centered)
#
# The eight-schools model (Rubin, 1981) is the canonical hierarchical-model
# stress test: a funnel-shaped posterior that trips up naive samplers. This
# example builds it in UncertainTea's static DSL using the automatic
# **non-centered reparameterization** (`iid(...; reparam=:noncentered)`), samples
# it with the batched NUTS engine, and prints real posterior summaries.
#
# The chains here are intentionally tiny so the docs build stays fast (this runs
# in CI on every push). Scale `num_chains`/`num_samples` up for real work.

using Random
using UncertainTea               # the @tea DSL, choicemap, distributions
using UncertainTea.Inference     # batched_nuts
using UncertainTea.Diagnostics   # summarize, rhat, ess

# ## The model
#
# Each school reports an observed treatment effect `y[i]` with known standard
# error `sigma[i]`. School effects `theta` are drawn hierarchically around a
# shared mean `mu` with a log-scale spread `tau`; the `reparam=:noncentered`
# option samples standardized offsets under the hood while keeping the `theta`
# addresses intact.

@tea static function eight_schools(sigma)
    mu ~ normal(0.0, 5.0)
    log_tau ~ normal(0.0, 1.5)
    tau = 0.001 + exp(log_tau)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

# ## The data
#
# The classic eight-schools observations and their standard errors. Observations
# enter the model as a `choicemap` over the `:y => i` addresses.

y = Float64[28, 8, -3, 7, -1, 1, 18, 12]
sigma = Float64[15, 10, 16, 11, 9, 11, 10, 18]
constraints = choicemap(((:y => i, y[i]) for i = 1:8)...)

# ## Sampling
#
# `batched_nuts` runs all chains together over a dense layout. On CPU it uses the
# reference host path; pass `backend=MetalBackend()` (with the `Metal` extension
# loaded) to run the device-resident path instead.

chains = batched_nuts(
    eight_schools,
    (sigma,),
    constraints;
    num_chains=4,
    num_samples=250,
    num_warmup=250,
    rng=MersenneTwister(1),
)

# ## Posterior summary
#
# `summarize` reports per-parameter means, quantiles, R-hat, and effective sample
# size, along with divergence/tree-depth diagnostics.

summary = summarize(chains)

# Population-level parameters, pulled out individually:

rhat(chains)

#-

ess(chains)

# ## Forest plot
#
# The hierarchical shrinkage at a glance: each school's posterior median and
# central 80% interval, hand-rolled from `posterior_array` with scatter plus
# error bars (no extra plotting-recipe dependency needed). The dashed line is
# the posterior median of the population mean `mu` — every school is pulled
# toward it, the noisier ones more strongly.

using Statistics
using Plots

draws = posterior_array(chains)                 # (samples, chains, params)
pnames = parameter_names(chains)
school(i) = vec(draws[:, :, findfirst(==("theta[$i]"), pnames)])
med = [median(school(i)) for i = 1:8]
q10 = [quantile(school(i), 0.10) for i = 1:8]
q90 = [quantile(school(i), 0.90) for i = 1:8]

scatter(med, 1:8; xerror=(med .- q10, q90 .- med), color=:steelblue,
    yticks=(1:8, ["school $i" for i = 1:8]), yflip=true,
    label="posterior median, 80% interval", legend=:bottomright,
    xlabel="treatment effect", size=(700, 400))
vline!([median(vec(draws[:, :, findfirst(==("mu"), pnames)]))];
    color=:gray, linestyle=:dash, label="population mean mu")
