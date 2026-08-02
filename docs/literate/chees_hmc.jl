# # ChEES-HMC (adaptive trajectory length)
#
# ChEES-HMC (Hoffman, Radul & Sountsov, AISTATS 2021) is a fixed-length,
# Halton-**jittered** HMC whose trajectory length is adapted from *cross-chain*
# statistics during warmup instead of per-chain doubling. Every chain does
# identical work each step (no control-flow divergence, no per-leaf host sync), so
# it maps to a handful of kernels with one synchronization per iteration — the
# GPU-native route. This example runs it on the host; pass
# `backend=MetalBackend()` (with the `Metal` extension loaded) to run the same
# adaptation device-resident, reducing the cross-chain criterion on-device.
#
# The chains here are intentionally tiny so the docs build stays fast (this runs
# in CI on every push). Scale `num_chains`/`num_samples` up for real work.

using Random
using UncertainTea               # the @tea DSL, choicemap, distributions
using UncertainTea.Inference     # batched_chees
using UncertainTea.Diagnostics   # summarize, rhat, ess

# ## The model
#
# A two-parameter Gaussian location/scale model over a handful of observations —
# small enough to build quickly while still exercising the adaptation.

@tea static function chees_demo(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

# ## The data

rng = MersenneTwister(1)
y = 0.4 .+ 1.2 .* randn(rng, 40)
constraints = choicemap(((:y => i, y[i]) for i = 1:40)...)

# ## Sampling
#
# `batched_chees` runs an ensemble of chains together; the ensemble's endpoints
# drive the trajectory-length update. ChEES's optimal acceptance target is 0.651
# (versus NUTS's 0.8). `num_chains` should be moderately large so the cross-chain
# estimator is well-conditioned.

chains = batched_chees(
    chees_demo,
    (40,),
    constraints;
    num_chains=16,
    num_samples=300,
    num_warmup=300,
    target_accept=0.651,
    rng=MersenneTwister(2),
)

# ## Posterior summary
#
# `summarize` reports per-parameter means, quantiles, R-hat, and effective sample
# size, along with divergence/tree-depth diagnostics.

summary = summarize(chains)

# Convergence (R-hat close to 1) and effective sample size:

rhat(chains)

#-

ess(chains)
