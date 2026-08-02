# # Gaussian Process Regression
#
# A Gaussian process regression with latent kernel hyperparameters, using the
# composable kernel language from the [Modeling](../modeling.md) page. The
# Gaussian noise is integrated out analytically inside `gaussianprocess`, so
# the only latents are the (log) hyperparameters — three parameters regardless
# of how many data points there are.
#
# The data and chains here are intentionally tiny so the docs build stays fast
# (this runs in CI on every push). Scale everything up for real work.

using Random
using UncertainTea

# ## The data
#
# Noisy draws from a smooth function on `[0, 5]`. Inputs are a `d × n` matrix
# (one column per observation; here `d = 1`).

rng = MersenneTwister(11)
n = 24
xs = sort(rand(rng, n) .* 5)
X = reshape(xs, 1, n)
y = sin.(1.8 .* xs) .+ 0.25 .* randn(rng, n)

# ## The model
#
# A Matérn-5/2 kernel — the common default when the RBF is too smooth — with
# log-normal priors on the lengthscale, signal deviation, and noise. Because
# the hyperparameters flow through `exp`, they are unconstrained latents and
# gradient-based samplers apply directly.

@tea static function gp_reg(X)
    log_l ~ normal(0.0, 1.0)
    log_v ~ normal(0.0, 1.0)
    log_n ~ normal(-1.0, 1.0)
    {:y} ~ gaussianprocess(X, matern52_kernel(exp(log_l), exp(log_v)), exp(log_n))
    return log_l
end

constraints = choicemap((:y, y))

# ## Sampling

chains = nuts_chains(
    gp_reg,
    (X,),
    constraints;
    num_chains=2,
    num_samples=200,
    num_warmup=200,
    rng=MersenneTwister(12),
)

# ## Posterior summary
#
# The posterior lengthscale should sit near the true wiggliness of the data
# (`sin(1.8x)` has a lengthscale of order one) and the noise near `0.25`.

summarize(chains)

#-

rhat(chains)

# ## Where to go from here
#
# - Swap the kernel: `kernel_product(periodic_kernel(l, v, p), rbf_kernel(L, 1.0))`
#   gives the locally-periodic composition for seasonal data, and a vector
#   lengthscale gives ARD over input dimensions.
# - For large `n`, `sparsegaussianprocess(X, Z, kernel, noise)` is the FITC
#   approximation with inducing inputs `Z`.
# - For non-Gaussian likelihoods (classification, counts) the function values
#   themselves become latent — see the latent-GP section of the
#   [Modeling](../modeling.md) page and `elliptical_slice`.
