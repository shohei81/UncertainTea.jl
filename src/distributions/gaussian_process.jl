# Gaussian process regression (issue #262): a zero-mean GP marginal likelihood
# over N outputs, with an RBF (squared-exponential) kernel built from the inputs
# and latent hyperparameters. It is a VECTOR observation family scored through the
# multivariate-normal marginal `N(0, K + noise^2 I)`, where the kernel matrix `K`
# is rebuilt from the (Dual-valued during gradients) hyperparameters each call — so
# the hyperparameter gradient flows through the dense Cholesky by ForwardDiff.
#
# CPU-reference only, like the other dense-covariance families: honestly reported
# unsupported by `backend_report`/`device_report`. Users center their outputs
# (the prior mean is zero). The O(N^3) Cholesky is the exact GP cost.

# RBF covariance `K[i,j] = variance^2 * exp(-0.5 * sum_d (x_di - x_dj)^2 / l_d^2)`
# plus `noise^2` on the diagonal, with a tiny jitter for numerical positive-
# definiteness. `inputs` is a D x N matrix (one column per point). `lengthscale` is
# either a scalar (isotropic: one shared lengthscale) or a length-D vector
# (Automatic Relevance Determination: one lengthscale per input dimension, so an
# irrelevant dimension can be switched off by a large `l_d`). Generic in the
# element type so ForwardDiff Duals / Enzyme flow through.
_gp_inv_two_l2(lengthscale::Number, ::Int) = inv(2 * lengthscale^2)
_gp_inv_two_l2(lengthscale::AbstractVector, d::Int) = inv(2 * lengthscale[d]^2)
_gp_lengthscale_eltype(lengthscale::Number) = typeof(lengthscale)
_gp_lengthscale_eltype(lengthscale::AbstractVector) = eltype(lengthscale)

function _gp_rbf_covariance(inputs::AbstractMatrix, lengthscale, variance, noise)
    n = size(inputs, 2)
    T = promote_type(eltype(inputs), _gp_lengthscale_eltype(lengthscale), typeof(variance), typeof(noise))
    v2 = variance^2
    diag_add = noise^2 + T(1e-8)
    K = Matrix{T}(undef, n, n)
    @inbounds for j = 1:n
        for i = 1:n
            d2 = zero(T)
            for k in axes(inputs, 1)
                dk = inputs[k, i] - inputs[k, j]
                d2 += dk * dk * _gp_inv_two_l2(lengthscale, k)
            end
            K[i, j] = v2 * exp(-d2) + (i == j ? diag_add : zero(T))
        end
    end
    return K
end

struct GaussianProcessDist{X<:AbstractMatrix,L,T} <: AbstractTeaDistribution
    inputs::X            # D x N
    lengthscale::L       # scalar (isotropic) or length-D vector (ARD)
    variance::T
    noise::T

    function GaussianProcessDist(inputs::AbstractMatrix, lengthscale, variance, noise)
        size(inputs, 2) >= 1 || throw(ArgumentError("gaussianprocess requires at least one input point"))
        if lengthscale isa AbstractVector
            length(lengthscale) == size(inputs, 1) || throw(
                ArgumentError(
                    "gaussianprocess ARD lengthscale length $(length(lengthscale)) must match the " *
                    "input dimension $(size(inputs, 1))",
                ),
            )
        end
        v, nz = promote(variance, noise)
        return new{typeof(inputs),typeof(lengthscale),typeof(v)}(inputs, lengthscale, v, nz)
    end
end

"""
    gaussianprocess(inputs, lengthscale, variance, noise)

Zero-mean Gaussian process regression likelihood with a squared-exponential (RBF)
kernel. `inputs` is a `D x N` matrix (one column per point) or a length-`N` vector
for 1-D inputs; `variance` and `noise` are positive scalars (typically `exp` of
latent log-hyperparameters). `lengthscale` is either a positive **scalar**
(isotropic — one shared lengthscale) or a length-`D` **vector** (Automatic
Relevance Determination — one lengthscale per input dimension, so an uninformative
dimension is pruned by a large `l_d`). As an observation `{:y} ~ gaussianprocess(X,
l, v, nz)` scores the length-`N` output vector `y` under `N(0, K)` with `K[i,j] =
v^2 exp(-0.5 sum_d (x_di - x_dj)^2 / l_d^2) + nz^2 delta_ij`. CPU-reference only
(the dense Cholesky is not device-lowered). The `D + 2` hyperparameter gradient of
the ARD marginal likelihood is a natural `reverse_mode_gradient` target (issue
#268) once `D` is large.
"""
function gaussianprocess(inputs, lengthscale, variance, noise)
    matrix = inputs isa AbstractVector ? reshape(collect(float.(inputs)), 1, length(inputs)) : inputs
    return GaussianProcessDist(matrix, lengthscale, variance, noise)
end

"""
    gp_cholesky(inputs, lengthscale, variance, noise)

Lower-triangular Cholesky factor `L` of the RBF kernel matrix `K` (`K = L L'`)
built from `inputs` and the hyperparameters, for **direct latent-function
inference**: a plain function usable as a deterministic binding inside `@tea`,
mirroring `scale_cholesky`. Where `gaussianprocess` scores the analytic
zero-mean marginal `N(0, K)` (Gaussian likelihood only), `gp_cholesky` feeds the
GP prior into `mvnormaldense` so the latent function values `f ~ N(0, K)` are
sampled directly and can drive **any** likelihood — Bernoulli/logit
classification, Poisson counts, etc. `inputs` is a `D x N` matrix (or length-`N`
vector for 1-D); `lengthscale` is a scalar or length-`D` ARD vector; `noise` is
the diagonal jitter/nugget that keeps `K` positive definite.

```julia
@tea static function gp_classification(X)
    logl ~ normal(0.0, 1.0)
    logv ~ normal(0.0, 1.0)
    L = gp_cholesky(X, exp(logl), exp(logv), 1e-6)   # deterministic binding
    f ~ mvnormaldense((0.0, 0.0, 0.0, 0.0), L)       # latent GP values (static-length zero mean)
    for i in 1:4
        {:y => i} ~ bernoullilogit(f[i])
    end
    return logl
end
```
"""
function gp_cholesky(inputs, lengthscale, variance, noise)
    matrix = inputs isa AbstractVector ? reshape(collect(float.(inputs)), 1, length(inputs)) : inputs
    K = _gp_rbf_covariance(matrix, lengthscale, variance, noise)
    return cholesky(Symmetric(K)).L
end

function logpdf(gp::GaussianProcessDist, y::AbstractVector)
    n = size(gp.inputs, 2)
    length(y) == n ||
        throw(ArgumentError("gaussianprocess observation length $(length(y)) must match the $(n) inputs"))
    K = _gp_rbf_covariance(gp.inputs, gp.lengthscale, gp.variance, gp.noise)
    factorization = cholesky(Symmetric(K); check=false)
    issuccess(factorization) || return convert(float(eltype(K)), -Inf)
    alpha = factorization \ y
    logdet_half = sum(log, view(factorization.L, diagind(factorization.L)))
    return -sum(y .* alpha) / 2 - logdet_half - n * log(2 * oftype(logdet_half, pi)) / 2
end

function Random.rand(rng::AbstractRNG, gp::GaussianProcessDist)
    K = _gp_rbf_covariance(gp.inputs, gp.lengthscale, gp.variance, gp.noise)
    L = cholesky(Symmetric(K)).L
    return L * randn(rng, size(gp.inputs, 2))
end
