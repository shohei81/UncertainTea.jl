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

# RBF covariance `K[i,j] = variance^2 * exp(-||x_i - x_j||^2 / (2 lengthscale^2))`
# plus `noise^2` on the diagonal, with a tiny jitter for numerical positive-
# definiteness. `inputs` is a D x N matrix (one column per point). Generic in the
# element type so ForwardDiff Duals flow through.
function _gp_rbf_covariance(inputs::AbstractMatrix, lengthscale, variance, noise)
    n = size(inputs, 2)
    T = promote_type(eltype(inputs), typeof(lengthscale), typeof(variance), typeof(noise))
    inv_two_l2 = one(T) / (2 * lengthscale^2)
    v2 = variance^2
    diag_add = noise^2 + T(1e-8)
    K = Matrix{T}(undef, n, n)
    @inbounds for j = 1:n
        for i = 1:n
            d2 = zero(T)
            for k in axes(inputs, 1)
                dk = inputs[k, i] - inputs[k, j]
                d2 += dk * dk
            end
            K[i, j] = v2 * exp(-d2 * inv_two_l2) + (i == j ? diag_add : zero(T))
        end
    end
    return K
end

struct GaussianProcessDist{X<:AbstractMatrix,T} <: AbstractTeaDistribution
    inputs::X            # D x N
    lengthscale::T
    variance::T
    noise::T

    function GaussianProcessDist(inputs::AbstractMatrix, lengthscale, variance, noise)
        size(inputs, 2) >= 1 || throw(ArgumentError("gaussianprocess requires at least one input point"))
        l, v, nz = promote(lengthscale, variance, noise)
        return new{typeof(inputs),typeof(l)}(inputs, l, v, nz)
    end
end

"""
    gaussianprocess(inputs, lengthscale, variance, noise)

Zero-mean Gaussian process regression likelihood with a squared-exponential (RBF)
kernel. `inputs` is a `D x N` matrix (one column per point) or a length-`N` vector
for 1-D inputs; `lengthscale`, `variance`, `noise` are positive scalars (typically
`exp` of latent log-hyperparameters). As an observation `{:y} ~
gaussianprocess(X, l, v, nz)` scores the length-`N` output vector `y` under
`N(0, K)` with `K[i,j] = v^2 exp(-||x_i - x_j||^2 / (2 l^2)) + nz^2 delta_ij`.
CPU-reference only (the dense Cholesky is not device-lowered).
"""
function gaussianprocess(inputs, lengthscale, variance, noise)
    matrix = inputs isa AbstractVector ? reshape(collect(float.(inputs)), 1, length(inputs)) : inputs
    return GaussianProcessDist(matrix, lengthscale, variance, noise)
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
