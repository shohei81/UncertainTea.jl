# Sparse Gaussian process regression via the FITC (Fully Independent Training
# Conditional) approximation (issue #281). The dense GP marginal `N(0, K_NN +
# noise^2 I)` costs an O(N^3) Cholesky; FITC replaces `K_NN` with a rank-`M`
# Nystrom approximation `Q_NN = K_NM K_MM^-1 K_MN` built from `M << N` inducing
# points, plus the exact diagonal correction `diag(K_NN - Q_NN)` that keeps the
# variances calibrated. The marginal becomes
#
#     y ~ N(0, Lambda + K_NM K_MM^-1 K_MN),   Lambda = diag(K_NN - Q_NN) + noise^2 I
#
# and the Woodbury identity evaluates its log density in O(N M^2 + M^3) instead of
# O(N^3). With the inducing points placed at the data (`inducing = inputs`) FITC
# reduces EXACTLY to the dense GP marginal, since then `Q_NN = K_NN` and the
# diagonal correction is zero -- a useful correctness anchor.
#
# CPU-reference only, like the dense `gaussianprocess`. The kernel matrices are
# rebuilt from the (Dual-valued during gradients) hyperparameters each call, so
# the hyperparameter gradient flows through by ForwardDiff / Enzyme.

# Cross RBF covariance `v^2 exp(-0.5 sum_d (a_di - b_dj)^2 / l_d^2)` between two
# point sets `a` (D x P) and `b` (D x Q); no noise, no jitter (a pure kernel
# block). `lengthscale` is a scalar (isotropic) or length-D vector (ARD).
function _gp_rbf_cross(a::AbstractMatrix, b::AbstractMatrix, lengthscale, variance)
    p = size(a, 2)
    q = size(b, 2)
    T = promote_type(eltype(a), eltype(b), _gp_lengthscale_eltype(lengthscale), typeof(variance))
    v2 = variance^2
    K = Matrix{T}(undef, p, q)
    @inbounds for j = 1:q
        for i = 1:p
            d2 = zero(T)
            for k in axes(a, 1)
                dk = a[k, i] - b[k, j]
                d2 += dk * dk * _gp_inv_two_l2(lengthscale, k)
            end
            K[i, j] = v2 * exp(-d2)
        end
    end
    return K
end

struct SparseGaussianProcessDist{X<:AbstractMatrix,Z<:AbstractMatrix,K<:AbstractGPKernel,T} <: AbstractTeaDistribution
    inputs::X            # D x N
    inducing::Z          # D x M
    kernel::K
    noise::T

    function SparseGaussianProcessDist(inputs::AbstractMatrix, inducing::AbstractMatrix, kernel::AbstractGPKernel, noise)
        size(inputs, 2) >= 1 || throw(ArgumentError("sparsegaussianprocess requires at least one input point"))
        size(inducing, 2) >= 1 || throw(ArgumentError("sparsegaussianprocess requires at least one inducing point"))
        size(inputs, 1) == size(inducing, 1) || throw(
            ArgumentError(
                "sparsegaussianprocess inputs and inducing points must share the input dimension, got " *
                "$(size(inputs, 1)) and $(size(inducing, 1))",
            ),
        )
        _gp_check_kernel_ard("sparsegaussianprocess", kernel, inputs)
        return new{typeof(inputs),typeof(inducing),typeof(kernel),typeof(noise)}(
            inputs, inducing, kernel, noise,
        )
    end
end

"""
    sparsegaussianprocess(inputs, inducing, lengthscale, variance, noise)

Sparse zero-mean Gaussian process regression likelihood via the FITC
approximation. `inputs` is a `D x N` matrix (or length-`N` vector for 1-D) and
`inducing` a `D x M` matrix of `M << N` inducing points (or length-`M` vector);
`lengthscale` is a positive scalar or length-`D` ARD vector, and `variance` /
`noise` are positive scalars. As an observation `{:y} ~ sparsegaussianprocess(X,
Z, l, v, nz)` scores the length-`N` output vector under `N(0, Lambda + K_NM
K_MM^-1 K_MN)` with `Lambda = diag(K_NN - Q_NN) + nz^2 I`, evaluated in
`O(N M^2 + M^3)` by the Woodbury identity instead of the dense GP's `O(N^3)`.
With `inducing = inputs` it reduces exactly to `gaussianprocess`. CPU-reference
only (not device-lowered). Expects centred outputs (the prior mean is zero).
"""
function sparsegaussianprocess(inputs, inducing, lengthscale, variance, noise)
    return sparsegaussianprocess(inputs, inducing, RBFKernel(lengthscale, variance), noise)
end

# kernel-spec form (issue #290)
function sparsegaussianprocess(inputs, inducing, kernel::AbstractGPKernel, noise)
    input_matrix = inputs isa AbstractVector ? reshape(collect(float.(inputs)), 1, length(inputs)) : inputs
    inducing_matrix = inducing isa AbstractVector ? reshape(collect(float.(inducing)), 1, length(inducing)) : inducing
    return SparseGaussianProcessDist(input_matrix, inducing_matrix, kernel, noise)
end

# FITC log marginal likelihood via Woodbury (issue #281). Returns -Inf when a
# Cholesky is not positive definite (an invalid hyperparameter draw), matching the
# dense `gaussianprocess` reject behavior.
function logpdf(sgp::SparseGaussianProcessDist, y::AbstractVector)
    n = size(sgp.inputs, 2)
    m = size(sgp.inducing, 2)
    length(y) == n || throw(ArgumentError(
        "sparsegaussianprocess observation length $(length(y)) must match the $(n) inputs",
    ))
    kern, nz = sgp.kernel, sgp.noise
    T = promote_type(eltype(sgp.inputs), _gp_kernel_eltype(kern), typeof(nz), eltype(y))

    Kmm = _gp_kernel_cross(kern, sgp.inducing, sgp.inducing)
    @inbounds for i = 1:m
        Kmm[i, i] += T(1e-6)                                   # jitter for a PD K_MM
    end
    Kmn = _gp_kernel_cross(kern, sgp.inducing, sgp.inputs)     # M x N

    chol_mm = cholesky(Symmetric(Kmm); check=false)
    issuccess(chol_mm) || return convert(float(T), -Inf)
    A = chol_mm.L \ Kmn                                        # M x N, Q_NN = A' A

    knn_diag = _gp_kernel_self(kern)                           # stationary self-covariance
    lambda = Vector{T}(undef, n)
    @inbounds for i = 1:n
        qii = zero(T)
        for r = 1:m
            qii += A[r, i]^2
        end
        lambda[i] = knn_diag - qii + nz^2 + T(1e-8)            # FITC diagonal + noise
    end
    any(<=(zero(T)), lambda) && return convert(float(T), -Inf)

    # B = I_M + A diag(1/lambda) A'   (M x M)
    A_over_lambda = similar(A)
    @inbounds for i = 1:n
        inv_li = inv(lambda[i])
        for r = 1:m
            A_over_lambda[r, i] = A[r, i] * inv_li
        end
    end
    B = A_over_lambda * transpose(A)
    @inbounds for r = 1:m
        B[r, r] += one(T)
    end
    chol_b = cholesky(Symmetric(B); check=false)
    issuccess(chol_b) || return convert(float(T), -Inf)

    y_over_lambda = y ./ lambda
    Aiy = A * y_over_lambda                                    # M vector
    c = chol_b.L \ Aiy
    quad = sum(y .* y_over_lambda) - sum(abs2, c)              # y' C^-1 y
    logdet_c = sum(log, lambda) + 2 * sum(log, view(chol_b.L, diagind(chol_b.L)))
    return -(quad + logdet_c + n * log(2 * oftype(quad, pi))) / 2
end

function Random.rand(rng::AbstractRNG, sgp::SparseGaussianProcessDist)
    n = size(sgp.inputs, 2)
    m = size(sgp.inducing, 2)
    kern, nz = sgp.kernel, sgp.noise
    Kmm = _gp_kernel_cross(kern, sgp.inducing, sgp.inducing)
    @inbounds for i = 1:m
        Kmm[i, i] += 1e-6
    end
    Kmn = _gp_kernel_cross(kern, sgp.inducing, sgp.inputs)
    A = cholesky(Symmetric(Kmm)).L \ Kmn
    lambda = [_gp_kernel_self(kern) - sum(abs2, view(A, :, i)) + nz^2 + 1e-8 for i = 1:n]
    # sample from N(0, Q_NN + diag(lambda)) = A' u + sqrt(lambda) .* eps
    return transpose(A) * randn(rng, m) .+ sqrt.(max.(lambda, 0.0)) .* randn(rng, n)
end
