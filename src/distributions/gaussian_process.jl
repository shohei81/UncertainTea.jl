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

# --- kernel specifications (issue #290) ---------------------------------------
#
# A GP kernel is a lightweight spec struct with two elementwise operations:
# `_gp_kernel_element(k, a, i, b, j)` (the covariance of columns a[:, i] and
# b[:, j]) and `_gp_kernel_self(k)` (the self-covariance k(x, x), constant for
# the stationary kernels here; used by the FITC diagonal correction). All
# hyperparameters may be latent-flowing (ForwardDiff / Enzyme generic), and
# `lengthscale` is a scalar (isotropic) or length-D ARD vector wherever it
# scales a distance. `kernel_sum` / `kernel_product` compose kernels.

abstract type AbstractGPKernel end

struct RBFKernel{L,T} <: AbstractGPKernel
    lengthscale::L
    variance::T
end

struct Matern32Kernel{L,T} <: AbstractGPKernel
    lengthscale::L
    variance::T
end

struct Matern52Kernel{L,T} <: AbstractGPKernel
    lengthscale::L
    variance::T
end

# The standard exp-sine-squared periodic kernel over the isotropic input
# distance; `period` is the repeat length.
struct PeriodicKernel{L,T,P} <: AbstractGPKernel
    lengthscale::L
    variance::T
    period::P
end

struct SumKernel{A<:AbstractGPKernel,B<:AbstractGPKernel} <: AbstractGPKernel
    a::A
    b::B
end

struct ProductKernel{A<:AbstractGPKernel,B<:AbstractGPKernel} <: AbstractGPKernel
    a::A
    b::B
end

"""
    rbf_kernel(lengthscale, variance)
    matern32_kernel(lengthscale, variance)
    matern52_kernel(lengthscale, variance)
    periodic_kernel(lengthscale, variance, period)
    kernel_sum(a, b), kernel_product(a, b)

GP kernel constructors for `gaussianprocess(inputs, kernel, noise)`,
`sparsegaussianprocess(inputs, inducing, kernel, noise)`, and
`gp_cholesky(inputs, kernel, noise)` (issue #290). `lengthscale` is a positive
scalar or a length-`D` ARD vector; `variance` the signal standard deviation;
`period` the periodic repeat length. Matérn 3/2 and 5/2 give once/twice
mean-square-differentiable sample paths (the common defaults when RBF is too
smooth); the periodic kernel is the standard exp-sine-squared form over the
isotropic input distance. `kernel_sum` / `kernel_product` compose kernels
(e.g. a locally-periodic `kernel_product(periodic_kernel(...), rbf_kernel(...))`).
"""
rbf_kernel(lengthscale, variance) = RBFKernel(lengthscale, variance)
matern32_kernel(lengthscale, variance) = Matern32Kernel(lengthscale, variance)
matern52_kernel(lengthscale, variance) = Matern52Kernel(lengthscale, variance)
periodic_kernel(lengthscale, variance, period) = PeriodicKernel(lengthscale, variance, period)
kernel_sum(a::AbstractGPKernel, b::AbstractGPKernel) = SumKernel(a, b)
kernel_product(a::AbstractGPKernel, b::AbstractGPKernel) = ProductKernel(a, b)

# The shared docstring above only binds to the first definition; attach it to
# the rest of the kernel constructors so `@ref` links and `?` lookups resolve.
@doc (@doc rbf_kernel) matern32_kernel
@doc (@doc rbf_kernel) matern52_kernel
@doc (@doc rbf_kernel) periodic_kernel
@doc (@doc rbf_kernel) kernel_sum
@doc (@doc rbf_kernel) kernel_product

# lengthscale-scaled squared distance sum_d ((a_d - b_d) / l_d)^2
@inline function _gp_scaled_sqdist(lengthscale, a, i, b, j)
    d2 = zero(promote_type(eltype(a), eltype(b), _gp_lengthscale_eltype(lengthscale)))
    @inbounds for k in axes(a, 1)
        dk = a[k, i] - b[k, j]
        d2 += dk * dk * (2 * _gp_inv_two_l2(lengthscale, k))
    end
    return d2
end

# unscaled squared distance sum_d (a_d - b_d)^2
@inline function _gp_sqdist(a, i, b, j)
    d2 = zero(promote_type(eltype(a), eltype(b)))
    @inbounds for k in axes(a, 1)
        dk = a[k, i] - b[k, j]
        d2 += dk * dk
    end
    return d2
end

@inline _gp_kernel_element(k::RBFKernel, a, i, b, j) =
    k.variance^2 * exp(-_gp_scaled_sqdist(k.lengthscale, a, i, b, j) / 2)

@inline function _gp_kernel_element(k::Matern32Kernel, a, i, b, j)
    r2 = _gp_scaled_sqdist(k.lengthscale, a, i, b, j)
    # guard r == 0 (the diagonal): sqrt has an infinite derivative at 0, so a
    # dual-valued lengthscale would produce a NaN gradient channel there; the
    # self-covariance is exactly variance^2
    r2 == 0 && return k.variance^2
    r = sqrt(r2)
    s = sqrt(oftype(r, 3)) * r
    return k.variance^2 * (1 + s) * exp(-s)
end

@inline function _gp_kernel_element(k::Matern52Kernel, a, i, b, j)
    r2 = _gp_scaled_sqdist(k.lengthscale, a, i, b, j)
    r2 == 0 && return k.variance^2      # see the Matern-3/2 diagonal guard
    r = sqrt(r2)
    s = sqrt(oftype(r, 5)) * r
    return k.variance^2 * (1 + s + 5 * r2 / 3) * exp(-s)
end

@inline function _gp_kernel_element(k::PeriodicKernel, a, i, b, j)
    r = sqrt(_gp_sqdist(a, i, b, j))
    sine = sin(oftype(r, pi) * r / k.period)
    return k.variance^2 * exp(-2 * sine * sine / (k.lengthscale^2))
end

@inline _gp_kernel_element(k::SumKernel, a, i, b, j) =
    _gp_kernel_element(k.a, a, i, b, j) + _gp_kernel_element(k.b, a, i, b, j)
@inline _gp_kernel_element(k::ProductKernel, a, i, b, j) =
    _gp_kernel_element(k.a, a, i, b, j) * _gp_kernel_element(k.b, a, i, b, j)

# self-covariance k(x, x): variance^2 for the stationary kernels here
@inline _gp_kernel_self(k::Union{RBFKernel,Matern32Kernel,Matern52Kernel,PeriodicKernel}) = k.variance^2
@inline _gp_kernel_self(k::SumKernel) = _gp_kernel_self(k.a) + _gp_kernel_self(k.b)
@inline _gp_kernel_self(k::ProductKernel) = _gp_kernel_self(k.a) * _gp_kernel_self(k.b)

# element type carried by a kernel's hyperparameters (for buffer allocation)
_gp_kernel_eltype(k::Union{RBFKernel,Matern32Kernel,Matern52Kernel}) =
    promote_type(_gp_lengthscale_eltype(k.lengthscale), typeof(k.variance))
_gp_kernel_eltype(k::PeriodicKernel) =
    promote_type(_gp_lengthscale_eltype(k.lengthscale), typeof(k.variance), typeof(k.period))
_gp_kernel_eltype(k::Union{SumKernel,ProductKernel}) =
    promote_type(_gp_kernel_eltype(k.a), _gp_kernel_eltype(k.b))

# cross-covariance block K[i, j] = k(a[:, i], b[:, j]) with no noise/jitter
function _gp_kernel_cross(k::AbstractGPKernel, a::AbstractMatrix, b::AbstractMatrix)
    T = promote_type(eltype(a), eltype(b), _gp_kernel_eltype(k))
    K = Matrix{T}(undef, size(a, 2), size(b, 2))
    @inbounds for j in axes(b, 2)
        for i in axes(a, 2)
            K[i, j] = _gp_kernel_element(k, a, i, b, j)
        end
    end
    return K
end

# full covariance K = k(X, X) + (noise^2 + jitter) I
function _gp_kernel_covariance(k::AbstractGPKernel, inputs::AbstractMatrix, noise)
    K = _gp_kernel_cross(k, inputs, inputs)
    T = eltype(K)
    diag_add = noise^2 + T(1e-8)
    @inbounds for i in axes(K, 1)
        K[i, i] += diag_add
    end
    return K
end

# retained as a thin RBF wrapper (issue #290 moved the general machinery to the
# kernel-spec layer above)
_gp_rbf_covariance_kernel(inputs, lengthscale, variance, noise) =
    _gp_kernel_covariance(RBFKernel(lengthscale, variance), inputs, noise)

# ARD lengthscale sanity against the input dimension (shared by every kernel
# that scales a distance per dimension)
function _gp_check_ard_lengthscale(family::String, lengthscale, inputs::AbstractMatrix)
    lengthscale isa AbstractVector || return nothing
    length(lengthscale) == size(inputs, 1) || throw(
        ArgumentError(
            "$(family) ARD lengthscale length $(length(lengthscale)) must match the " *
            "input dimension $(size(inputs, 1))",
        ),
    )
    return nothing
end

_gp_check_kernel_ard(family::String, k::Union{RBFKernel,Matern32Kernel,Matern52Kernel}, inputs) =
    _gp_check_ard_lengthscale(family, k.lengthscale, inputs)
_gp_check_kernel_ard(::String, ::PeriodicKernel, inputs) = nothing
function _gp_check_kernel_ard(family::String, k::Union{SumKernel,ProductKernel}, inputs)
    _gp_check_kernel_ard(family, k.a, inputs)
    _gp_check_kernel_ard(family, k.b, inputs)
    return nothing
end

struct GaussianProcessDist{X<:AbstractMatrix,K<:AbstractGPKernel,T} <: AbstractTeaDistribution
    inputs::X            # D x N
    kernel::K
    noise::T

    function GaussianProcessDist(inputs::AbstractMatrix, kernel::AbstractGPKernel, noise)
        size(inputs, 2) >= 1 || throw(ArgumentError("gaussianprocess requires at least one input point"))
        _gp_check_kernel_ard("gaussianprocess", kernel, inputs)
        return new{typeof(inputs),typeof(kernel),typeof(noise)}(inputs, kernel, noise)
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
    return GaussianProcessDist(matrix, RBFKernel(lengthscale, variance), noise)
end

# kernel-spec form (issue #290): `gaussianprocess(X, matern52_kernel(l, v), noise)`
function gaussianprocess(inputs, kernel::AbstractGPKernel, noise)
    matrix = inputs isa AbstractVector ? reshape(collect(float.(inputs)), 1, length(inputs)) : inputs
    return GaussianProcessDist(matrix, kernel, noise)
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
    K = _gp_kernel_covariance(RBFKernel(lengthscale, variance), matrix, noise)
    return cholesky(Symmetric(K)).L
end

# kernel-spec form (issue #290)
function gp_cholesky(inputs, kernel::AbstractGPKernel, noise)
    matrix = inputs isa AbstractVector ? reshape(collect(float.(inputs)), 1, length(inputs)) : inputs
    K = _gp_kernel_covariance(kernel, matrix, noise)
    return cholesky(Symmetric(K)).L
end

function logpdf(gp::GaussianProcessDist, y::AbstractVector)
    n = size(gp.inputs, 2)
    length(y) == n ||
        throw(ArgumentError("gaussianprocess observation length $(length(y)) must match the $(n) inputs"))
    K = _gp_kernel_covariance(gp.kernel, gp.inputs, gp.noise)
    factorization = cholesky(Symmetric(K); check=false)
    issuccess(factorization) || return convert(float(eltype(K)), -Inf)
    alpha = factorization \ y
    logdet_half = sum(log, view(factorization.L, diagind(factorization.L)))
    return -sum(y .* alpha) / 2 - logdet_half - n * log(2 * oftype(logdet_half, pi)) / 2
end

function Random.rand(rng::AbstractRNG, gp::GaussianProcessDist)
    K = _gp_kernel_covariance(gp.kernel, gp.inputs, gp.noise)
    L = cholesky(Symmetric(K)).L
    return L * randn(rng, size(gp.inputs, 2))
end
