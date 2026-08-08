# CPU-reference distributions (structs, builders, logpdf, rand): structured families (mvnormal, mvnormaldense, dirichlet, mixture, broadcast/iid vector machinery).

struct DirichletDist{T<:Real} <: AbstractTeaDistribution
    alpha::Vector{T}

    function DirichletDist(alpha::Vector{T}) where {T<:Real}
        length(alpha) >= 2 || throw(ArgumentError("dirichlet requires at least 2 concentration parameters"))
        for value in alpha
            _primal(value) > zero(T) || throw(ArgumentError("dirichlet requires alpha > 0"))
        end
        return new{T}(alpha)
    end
end

# T<:Real (not AbstractFloat) so the ForwardDiff fallback can score models
# whose mean/scale vectors carry Dual entries (e.g. a latent mean), matching
# NormalDist and DirichletDist; rand stays restricted to AbstractFloat.
struct MvNormalDist{T<:Real} <: AbstractTeaDistribution
    mu::Vector{T}
    sigma::Vector{T}

    function MvNormalDist(mu::Vector{T}, sigma::Vector{T}) where {T<:Real}
        isempty(mu) && throw(ArgumentError("mvnormal requires at least one dimension"))
        length(mu) == length(sigma) || throw(ArgumentError("mvnormal requires mean and scale vectors with the same length"))
        for value in sigma
            _primal(value) > zero(T) || throw(ArgumentError("mvnormal requires sigma > 0 in every dimension"))
        end
        return new{T}(mu, sigma)
    end
end

# Finite marginalized mixture of AbstractTeaDistribution components. `weights` is
# kept generic (rather than pinned to Float64) so a latent simplex fed in through a
# dirichlet parameter slot arrives as ForwardDiff Duals and stays differentiable.
struct MixtureDist{W<:Real,C<:Tuple} <: AbstractTeaDistribution
    weights::Vector{W}
    components::C

    function MixtureDist(weights::Vector{W}, components::C) where {W<:Real,C<:Tuple}
        isempty(components) && throw(ArgumentError("mixture requires at least one component"))
        length(weights) == length(components) || throw(
            ArgumentError(
                "mixture requires one weight per component (got $(length(weights)) weights for $(length(components)) components)",
            ),
        )
        total = zero(W)
        for w in weights
            _primal(w) >= zero(W) || throw(ArgumentError("mixture weights must be nonnegative"))
            total += w
        end
        abs(total - one(W)) <= oftype(total, 1e-8) ||
            throw(ArgumentError("mixture weights must sum to 1 (within 1e-8)"))
        for comp in components
            comp isa AbstractTeaDistribution ||
                throw(ArgumentError("mixture components must be UncertainTea distributions"))
        end
        return new{W,C}(weights, components)
    end
end

# Dense-covariance multivariate normal parameterized by a lower-triangular
# Cholesky factor `scale_tril` (covariance = L * L'). Element types stay generic
# (like MixtureDist) so a latent mean built from ForwardDiff Duals remains
# differentiable. Only the lower triangle of `scale_tril` is ever read — any
# upper-triangular content is ignored — which lets callers pass a full matrix
# without wrapping it in `LinearAlgebra.LowerTriangular`.
struct MvNormalDenseDist{M<:AbstractVector,S<:AbstractMatrix} <: AbstractTeaDistribution
    mu::M
    scale_tril::S

    function MvNormalDenseDist(mu::AbstractVector, scale_tril::AbstractMatrix)
        isempty(mu) && throw(ArgumentError("mvnormaldense requires at least one dimension"))
        size(scale_tril, 1) == size(scale_tril, 2) || throw(ArgumentError(
            "mvnormaldense requires a square scale_tril matrix, got size $(size(scale_tril))",
        ))
        size(scale_tril, 1) == length(mu) || throw(
            ArgumentError(
                "mvnormaldense requires scale_tril of size $(length(mu))x$(length(mu)) to match the mean length, got $(size(scale_tril))",
            ),
        )
        for index = 1:size(scale_tril, 1)
            _primal(scale_tril[index, index]) > 0 || throw(ArgumentError(
                "mvnormaldense requires strictly positive scale_tril diagonal entries",
            ))
        end
        return new{typeof(mu),typeof(scale_tril)}(mu, scale_tril)
    end
end

# LKJ prior over the Cholesky factor of a d x d correlation matrix, scored on
# the column-major PACKED lower triangle (length d*(d+1)/2, diagonal included) —
# the same layout `CholeskyCorrTransform` produces. `eta` stays generic so a
# latent-dependent concentration remains differentiable.
struct LKJCholeskyDist{T<:Real} <: AbstractTeaDistribution
    d::Int
    eta::T

    function LKJCholeskyDist(d::Int, eta::Real)
        d >= 2 || throw(ArgumentError("lkjcholesky requires a dimension d >= 2"))
        _primal(eta) > 0 || throw(ArgumentError("lkjcholesky requires a concentration eta > 0"))
        promoted_eta = float(eta)
        return new{typeof(promoted_eta)}(d, promoted_eta)
    end
end

# Diagonal-scale multivariate Student-t: the heavy-tailed analogue of `mvnormal`
# (scale matrix Sigma = diag(sigma^2)). Unlike a product of univariate t's, the d
# dimensions share ONE chi-square mixing variable, so the density couples through
# a single quadratic form. Element types stay generic (like `MvNormalDist`) so a
# latent-dependent mean/scale/df remains ForwardDiff-differentiable.
struct MvStudentTDist{N<:Real,T<:Real} <: AbstractTeaDistribution
    nu::N
    mu::Vector{T}
    sigma::Vector{T}

    function MvStudentTDist(nu::N, mu::Vector{T}, sigma::Vector{T}) where {N<:Real,T<:Real}
        _primal(nu) > zero(_primal(nu)) || throw(ArgumentError("mvstudentt requires degrees of freedom nu > 0"))
        isempty(mu) && throw(ArgumentError("mvstudentt requires at least one dimension"))
        length(mu) == length(sigma) ||
            throw(ArgumentError("mvstudentt requires mean and scale vectors with the same length"))
        for value in sigma
            _primal(value) > zero(T) ||
                throw(ArgumentError("mvstudentt requires sigma > 0 in every dimension"))
        end
        return new{N,T}(nu, mu, sigma)
    end
end

# Dense-scale multivariate Student-t parameterized by the lower-triangular
# Cholesky factor `scale_tril` of the scale matrix (Sigma = L * L'). The dense
# analogue of `mvnormaldense`; only the lower triangle of `scale_tril` is read.
struct MvStudentTDenseDist{N<:Real,M<:AbstractVector,S<:AbstractMatrix} <: AbstractTeaDistribution
    nu::N
    mu::M
    scale_tril::S

    function MvStudentTDenseDist(nu::N, mu::AbstractVector, scale_tril::AbstractMatrix) where {N<:Real}
        _primal(nu) > zero(_primal(nu)) ||
            throw(ArgumentError("mvstudenttdense requires degrees of freedom nu > 0"))
        isempty(mu) && throw(ArgumentError("mvstudenttdense requires at least one dimension"))
        size(scale_tril, 1) == size(scale_tril, 2) || throw(ArgumentError(
            "mvstudenttdense requires a square scale_tril matrix, got size $(size(scale_tril))",
        ))
        size(scale_tril, 1) == length(mu) || throw(
            ArgumentError(
                "mvstudenttdense requires scale_tril of size $(length(mu))x$(length(mu)) to match the mean length, got $(size(scale_tril))",
            ),
        )
        for index = 1:size(scale_tril, 1)
            _primal(scale_tril[index, index]) > 0 ||
                throw(ArgumentError("mvstudenttdense requires strictly positive scale_tril diagonal entries"))
        end
        return new{N,typeof(mu),typeof(scale_tril)}(nu, mu, scale_tril)
    end
end

# Wishart / inverse-Wishart covariance-matrix priors. The value is the PACKED
# column-major lower triangle of the Cholesky factor L of the sampled PD matrix
# M = L * L' (the same packed layout `CholeskyCorrTransform`/`CholeskyCovTransform`
# produce), so `logpdf` is the density over L. `scale` is the d x d scale matrix S
# (Wishart) / S (inverse-Wishart); only its lower triangle is read. `nu` is the
# degrees of freedom. These are CPU-reference families (no backend/device lowering).
struct WishartDist{N<:Real,S<:AbstractMatrix} <: AbstractTeaDistribution
    d::Int
    nu::N
    scale::S

    function WishartDist(nu::N, scale::AbstractMatrix) where {N<:Real}
        size(scale, 1) == size(scale, 2) ||
            throw(ArgumentError("wishart requires a square scale matrix, got size $(size(scale))"))
        d = size(scale, 1)
        d >= 1 || throw(ArgumentError("wishart requires a dimension d >= 1"))
        _primal(nu) > d - 1 || throw(ArgumentError("wishart requires degrees of freedom nu > d - 1 = $(d - 1)"))
        return new{N,typeof(scale)}(d, nu, scale)
    end
end

struct InverseWishartDist{N<:Real,S<:AbstractMatrix} <: AbstractTeaDistribution
    d::Int
    nu::N
    scale::S

    function InverseWishartDist(nu::N, scale::AbstractMatrix) where {N<:Real}
        size(scale, 1) == size(scale, 2) ||
            throw(ArgumentError("inversewishart requires a square scale matrix, got size $(size(scale))"))
        d = size(scale, 1)
        d >= 1 || throw(ArgumentError("inversewishart requires a dimension d >= 1"))
        _primal(nu) > d - 1 ||
            throw(ArgumentError("inversewishart requires degrees of freedom nu > d - 1 = $(d - 1)"))
        return new{N,typeof(scale)}(d, nu, scale)
    end
end

# Broadcast (vectorized) normal observation. `mu` and `sigma` may each be a real
# scalar or an `AbstractVector`; a single vector-valued choice scores every element
# elementwise. This is the runtime counterpart of the `{:y} ~ normal.(mu, sigma)`
# dot-call DSL syntax and of the backend `BackendBroadcastNormalChoicePlanStep`.
struct BroadcastNormalDist{M,S} <: AbstractTeaDistribution
    mu::M
    sigma::S
end

BroadcastNormalDist(mu::Union{Real,AbstractVector}, sigma::Union{Real,AbstractVector}) =
    BroadcastNormalDist{typeof(mu),typeof(sigma)}(mu, sigma)

_broadcast_arg_length(::Real) = nothing
_broadcast_arg_length(v::AbstractVector) = length(v)
_broadcast_arg_getindex(x::Real, ::Int) = x
_broadcast_arg_getindex(v::AbstractVector, i::Int) = v[i]

function _broadcast_normal_length(mu, sigma)
    mu_length = _broadcast_arg_length(mu)
    sigma_length = _broadcast_arg_length(sigma)
    if !isnothing(mu_length) && !isnothing(sigma_length)
        mu_length == sigma_length ||
            throw(
                DimensionMismatch(
                    "broadcast normal requires vector arguments of equal length, got $(mu_length) and $(sigma_length)",
                ),
            )
        return mu_length
    end
    return something(mu_length, sigma_length, Some(nothing))
end

function _broadcast_normal_check_length(mu, sigma, n::Int)
    for arg in (mu, sigma)
        arg_length = _broadcast_arg_length(arg)
        isnothing(arg_length) && continue
        arg_length == n ||
            throw(DimensionMismatch("broadcast normal argument length $(arg_length) does not match value length $n"))
    end
    return nothing
end

# --- generic broadcast scalar observations (issue #287) ----------------------
#
# `BroadcastScalarDist{F}` is the family-generic runtime counterpart of the
# dot-call observation `{:y} ~ family.(args...)` for every broadcast family
# other than `normal` (which keeps its dedicated `BroadcastNormalDist` /
# backend-native machinery unchanged). One generic type + two 1-line glue
# methods per family — mapping onto the single-source scalar kernels and
# partials from issue #285 — replace the per-family plumbing a new broadcast
# family used to need. Observation-only: latents are rejected at lowering.
struct BroadcastScalarDist{F,A<:Tuple} <: AbstractTeaDistribution
    arguments::A
end

BroadcastScalarDist{F}(args...) where {F} = BroadcastScalarDist{F,typeof(args)}(args)

# Per-element log density, delegating to the single-source scalar kernels
# (src/distributions/scalar_kernels.jl). The argument order is the DSL
# constructor order; `y` is the observed element.
@inline _broadcast_element_logpdf(::Val{:poisson}, y, lambda) = _backend_poisson_logpdf(lambda, y)
@inline _broadcast_element_logpdf(::Val{:bernoulli}, y, p) = _backend_bernoulli_logpdf(p, y)
@inline _broadcast_element_logpdf(::Val{:bernoullilogit}, y, eta) = _backend_bernoullilogit_logpdf(eta, y)
@inline _broadcast_element_logpdf(::Val{:exponential}, y, rate) = _backend_exponential_logpdf(rate, y)
@inline _broadcast_element_logpdf(::Val{:studentt}, y, nu, mu, sigma) =
    _backend_studentt_logpdf(nu, mu, sigma, y)

# Per-element analytic partials with respect to the ARGUMENTS (the observed `y`
# is data), in constructor-argument order, or `nothing` to skip the gradient
# contribution (off-support element), mirroring the scalar accumulate guards.
@inline function _broadcast_element_partials(::Val{:poisson}, y, lambda)
    count = _poisson_count(y)
    isnothing(count) && return nothing
    return (_poisson_logpdf_partials(lambda, count),)
end
@inline _broadcast_element_partials(::Val{:bernoulli}, y, p) = (_bernoulli_logpdf_partials(p, y),)
@inline function _broadcast_element_partials(::Val{:bernoullilogit}, y, eta)
    support = _bernoulli_value(y)
    isnothing(support) && return nothing
    return (_bernoullilogit_logpdf_partials(eta, support),)
end
@inline function _broadcast_element_partials(::Val{:exponential}, y, rate)
    y >= 0 || return nothing
    dvalue, drate = _exponential_logpdf_partials(rate, y)
    return (drate,)
end
@inline function _broadcast_element_partials(::Val{:studentt}, y, nu, mu, sigma)
    dvalue, dnu, dmu, dsigma = _studentt_logpdf_partials(nu, mu, sigma, y)
    return (dnu, dmu, dsigma)
end

# Per-element base distribution for `rand` (prior/predictive draws).
@inline _broadcast_element_dist(::Val{:poisson}, lambda) = poisson(lambda)
@inline _broadcast_element_dist(::Val{:bernoulli}, p) = bernoulli(p)
@inline _broadcast_element_dist(::Val{:bernoullilogit}, eta) = bernoullilogit(eta)
@inline _broadcast_element_dist(::Val{:exponential}, rate) = exponential(rate)
@inline _broadcast_element_dist(::Val{:studentt}, nu, mu, sigma) = studentt(nu, mu, sigma)

function _broadcast_scalar_length(arguments::Tuple)
    n = nothing
    for arg in arguments
        arg_length = _broadcast_arg_length(arg)
        isnothing(arg_length) && continue
        if isnothing(n)
            n = arg_length
        else
            n == arg_length || throw(
                DimensionMismatch(
                    "broadcast observation requires vector arguments of equal length, got $(n) and $(arg_length)",
                ),
            )
        end
    end
    return n
end

function _broadcast_scalar_check_length(arguments::Tuple, n::Int)
    for arg in arguments
        arg_length = _broadcast_arg_length(arg)
        isnothing(arg_length) && continue
        arg_length == n || throw(DimensionMismatch(
            "broadcast observation argument length $(arg_length) does not match value length $n",
        ))
    end
    return nothing
end

function logpdf(dist::BroadcastScalarDist{F}, x) where {F}
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector ||
        throw(ArgumentError("broadcast observation logpdf expects a vector or tuple value"))
    n = length(values)
    n >= 1 || throw(ArgumentError("broadcast observation requires a non-empty value"))
    _broadcast_scalar_check_length(dist.arguments, n)
    accumulator = _broadcast_element_logpdf(
        Val(F), float(values[1]),
        map(arg -> _broadcast_arg_getindex(arg, 1), dist.arguments)...,
    )
    for index = 2:n
        accumulator += _broadcast_element_logpdf(
            Val(F), float(values[index]),
            map(arg -> _broadcast_arg_getindex(arg, index), dist.arguments)...,
        )
    end
    return accumulator
end

function Random.rand(rng::AbstractRNG, dist::BroadcastScalarDist{F}) where {F}
    n = _broadcast_scalar_length(dist.arguments)
    isnothing(n) && throw(ArgumentError(
        "broadcast observation rand requires at least one vector argument to determine the length",
    ))
    return [
        rand(rng, _broadcast_element_dist(Val(F), map(arg -> _broadcast_arg_getindex(arg, index), dist.arguments)...))
        for index = 1:n
    ]
end

"""
    iid(base, n)

Vector of `n` independent draws from the `base` distribution under a single
address: `theta ~ iid(normal(mu, tau), 8)` inside a `@tea` model gives a
length-8 vector choice. In static models the count `n` must be a literal
integer at macro-expansion time (the plan needs the shape). For the
location-scale bases `normal`, `studentt`, and `laplace`,
`theta ~ iid(normal(mu, tau), 8, reparam=:noncentered)` samples standardized
coordinates and rescales — the standard fix for funnel geometries in
hierarchical models.
"""
function iid(base::AbstractTeaDistribution, n::Integer)
    n >= 1 || throw(ArgumentError("iid requires n >= 1"))
    return IIDDist(base, Int(n))
end

# `n` independent-and-identically-distributed draws from `base` under a single
# address. Runtime counterpart of the `eps ~ iid(dist_call, n)` DSL sugar.
struct IIDDist{D<:AbstractTeaDistribution} <: AbstractTeaDistribution
    base::D
    n::Int

    function IIDDist(base::D, n::Int) where {D<:AbstractTeaDistribution}
        n >= 1 || throw(ArgumentError("iid requires n >= 1"))
        return new{D}(base, n)
    end
end

"""
    dirichlet(alpha)
    dirichlet(a1, a2, ...)

Dirichlet distribution over probability vectors (the simplex) with
concentration parameters `alpha` (a vector, or separate positive arguments).
"""
function dirichlet(alpha::AbstractVector)
    promoted = map(float, collect(alpha))
    return DirichletDist(promoted)
end

function dirichlet(alpha::Vararg{Real})
    return DirichletDist(collect(promote(map(float, alpha)...)))
end

function _mvnormal_vector(values)
    if values isa AbstractVector || values isa Tuple
        return collect(values)
    end
    throw(ArgumentError("mvnormal requires vector-like mean and scale arguments"))
end

function _mvnormal_promoted_type(vectors...)
    promoted_type = nothing
    for vector in vectors
        for value in vector
            promoted_type = isnothing(promoted_type) ? typeof(value) : promote_type(promoted_type, typeof(value))
        end
    end
    isnothing(promoted_type) && throw(ArgumentError("mvnormal requires at least one dimension"))
    return float(promoted_type)
end

"""
    mvnormal(mu, sigma)

Multivariate normal with independent components: mean vector `mu` and
per-component standard deviations `sigma` (a diagonal covariance). For a full
covariance use `mvnormaldense`.
"""
function mvnormal(mu, sigma)
    mu_values = _mvnormal_vector(mu)
    sigma_values = _mvnormal_vector(sigma)
    length(mu_values) == length(sigma_values) ||
        throw(ArgumentError("mvnormal requires mean and scale vectors with the same length"))
    isempty(mu_values) && throw(ArgumentError("mvnormal requires at least one dimension"))
    promoted_type = _mvnormal_promoted_type(mu_values, sigma_values)
    return MvNormalDist(
        promoted_type[value for value in mu_values],
        promoted_type[value for value in sigma_values],
    )
end

"""
    mvnormaldense(mu, scale_tril)

Multivariate normal with mean vector `mu` and dense covariance given by its
lower-triangular Cholesky factor `scale_tril` (covariance `L * L'` with
`L = scale_tril`).
"""
function mvnormaldense(mu, scale_tril)
    (mu isa AbstractVector || mu isa Tuple) ||
        throw(ArgumentError("mvnormaldense requires a vector-like mean argument"))
    scale_tril isa AbstractMatrix ||
        throw(ArgumentError("mvnormaldense requires a matrix scale_tril argument"))
    # `map(identity, collect(...))` narrows a `Vector{Any}` (e.g. a compiled
    # `[mu, mu]` vector expression holding ForwardDiff Duals) to the promoted
    # element type so downstream arithmetic stays differentiable.
    mu_values = map(identity, collect(mu))
    return MvNormalDenseDist(mu_values, scale_tril)
end

function mvstudentt(nu, mu, sigma)
    mu_values = _mvnormal_vector(mu)
    sigma_values = _mvnormal_vector(sigma)
    length(mu_values) == length(sigma_values) ||
        throw(ArgumentError("mvstudentt requires mean and scale vectors with the same length"))
    isempty(mu_values) && throw(ArgumentError("mvstudentt requires at least one dimension"))
    promoted_type = _mvnormal_promoted_type(mu_values, sigma_values)
    return MvStudentTDist(
        nu,
        promoted_type[value for value in mu_values],
        promoted_type[value for value in sigma_values],
    )
end

function mvstudenttdense(nu, mu, scale_tril)
    (mu isa AbstractVector || mu isa Tuple) ||
        throw(ArgumentError("mvstudenttdense requires a vector-like mean argument"))
    scale_tril isa AbstractMatrix ||
        throw(ArgumentError("mvstudenttdense requires a matrix scale_tril argument"))
    mu_values = map(identity, collect(mu))
    return MvStudentTDenseDist(nu, mu_values, scale_tril)
end

function wishart(nu, scale)
    scale isa AbstractMatrix ||
        throw(ArgumentError("wishart requires a matrix scale argument"))
    return WishartDist(nu, scale)
end

function inversewishart(nu, scale)
    scale isa AbstractMatrix ||
        throw(ArgumentError("inversewishart requires a matrix scale argument"))
    return InverseWishartDist(nu, scale)
end

# `d` accepts any integer-valued number: the compiled evaluator reconstructs the
# distribution from the `DistributionSpec` argument vector, where an integer
# literal may arrive promoted to Float64 (see `_lkjcholesky_static_size`).
function lkjcholesky(d, eta)
    (d isa Real && isinteger(d)) ||
        throw(ArgumentError("lkjcholesky requires an integer dimension d"))
    return LKJCholeskyDist(Int(d), eta)
end

# Log of the LKJ normalizing constant over correlation matrices (Lewandowski,
# Kurowicka & Joe 2009, cvine construction): the density over the free
# below-diagonal Cholesky coordinates is
#   prod_{i=2..d} L[i,i]^(d - i + 2*eta - 2) / c_d(eta)
# with
#   log c_d(eta) = sum_{k=1}^{d-1} (d - k) * [(2*eta - 2 + d - k) * log(2)
#                                             + log Beta(a_k, a_k)],
#   a_k = eta + (d - 1 - k) / 2,
# i.e. one symmetric Beta factor (rescaled to (-1, 1)) per canonical partial
# correlation at cvine tree level k, replicated (d - k) times.
function _lkj_log_normalizing_constant(d::Int, eta)
    eta_f = float(eta)
    log_c = zero(eta_f)
    for k = 1:(d-1)
        a = eta_f + (d - 1 - k) / 2
        log_beta = loggamma(a) + loggamma(a) - loggamma(2 * a)
        log_c += (d - k) * ((2 * eta_f - 2 + d - k) * log(oftype(eta_f, 2)) + log_beta)
    end
    return -log_c
end

# d/deta of `_lkj_log_normalizing_constant`: each cvine Beta factor contributes
# (d - k) * (2 log 2 + d logB(a_k, a_k)/da_k) with d logB/da = 2 (psi(a) - psi(2a)).
function _lkj_log_normalizing_constant_deta(d::Int, eta)
    eta_f = float(eta)
    dlog_c = zero(eta_f)
    for k = 1:(d-1)
        a = eta_f + (d - 1 - k) / 2
        dlog_c += (d - k) * (2 * log(oftype(eta_f, 2)) + 2 * (digamma(a) - digamma(2 * a)))
    end
    return -dlog_c
end

# Un-pack a column-major packed correlation Cholesky factor and scale row i by
# `scales[i]`, producing the dense d x d lower-triangular `scale_tril` for
# `mvnormaldense` (covariance = diag(scales) * Omega * diag(scales)). Plain
# loops keep ForwardDiff Duals flowing through, and `map(identity, collect(...))`
# narrows `Vector{Any}` inputs from compiled expressions (mirroring
# `mvnormaldense`).
function scale_cholesky(scales::AbstractVector, packed_corr_chol::AbstractVector)
    isempty(scales) && throw(ArgumentError("scale_cholesky requires at least one scale"))
    scale_values = map(identity, collect(scales))
    packed_values = map(identity, collect(packed_corr_chol))
    d = length(scale_values)
    expected = (d * (d + 1)) ÷ 2
    length(packed_values) == expected || throw(
        DimensionMismatch(
            "scale_cholesky expected a packed lower triangle of length $expected for $d scales, got $(length(packed_values))",
        ),
    )
    T = promote_type(eltype(scale_values), eltype(packed_values))
    result = zeros(T, d, d)
    for col = 1:d
        for row = col:d
            result[row, col] = scale_values[row] * packed_values[_packed_lower_index(d, row, col)]
        end
    end
    return result
end

# `map(identity, collect(...))` narrows a `Vector{Any}` (the compiled
# evaluator's materialization of a vector-literal weights argument, issue #75)
# to the promoted element type, mirroring `mvnormaldense` and `scale_cholesky`;
# heterogeneous weights are then promoted to a common Real type.
"""
    mixture(weights, components...)

Finite mixture of distributions with the component indicator marginalized out:
`mixture([0.3, 0.7], normal(-2.0, 1.0), normal(2.0, 1.0))`. `weights` must be
nonnegative and sum to 1, with one weight per component; the log-density is the
log-sum-exp over the weighted component log-densities.
"""
function mixture(weights, components...)
    isempty(components) && throw(ArgumentError("mixture requires at least one component"))
    weight_values = map(identity, collect(weights))
    isempty(weight_values) && throw(
        ArgumentError(
            "mixture requires one weight per component (got 0 weights for $(length(components)) components)",
        ),
    )
    return MixtureDist(collect(promote(weight_values...)), components)
end

function Random.rand(rng::AbstractRNG, dist::DirichletDist)
    draws = Vector{eltype(dist.alpha)}(undef, length(dist.alpha))
    total = zero(eltype(dist.alpha))
    for index in eachindex(dist.alpha)
        draw = _rand_gamma_marsaglia(rng, float(dist.alpha[index]), one(float(dist.alpha[index])))
        draws[index] = draw
        total += draw
    end
    for index in eachindex(draws)
        draws[index] /= total
    end
    return draws
end

function Random.rand(rng::AbstractRNG, dist::MvNormalDist{T}) where {T<:AbstractFloat}
    draws = Vector{T}(undef, length(dist.mu))
    for index in eachindex(draws)
        draws[index] = dist.mu[index] + dist.sigma[index] * randn(rng, T)
    end
    return draws
end

# Draw mu + L * z with z ~ standard normal, reading only the lower triangle of
# `scale_tril`.
function Random.rand(rng::AbstractRNG, dist::MvNormalDenseDist)
    dimension = length(dist.mu)
    T = float(promote_type(typeof(float(dist.mu[1])), typeof(float(dist.scale_tril[1, 1]))))
    noise = randn(rng, T, dimension)
    draws = Vector{T}(undef, dimension)
    for row = 1:dimension
        accumulator = T(dist.mu[row])
        for col = 1:row
            accumulator += T(dist.scale_tril[row, col]) * noise[col]
        end
        draws[row] = accumulator
    end
    return draws
end

# Multivariate-t draw: sample a Gaussian z ~ N(0, Sigma) and scale by an
# inverse-sqrt gamma mixing variable w ~ Gamma(nu/2, nu/2) (mean 1), i.e.
# x = mu + z / sqrt(w) -- the standard normal / sqrt(chi-square/nu) construction,
# mirroring the scalar `StudentTDist` rand.
function Random.rand(rng::AbstractRNG, dist::MvStudentTDist{N,T}) where {N<:Real,T<:AbstractFloat}
    nu = float(dist.nu)
    w = _rand_gamma_marsaglia(rng, nu / 2, nu / 2)
    scale = inv(sqrt(w))
    draws = Vector{T}(undef, length(dist.mu))
    for index in eachindex(draws)
        draws[index] = dist.mu[index] + dist.sigma[index] * randn(rng, T) * T(scale)
    end
    return draws
end

function Random.rand(rng::AbstractRNG, dist::MvStudentTDenseDist)
    dimension = length(dist.mu)
    T = float(promote_type(typeof(float(dist.mu[1])), typeof(float(dist.scale_tril[1, 1]))))
    nu = float(dist.nu)
    w = _rand_gamma_marsaglia(rng, nu / 2, nu / 2)
    scale = T(inv(sqrt(w)))
    noise = randn(rng, T, dimension)
    draws = Vector{T}(undef, dimension)
    for row = 1:dimension
        accumulator = zero(T)
        for col = 1:row
            accumulator += T(dist.scale_tril[row, col]) * noise[col]
        end
        draws[row] = T(dist.mu[row]) + accumulator * scale
    end
    return draws
end

# Plain-loop lower Cholesky factor of the (lower triangle of the) d x d matrix S,
# S = C * C'. ForwardDiff-safe (no LinearAlgebra factorization object).
function _dense_cholesky_lower(S::AbstractMatrix)
    d = size(S, 1)
    Tc = typeof(sqrt(float(S[1, 1])))
    C = zeros(Tc, d, d)
    for col = 1:d
        diag = float(S[col, col])
        for k = 1:(col-1)
            diag -= C[col, k] * C[col, k]
        end
        C[col, col] = sqrt(diag)
        for row = (col+1):d
            acc = float(S[row, col])
            for k = 1:(col-1)
                acc -= C[row, k] * C[col, k]
            end
            C[row, col] = acc / C[col, col]
        end
    end
    return C
end

# Forward substitution solving the lower-triangular system A * Y = B for a full
# matrix right-hand side B (both A, B are d x d). Plain loops keep it Dual-safe.
function _forward_solve_matrix(A::AbstractMatrix, B::AbstractMatrix)
    d = size(A, 1)
    Ty = typeof(float(B[1, 1]) / float(A[1, 1]))
    Y = zeros(Ty, d, d)
    for col = 1:d
        for row = 1:d
            acc = float(B[row, col])
            for k = 1:(row-1)
                acc -= A[row, k] * Y[k, col]
            end
            Y[row, col] = acc / A[row, row]
        end
    end
    return Y
end

# Bartlett-decomposition Wishart draw: with S = C C', a sample W ~ Wishart(nu, S)
# has Cholesky factor L = C * A, where A is lower-triangular with A[i,i] =
# sqrt(chi-square_{nu-i+1}) and A[i,j] ~ N(0,1) for i > j. Returns the packed
# lower triangle of L.
function _rand_wishart_factor(rng::AbstractRNG, d::Int, nu::Float64, C::AbstractMatrix)
    A = zeros(Float64, d, d)
    for i = 1:d
        # chi-square_k = Gamma(k/2, rate = 1/2)
        A[i, i] = sqrt(_rand_gamma_marsaglia(rng, (nu - i + 1) / 2, 0.5))
        for j = 1:(i-1)
            A[i, j] = randn(rng, Float64)
        end
    end
    L = zeros(Float64, d, d)
    for row = 1:d
        for col = 1:row
            acc = 0.0
            for k = col:row
                acc += C[row, k] * A[k, col]
            end
            L[row, col] = acc
        end
    end
    packed = Vector{Float64}(undef, (d * (d + 1)) ÷ 2)
    for col = 1:d, row = col:d
        packed[_packed_lower_index(d, row, col)] = L[row, col]
    end
    return packed
end

function Random.rand(rng::AbstractRNG, dist::WishartDist)
    d = dist.d
    C = _dense_cholesky_lower(Float64.(dist.scale))
    return _rand_wishart_factor(rng, d, Float64(dist.nu), C)
end

# X ~ InverseWishart(nu, S) iff X^{-1} ~ Wishart(nu, S^{-1}). Sample W in that
# Wishart, invert to X = W^{-1}, and return the packed lower Cholesky factor of X.
function Random.rand(rng::AbstractRNG, dist::InverseWishartDist)
    d = dist.d
    S = Float64.(dist.scale)
    Sinv = inv(S)
    # symmetrize to kill round-off asymmetry before the Cholesky
    Sinv = (Sinv + Sinv') / 2
    Cinv = _dense_cholesky_lower(Sinv)
    packed_w = _rand_wishart_factor(rng, d, Float64(dist.nu), Cinv)
    Lw = zeros(Float64, d, d)
    for col = 1:d, row = col:d
        Lw[row, col] = packed_w[_packed_lower_index(d, row, col)]
    end
    W = Lw * Lw'
    X = inv(W)
    X = (X + X') / 2
    Lx = _dense_cholesky_lower(X)
    packed = Vector{Float64}(undef, (d * (d + 1)) ÷ 2)
    for col = 1:d, row = col:d
        packed[_packed_lower_index(d, row, col)] = Lx[row, col]
    end
    return packed
end

# LKJ cvine construction (Lewandowski, Kurowicka & Joe 2009): the canonical
# partial correlation feeding below-diagonal entry (i, j) sits at cvine tree
# level j and is distributed as w = 2 * Beta(a_j, a_j) - 1 with
# a_j = eta + (d - 1 - j) / 2; running Stan's `cholesky_corr_constrain` forward
# recursion on those w values yields the packed correlation Cholesky factor.
function Random.rand(rng::AbstractRNG, dist::LKJCholeskyDist)
    d = dist.d
    eta = Float64(dist.eta)
    packed = Vector{Float64}(undef, (d * (d + 1)) ÷ 2)
    packed[_packed_lower_index(d, 1, 1)] = 1.0
    for row = 2:d
        sum_sqs = 0.0
        for col = 1:(row-1)
            a = eta + (d - 1 - col) / 2
            w = 2.0 * rand(rng, BetaDist(a, a)) - 1.0
            entry = w * sqrt(1 - sum_sqs)
            packed[_packed_lower_index(d, row, col)] = entry
            sum_sqs += entry * entry
        end
        packed[_packed_lower_index(d, row, row)] = sqrt(1 - sum_sqs)
    end
    return packed
end

# The scoring path validates the scale through the `normal` builder; the
# sampling path must enforce the same constraint on every element (issue #87).
_broadcast_normal_validate_sigma(sigma::Real) =
    sigma > zero(sigma) ? nothing : throw(ArgumentError("normal requires sigma > 0"))

function _broadcast_normal_validate_sigma(sigma::AbstractVector)
    for value in sigma
        value > zero(value) || throw(ArgumentError("normal requires sigma > 0"))
    end
    return nothing
end

function Random.rand(rng::AbstractRNG, dist::BroadcastNormalDist)
    _broadcast_normal_validate_sigma(dist.sigma)
    n = _broadcast_normal_length(dist.mu, dist.sigma)
    isnothing(n) && throw(
        ArgumentError(
            "broadcast normal with all-scalar arguments cannot infer a sample length; " *
            "constrain the observation or supply at least one vector argument",
        ),
    )
    mu1 = float(_broadcast_arg_getindex(dist.mu, 1))
    sigma1 = float(_broadcast_arg_getindex(dist.sigma, 1))
    T = float(promote_type(typeof(mu1), typeof(sigma1)))
    draws = Vector{T}(undef, n)
    for index = 1:n
        mu = _broadcast_arg_getindex(dist.mu, index)
        sigma = _broadcast_arg_getindex(dist.sigma, index)
        draws[index] = mu + sigma * randn(rng, T)
    end
    return draws
end

function Random.rand(rng::AbstractRNG, dist::IIDDist)
    first_draw = rand(rng, dist.base)
    draws = Vector{typeof(first_draw)}(undef, dist.n)
    draws[1] = first_draw
    for index = 2:dist.n
        draws[index] = rand(rng, dist.base)
    end
    return draws
end

function Random.rand(rng::AbstractRNG, dist::MixtureDist)
    threshold = rand(rng, Float64)
    cumulative = 0.0
    index = length(dist.components)
    for (k, w) in enumerate(dist.weights)
        cumulative += w
        if threshold <= cumulative
            index = k
            break
        end
    end
    return rand(rng, dist.components[index])
end

function logpdf(dist::DirichletDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("dirichlet logpdf expects a vector or tuple value"))
    length(values) == length(dist.alpha) || return -Inf

    promoted_values = map(float, collect(values))
    total = zero(eltype(promoted_values))
    accumulator = loggamma(sum(dist.alpha)) - sum(loggamma, dist.alpha)
    for (value, alpha) in zip(promoted_values, dist.alpha)
        _primal(value) > zero(_primal(value)) || return oftype(value, -Inf)
        total += value
        accumulator += (alpha - one(alpha)) * log(value)
    end
    abs(total - one(total)) <= sqrt(eps(float(total))) * length(promoted_values) * 16 || return oftype(total, -Inf)
    return accumulator
end

function logpdf(dist::MvNormalDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("mvnormal logpdf expects a vector or tuple value"))
    length(values) == length(dist.mu) || return -Inf

    accumulator = logpdf(normal(dist.mu[1], dist.sigma[1]), values[1])
    for index = 2:length(values)
        accumulator += logpdf(normal(dist.mu[index], dist.sigma[index]), values[index])
    end
    return accumulator
end

# Dense mvnormal log density via hand-rolled forward substitution solving
# L z = x - mu, reading only the lower triangle of `scale_tril`. Avoiding
# LinearAlgebra factorization objects keeps every operation a plain scalar
# loop that ForwardDiff Duals flow through:
#   logpdf = -sum_i log(L[i,i]) - z'z / 2 - d * log(2*pi) / 2.
function logpdf(dist::MvNormalDenseDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("mvnormaldense logpdf expects a vector or tuple value"))
    dimension = length(dist.mu)
    length(values) == dimension || return -Inf
    L = dist.scale_tril
    z1 = (values[1] - dist.mu[1]) / L[1, 1]
    solved = Vector{typeof(z1)}(undef, dimension)
    solved[1] = z1
    log_det = log(L[1, 1])
    quadratic = z1 * z1
    for row = 2:dimension
        residual = values[row] - dist.mu[row]
        for col = 1:(row-1)
            residual -= L[row, col] * solved[col]
        end
        z = residual / L[row, row]
        solved[row] = z
        log_det += log(L[row, row])
        quadratic += z * z
    end
    return -log_det - quadratic / 2 - dimension * log(2 * pi) / 2
end

# Shared multivariate-t log density given the Cholesky log-determinant
# `log_det = sum_i log L[i,i]` (so log|Sigma| = 2 log_det) and the Mahalanobis
# quadratic form `q = (x - mu)' Sigma^{-1} (x - mu)`:
#   loggamma((nu+d)/2) - loggamma(nu/2) - (d/2) log(nu*pi) - log_det
#     - ((nu+d)/2) * log1p(q / nu).
function _mvstudentt_log_density(nu, d::Int, log_det, quadratic)
    half = (nu + d) / 2
    return loggamma(half) - loggamma(nu / 2) - (d / 2) * log(nu * float(pi)) - log_det -
           half * log1p(quadratic / nu)
end

function logpdf(dist::MvStudentTDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("mvstudentt logpdf expects a vector or tuple value"))
    d = length(dist.mu)
    length(values) == d || return -Inf
    z1 = (values[1] - dist.mu[1]) / dist.sigma[1]
    quadratic = z1 * z1
    log_det = log(dist.sigma[1])
    for index = 2:d
        z = (values[index] - dist.mu[index]) / dist.sigma[index]
        quadratic += z * z
        log_det += log(dist.sigma[index])
    end
    return _mvstudentt_log_density(dist.nu, d, log_det, quadratic)
end

# Dense multivariate-t density via the same forward substitution as
# `MvNormalDenseDist`, wrapping the quadratic form in the Student-t tail.
function logpdf(dist::MvStudentTDenseDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("mvstudenttdense logpdf expects a vector or tuple value"))
    d = length(dist.mu)
    length(values) == d || return -Inf
    L = dist.scale_tril
    z1 = (values[1] - dist.mu[1]) / L[1, 1]
    solved = Vector{typeof(z1)}(undef, d)
    solved[1] = z1
    log_det = log(L[1, 1])
    quadratic = z1 * z1
    for row = 2:d
        residual = values[row] - dist.mu[row]
        for col = 1:(row-1)
            residual -= L[row, col] * solved[col]
        end
        z = residual / L[row, row]
        solved[row] = z
        log_det += log(L[row, row])
        quadratic += z * z
    end
    return _mvstudentt_log_density(dist.nu, d, log_det, quadratic)
end

# Log of the multivariate gamma function Gamma_d(a) =
#   pi^(d(d-1)/4) * prod_{j=1}^d Gamma(a + (1 - j)/2).
function _log_multivariate_gamma(d::Int, a)
    total = (d * (d - 1) / 4) * log(float(pi)) + zero(float(a))
    for j = 1:d
        total += loggamma(a + (1 - j) / 2)
    end
    return total
end

# Unpack a column-major packed lower triangle into a dense d x d lower-tri matrix.
function _unpack_lower(values::AbstractVector, d::Int)
    T = typeof(float(values[firstindex(values)]))
    L = zeros(T, d, d)
    for col = 1:d, row = col:d
        L[row, col] = values[_packed_lower_index(d, row, col)]
    end
    return L
end

# Wishart / inverse-Wishart log density over the packed Cholesky factor L of the
# sampled PD matrix M = L L'. The distribution's own PD-matrix density is scored
# at M, then the constant-Jacobian term of the map L -> M = L L',
#   log|dM/dL| = d*log(2) + sum_i (d + 1 - i) * log(L[i,i]),
# converts it to a proper density over the free coordinates of L. Support: a
# strictly positive Cholesky diagonal (else -Inf).
function logpdf(dist::WishartDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("wishart logpdf expects a vector or tuple value"))
    d = dist.d
    length(values) == (d * (d + 1)) ÷ 2 || return -Inf
    L = _unpack_lower(values, d)
    acc0 = zero(float(values[firstindex(values)])) + zero(float(dist.nu))
    log_det_M = zero(acc0)
    jac = d * log(oftype(acc0, 2))
    for i = 1:d
        _primal(L[i, i]) > zero(_primal(L[i, i])) || return oftype(acc0, -Inf)
        log_det_M += log(L[i, i])
        jac += (d + 1 - i) * log(L[i, i])
    end
    log_det_M *= 2
    C = _dense_cholesky_lower(dist.scale)
    log_det_S = zero(acc0)
    for i = 1:d
        log_det_S += log(C[i, i])
    end
    log_det_S *= 2
    # tr(S^{-1} M) = || C^{-1} L ||_F^2 with S = C C', M = L L'.
    Y = _forward_solve_matrix(C, L)
    trace = zero(acc0)
    for col = 1:d, row = 1:d
        trace += Y[row, col] * Y[row, col]
    end
    nu = dist.nu
    density =
        ((nu - d - 1) / 2) * log_det_M - trace / 2 - (nu * d / 2) * log(oftype(acc0, 2)) -
        (nu / 2) * log_det_S - _log_multivariate_gamma(d, nu / 2)
    return density + jac
end

function logpdf(dist::InverseWishartDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("inversewishart logpdf expects a vector or tuple value"))
    d = dist.d
    length(values) == (d * (d + 1)) ÷ 2 || return -Inf
    L = _unpack_lower(values, d)
    acc0 = zero(float(values[firstindex(values)])) + zero(float(dist.nu))
    log_det_X = zero(acc0)
    jac = d * log(oftype(acc0, 2))
    for i = 1:d
        _primal(L[i, i]) > zero(_primal(L[i, i])) || return oftype(acc0, -Inf)
        log_det_X += log(L[i, i])
        jac += (d + 1 - i) * log(L[i, i])
    end
    log_det_X *= 2
    C = _dense_cholesky_lower(dist.scale)
    log_det_S = zero(acc0)
    for i = 1:d
        log_det_S += log(C[i, i])
    end
    log_det_S *= 2
    # tr(S X^{-1}) = || L^{-1} C ||_F^2 with X = L L', S = C C'.
    W = _forward_solve_matrix(L, C)
    trace = zero(acc0)
    for col = 1:d, row = 1:d
        trace += W[row, col] * W[row, col]
    end
    nu = dist.nu
    density =
        (nu / 2) * log_det_S - ((nu + d + 1) / 2) * log_det_X - trace / 2 -
        (nu * d / 2) * log(oftype(acc0, 2)) - _log_multivariate_gamma(d, nu / 2)
    return density + jac
end

# LKJ log density over the free (below-diagonal) coordinates of the packed
# correlation Cholesky factor:
#   lpdf = sum_{i=2..d} (d - i + 2*eta - 2) * log(L[i,i]) + log(1 / c_d(eta)).
# Support: packed length d*(d+1)/2, strictly positive diagonal entries, and
# every row of the unpacked factor a unit vector within tolerance (else -Inf) --
# the Cholesky factor of a correlation matrix has unit-norm rows (issue #78).
function logpdf(dist::LKJCholeskyDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("lkjcholesky logpdf expects a vector or tuple value"))
    d = dist.d
    expected = (d * (d + 1)) ÷ 2
    length(values) == expected || return -Inf

    accumulator = _lkj_log_normalizing_constant(d, dist.eta) + zero(float(values[firstindex(values)]))
    tolerance = sqrt(eps(Float64)) * d * 16
    for row = 1:d
        diagonal = values[_packed_lower_index(d, row, row)]
        _primal(diagonal) > zero(_primal(diagonal)) || return oftype(accumulator, -Inf)
        sum_sqs = zero(float(diagonal))
        for col = 1:row
            entry = values[_packed_lower_index(d, row, col)]
            sum_sqs += entry * entry
        end
        abs(sum_sqs - one(sum_sqs)) <= tolerance || return oftype(accumulator, -Inf)
        if row >= 2
            accumulator += (d - row + 2 * dist.eta - 2) * log(diagonal)
        end
    end
    return accumulator
end

function logpdf(dist::BroadcastNormalDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("broadcast normal logpdf expects a vector or tuple value"))
    n = length(values)
    n >= 1 || throw(ArgumentError("broadcast normal requires a non-empty value"))
    _broadcast_normal_check_length(dist.mu, dist.sigma, n)
    accumulator = logpdf(
        normal(_broadcast_arg_getindex(dist.mu, 1), _broadcast_arg_getindex(dist.sigma, 1)),
        values[1],
    )
    for index = 2:n
        accumulator += logpdf(
            normal(_broadcast_arg_getindex(dist.mu, index), _broadcast_arg_getindex(dist.sigma, index)),
            values[index],
        )
    end
    return accumulator
end

function logpdf(dist::IIDDist, x)
    values = x isa Tuple ? collect(x) : x
    values isa AbstractVector || throw(ArgumentError("iid logpdf expects a vector or tuple value"))
    length(values) == dist.n ||
        throw(DimensionMismatch("iid expects a value of length $(dist.n), got $(length(values))"))
    accumulator = logpdf(dist.base, values[1])
    for index = 2:dist.n
        accumulator += logpdf(dist.base, values[index])
    end
    return accumulator
end

# Per-component terms log(w_k) + logpdf(component_k, x) as a tuple. A zero weight
# yields log(0) = -Inf, which drops out of the max-shifted logsumexp below, so no
# explicit skipping is needed. Recursion keeps the tuple type-stable.
function _mixture_log_terms(weights, components::Tuple, x, index::Int)
    isempty(components) && return ()
    term = log(weights[index]) + logpdf(first(components), x)
    return (term, _mixture_log_terms(weights, Base.tail(components), x, index + 1)...)
end

function logpdf(dist::MixtureDist, x)
    terms = _mixture_log_terms(dist.weights, dist.components, x, 1)
    m = maximum(terms)
    isfinite(m) || return oftype(m, -Inf)
    total = zero(m)
    for term in terms
        total += exp(term - m)
    end
    return m + log(total)
end
