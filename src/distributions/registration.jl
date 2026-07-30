# Public registration API for user-defined distributions.
#
# A distribution defined outside the package participates in `@tea` models via
# the CPU reference path (generate/assess/logjoint and the ForwardDiff
# gradient fallback) once registered here. Registered families are honestly
# reported unsupported by `backend_report`/`device_report`, exactly like the
# built-in CPU-only families.
#
# Contract for the registered builder's return value:
#   - subtype `UncertainTea.AbstractTeaDistribution`
#   - extend `UncertainTea.logpdf(dist, x)` (return -Inf outside the support;
#     keep it ForwardDiff-Dual-friendly if the family is used as a latent)
#   - extend `Random.rand(rng::AbstractRNG, dist)`
#
# Registration is consulted when a `@tea` model is DEFINED, so register a
# family before defining models that use it (the same order Julia requires
# for any function you call at top level).

const BUILTIN_DISTRIBUTION_FAMILIES = (
    :normal,
    :lognormal,
    :laplace,
    :exponential,
    :gamma,
    :inversegamma,
    :weibull,
    :beta,
    :dirichlet,
    :bernoulli,
    :bernoullilogit,
    :binomial,
    :geometric,
    :negativebinomial,
    :poisson,
    :studentt,
    :categorical,
    :mvnormal,
    :mvnormaldense,
    :lkjcholesky,
    :truncatednormal,
    :truncatedstudentt,
    :mixture,
    :iid,
    :cauchy,
    :halfnormal,
    :halfcauchy,
    :uniform,
    :logistic,
    :gumbel,
)

struct UserDistributionRegistration
    builder::Any
    transform::Union{Nothing,AbstractParameterTransform}
end

const USER_DISTRIBUTION_REGISTRY = Dict{Symbol,UserDistributionRegistration}()

"""
    register_distribution(family::Symbol; builder, transform=nothing)

Register a user-defined distribution so `x ~ \$family(args...)` works inside
`@tea` models. `builder` is the function (or type constructor) called with the
model-side arguments; it must return an `AbstractTeaDistribution` implementing
`UncertainTea.logpdf(dist, x)` and `Random.rand(rng, dist)`.

`transform` declares the unconstrained parameterization used when the family
appears as a latent: one of the exported parameter transforms (e.g.
`IdentityTransform()` for real-line support, `LogTransform()` for positive
support, `LogitTransform()` for (0,1), `BoundedTransform(lower, upper)`).
Leave it `nothing` for observation-only families -- a latent then gets no
parameter slot, matching how unsupported built-in latents behave.

Models capture the builder and transform when they are DEFINED, so
re-registering a family affects only models defined afterwards; existing
models keep the distribution they were defined with. Built-in family names
cannot be overridden.
"""
function register_distribution(family::Symbol; builder, transform::Union{Nothing,AbstractParameterTransform}=nothing)
    family in BUILTIN_DISTRIBUTION_FAMILIES &&
        throw(ArgumentError("cannot register `$family`: it is a built-in distribution family"))
    # Inside @tea bodies a registered family name shadows same-named function
    # calls, so refuse names of primitives commonly used in model expressions.
    family in GPU_BACKEND_SUPPORTED_PRIMITIVES &&
        throw(ArgumentError("cannot register `$family`: it is a primitive used in model expressions"))
    builder isa Union{Function,Type} ||
        throw(ArgumentError("the builder for `$family` must be a function or type constructor, got $(typeof(builder))"))
    registration = UserDistributionRegistration(builder, transform)
    USER_DISTRIBUTION_REGISTRY[family] = registration
    return registration
end

"""
    registered_distributions() -> Vector{Symbol}

The user-registered distribution family names, sorted.
"""
registered_distributions() = sort!(collect(keys(USER_DISTRIBUTION_REGISTRY)))

function _registered_user_distribution(family::Symbol)
    return get(USER_DISTRIBUTION_REGISTRY, family, nothing)
end
