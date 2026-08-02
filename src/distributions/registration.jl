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
    :betabinomial,
    :multinomial,
    :discreteuniform,
    :geometric,
    :negativebinomial,
    :poisson,
    :studentt,
    :categorical,
    :mvnormal,
    :mvnormaldense,
    :gaussianprocess,
    :sparsegaussianprocess,
    :hmm,
    :orderedlogistic,
    :zeroinflatedpoisson,
    :zeroinflatednegativebinomial,
    :vonmises,
    :mvstudentt,
    :mvstudenttdense,
    :wishart,
    :inversewishart,
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
    :pareto,
    :frechet,
    :rayleigh,
    :inversegaussian,
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

!!! tip "Distributions.jl one-liner"
    With Distributions.jl loaded, the `UncertainTeaDistributionsExt` extension
    adds a positional method that writes the builder for you:
    `register_distribution(:skewnormal, Distributions.SkewNormal)` — see
    [`wrap_distribution`](@ref) for the support-mapping rules and limitations.
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

# Canonical package-extension pattern (see `to_mcmcchains`): the core declares
# the function with no methods; the UncertainTeaDistributionsExt extension
# (loaded when Distributions.jl is present) adds the implementation. Calling
# without Distributions loaded raises a MethodError — the intended "load
# Distributions.jl" signal.
"""
    wrap_distribution(dist::Distributions.UnivariateDistribution) -> AbstractTeaDistribution

Wrap a Distributions.jl **univariate** distribution instance so it satisfies
the [`register_distribution`](@ref) builder contract (`UncertainTea.logpdf` +
`Random.rand` on an `AbstractTeaDistribution`). Requires the optional
`Distributions` dependency to be loaded (which activates the
`UncertainTeaDistributionsExt` package extension).

The extension also adds the positional registration one-liner

    register_distribution(family::Symbol, constructor; support=:auto)

where `constructor` is a Distributions.jl distribution type (or any callable
returning one); the model-side arguments are forwarded to it and the result is
wrapped automatically, so

```julia
import Distributions
register_distribution(:skewnormal, Distributions.SkewNormal)

@tea static function robust_location()
    mu ~ skewnormal(0.0, 1.0, 4.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end
```

works end to end (generate/logjoint/NUTS on the CPU reference path).

`support` picks the unconstrained parameterization used when the family
appears as a LATENT:

- `:auto` (default) — read `Distributions.support(constructor)` when the
  support is a fixed property of the type and map it to the matching
  transform: `(-Inf, Inf)` → identity, `(0, Inf)` → log, `(0, 1)` → logit,
  other finite/half-infinite bounds → the bounded transforms. Discrete
  families register observation-only (no transform). If the support depends
  on the parameters (e.g. `Uniform`) `:auto` raises an `ArgumentError` asking
  for an explicit choice.
- `:real`, `:positive`, `:unit` — force the identity / log / logit transform.
- `(lower, upper)` — a bounded interval (either bound may be infinite).
- `:none` — observation-only: a latent draw from the family gets no parameter
  slot, exactly like `register_distribution(...; transform=nothing)`.

Honest limitations: univariate only (multivariate/matrix-variate families are
rejected); discrete families work as observations but not as gradient-sampled
latents (see `gibbs`/`marginalize` for discrete structure); the wrapped
density evaluates through Distributions.jl's generic `logpdf`, so latent use
additionally requires that `logpdf` to be ForwardDiff-differentiable (true
for the standard continuous families). Like every registered family, wrapped
distributions run on the CPU reference path and are honestly reported
unsupported by `backend_report`/`device_report`.
"""
function wrap_distribution end
