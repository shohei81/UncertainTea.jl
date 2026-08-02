module UncertainTeaDistributionsExt

# Distributions.jl adapter (issue #340): any Distributions.jl UNIVARIATE
# distribution already implements the logpdf/rand pair that
# `register_distribution` builders must provide — this extension closes the
# remaining gap (the `AbstractTeaDistribution` supertype and the support ->
# transform mapping) so one line registers a Distributions.jl family for use
# inside `@tea` models. See the `wrap_distribution` docstring in
# src/distributions/registration.jl for the user-facing contract.

using UncertainTea
using UncertainTea:
    AbstractParameterTransform, IdentityTransform, LogTransform, LogitTransform,
    BoundedTransform, LowerBoundedTransform, UpperBoundedTransform
import Distributions
import Random

# The builder-contract shim: subtype `AbstractTeaDistribution`, delegate
# `logpdf`/`rand` to Distributions.jl. `Distributions.logpdf` already returns
# -Inf outside the support and is generic over `Real` (so ForwardDiff Duals
# flow through both the distribution parameters and the evaluation point for
# the standard continuous families).
struct WrappedUnivariateDistribution{D<:Distributions.UnivariateDistribution} <:
       UncertainTea.AbstractTeaDistribution
    dist::D
end

UncertainTea.logpdf(wrapped::WrappedUnivariateDistribution, x) = Distributions.logpdf(wrapped.dist, x)

Random.rand(rng::Random.AbstractRNG, wrapped::WrappedUnivariateDistribution) = rand(rng, wrapped.dist)

Base.show(io::IO, wrapped::WrappedUnivariateDistribution) =
    print(io, "wrap_distribution(", wrapped.dist, ")")

UncertainTea.wrap_distribution(dist::Distributions.UnivariateDistribution) =
    WrappedUnivariateDistribution(dist)

# Honest limitation: multivariate/matrix-variate values do not fit the scalar
# parameter-slot / transform machinery the registration path relies on.
function UncertainTea.wrap_distribution(dist::Distributions.Distribution)
    throw(
        ArgumentError(
            "wrap_distribution supports univariate distributions only, got $(typeof(dist)); " *
            "multivariate and matrix-variate families are out of scope for the Distributions.jl adapter",
        ),
    )
end

# --- support -> transform mapping --------------------------------------------

_is_unit_interval(lower::Real, upper::Real) = iszero(lower) && isone(upper)

function _interval_transform(lower::Real, upper::Real)
    lower < upper || throw(ArgumentError("support requires lower < upper, got ($lower, $upper)"))
    isinf(lower) && isinf(upper) && return IdentityTransform()
    if isinf(upper)
        return iszero(lower) ? LogTransform() : LowerBoundedTransform(lower)
    end
    isinf(lower) && return UpperBoundedTransform(upper)
    return _is_unit_interval(lower, upper) ? LogitTransform() : BoundedTransform(lower, upper)
end

_auto_support_error(constructor, reason) = throw(
    ArgumentError(
        "cannot infer the latent support of `$constructor` $reason; pass " *
        "support=:real / :positive / :unit / (lower, upper) explicitly, or " *
        "support=:none for observation-only use",
    ),
)

function _auto_transform(constructor)
    constructor isa Type ||
        _auto_support_error(constructor, "because it is not a distribution type")
    # Discrete families register observation-only: the gradient samplers cannot
    # move a discrete latent, matching how transform-less registrations behave.
    constructor <: Distributions.DiscreteUnivariateDistribution && return nothing
    constructor <: Distributions.ContinuousUnivariateDistribution ||
        _auto_support_error(constructor, "because it is not a univariate distribution type")
    interval = try
        # Defined on the TYPE only when the support is parameter-independent
        # (e.g. Normal/Gamma/Beta yes; Uniform no).
        Distributions.support(constructor)
    catch err
        err isa MethodError ||
            rethrow()
        _auto_support_error(constructor, "from its type (the support depends on the parameters)")
    end
    return _interval_transform(minimum(interval), maximum(interval))
end

function _support_transform(support, constructor)
    support isa AbstractParameterTransform && return support
    support isa Tuple{Real,Real} && return _interval_transform(support...)
    support === :auto && return _auto_transform(constructor)
    support === :none && return nothing
    support === :real && return IdentityTransform()
    support === :positive && return LogTransform()
    (support === :unit || support === :unit_interval) && return LogitTransform()
    throw(
        ArgumentError(
            "unknown support $(repr(support)); expected :auto, :real, :positive, :unit, " *
            "(lower, upper), :none, or a parameter transform",
        ),
    )
end

# --- the registration one-liner -----------------------------------------------

# Positional method added to the core `register_distribution` (whose kwarg-only
# method stays the generic entry point): build the wrapping builder and map the
# support to a transform. `constructor` is typically a Distributions.jl
# distribution TYPE, but any callable returning a `UnivariateDistribution`
# works (e.g. `args -> Distributions.truncated(Distributions.Normal(args...), 0, Inf)`
# with an explicit `support`).
function UncertainTea.register_distribution(family::Symbol, constructor; support=:auto)
    constructor isa Union{Function,Type} || throw(
        ArgumentError(
            "the constructor for `$family` must be a Distributions.jl distribution type " *
            "or a callable returning one, got $(typeof(constructor))",
        ),
    )
    transform = _support_transform(support, constructor)
    builder = (args...) -> UncertainTea.wrap_distribution(constructor(args...))
    return UncertainTea.register_distribution(family; builder=builder, transform=transform)
end

end # module
