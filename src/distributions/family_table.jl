# Single-source table of the built-in distribution families (issue #331,
# stage 1 [R3-C9]). Every hand-maintained family allowlist is DERIVED from
# this table at include time, so adding a family means editing one row here
# instead of hunting down scattered symbol lists:
#
#   - `BUILTIN_DISTRIBUTION_FAMILIES` (distributions/registration.jl)
#   - `_KNOWN_DISTRIBUTION_FAMILIES`, `_BROADCAST_DISTRIBUTION_FAMILIES`, and
#     the `_qualify_builtin_distribution` guard (frontend.jl)
#   - the scalar-latent membership guard of `_supported_distribution_family`
#     and the identity/log/logit transform groups (frontend.jl, ir.jl)
#   - `_MIXTURE_REAL_LINE_FAMILIES` (ir.jl)
#   - `GPU_BACKEND_SUPPORTED_DISTRIBUTIONS` (backend/lowering/continuous.jl)
#
# Stage 2 of #331 additionally generates the per-family loop bodies from the
# `step`/`params`/`offsupport` fields below, on top of the issue-#285
# single-source scalar kernels (`_backend_<family>_logpdf(params..., value)` +
# `_<family>_logpdf_partials(params..., value)` with standardized
# `(dvalue, dparams...)` channel order):
#
#   - the `_accumulate_<family>_gradient!` loops and the per-family
#     `_score_backend_step_and_gradient!` wrappers (batched/gradients/continuous.jl)
#   - the per-family scalar/batched/observed-loop scoring methods
#     (backend/scoring/continuous.jl)
#   - the one-/two-argument `_lower_device_step!` wrappers
#     (device/plan/lowering.jl)
#
# Families whose bodies deviate from the common template stay hand-written at
# the consuming site with a comment saying why (weibull's issue-#86 boundary
# scale channel, uniform's value-channel-free partials, laplace's batched
# binding write, the noncentered-normal z-space step).
# test/uncertaintea/core/family_table_consistency.jl pins the memberships and
# the coherence invariants between the flags.
#
# This file is included before ir.jl, so it must not reference any other
# package code.

struct DistributionFamilySpec
    # the lowercase constructor-function name used inside `@tea` models
    family::Symbol
    # `@tea` recognizes the name as a distribution constructor: capitalized
    # misspellings are corrected to it, and dot-call broadcasts are vetted
    # against it (`_KNOWN_DISTRIBUTION_FAMILIES`)
    known::Bool
    # `@tea` splices the package-qualified constructor for the bare name
    # (the `_qualify_builtin_distribution` guard); the families that are
    # known but NOT macro-qualified resolve through the user's scope
    macro_qualified::Bool
    # may be dot-called as a broadcast observation, e.g. `normal.(mu, sigma)`
    # (`_BROADCAST_DISTRIBUTION_FAMILIES`)
    broadcastable::Bool
    # static-mode latent parameter-slot support:
    #   :none        observation-only, or special-cased outside the table (iid)
    #   :scalar      unconditional scalar slot (`_supported_distribution_family`
    #                accepts the family with no static-argument checks)
    #   :conditional a slot exists only when static arguments (bounds, sizes,
    #                component families) can be read off at macro time; the
    #                per-family checks stay hand-written in frontend.jl/ir.jl
    latent::Symbol
    # unconstrained-parameterization kind for a latent slot: :identity, :log,
    # or :logit for :scalar latents (these three groups are derivation
    # sources); :bounded, :lowerbounded, :simplex, :vector, :choleskycov,
    # :choleskycorr, :truncated, or :mixture for :conditional latents
    # (informational -- the transform construction needs the static
    # arguments); `nothing` when `latent === :none`
    transform::Union{Symbol,Nothing}
    # eligible as a component of a latent `mixture(...)`: a real-line
    # location-scale family, so IdentityTransform is exact
    # (`_MIXTURE_REAL_LINE_FAMILIES`)
    mixture_component::Bool
    # lowered by the batched CPU/GPU backend
    # (`GPU_BACKEND_SUPPORTED_DISTRIBUTIONS`)
    backend::Bool
    # lowered by the Metal device plan (has a `_lower_device_step!` wrapper
    # in device/plan/lowering.jl); recorded, not yet derived (see header)
    device::Bool
    # -- stage 2 (#331) generation metadata -------------------------------
    # CamelCase stem of the plan-step types, `Backend<step>ChoicePlanStep` /
    # `Device<step>ChoiceStep`; `nothing` for families whose per-site bodies
    # are not generated from the table
    step::Union{Symbol,Nothing}
    # plan-step parameter FIELD names, in the kernel argument order
    # `_backend_<family>_logpdf(params..., value)`; empty when not generated
    params::Tuple{Vararg{Symbol}}
    # gradient-path off-support predicate over `value` and the params (the
    # issue-#343 poison guard of the `_accumulate_<family>_gradient!` loop);
    # `nothing` means no guard (full-support family) or that the accumulate
    # loop stays hand-written (weibull, uniform)
    offsupport::Union{Expr,Nothing}
end

function _family_spec(
    family::Symbol;
    known::Bool=true,
    macro_qualified::Bool=known,
    broadcastable::Bool=false,
    latent::Symbol=:none,
    transform::Union{Symbol,Nothing}=nothing,
    mixture_component::Bool=false,
    backend::Bool=false,
    device::Bool=false,
    step::Union{Symbol,Nothing}=nothing,
    params::Tuple{Vararg{Symbol}}=(),
    offsupport::Union{Expr,Nothing}=nothing,
)
    return DistributionFamilySpec(
        family, known, macro_qualified, broadcastable, latent, transform,
        mixture_component, backend, device, step, params, offsupport,
    )
end

# Row order is load-bearing for exactly one derived list:
# `GPU_BACKEND_SUPPORTED_DISTRIBUTIONS` keeps its historical order (the
# gradient-crosscheck suite derives RNG seeds from `enumerate` over it), so
# the `backend=true` rows appear in that order and the backend-unsupported
# families are interleaved next to their relatives. Every other derived list
# is consumed membership-only.
const DISTRIBUTION_FAMILY_TABLE = (
    _family_spec(
        :normal;
        broadcastable=true,
        latent=:scalar,
        transform=:identity,
        mixture_component=true,
        backend=true,
        device=true,
        step=:Normal,
        params=(:mu, :sigma),
    ),
    _family_spec(
        :lognormal;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:Lognormal,
        params=(:mu, :sigma),
        # subnormal values are the exp-underflow boundary (issues #345/#367):
        # the kernel scores them -Inf, so poison alongside the exact-0.0 case
        offsupport=:(!(value > 0) || issubnormal(value)),
    ),
    _family_spec(
        :laplace;
        latent=:scalar,
        transform=:identity,
        mixture_component=true,
        backend=true,
        device=true,
        step=:Laplace,
        params=(:mu, :scale),
    ),
    _family_spec(
        :exponential;
        broadcastable=true,
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:Exponential,
        params=(:rate,),
        offsupport=:(!(value >= 0)),
    ),
    _family_spec(
        :gamma;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:Gamma,
        params=(:shape, :rate),
        # subnormal values are the exp-underflow boundary (issue #345): the
        # kernel scores them -Inf, so poison alongside the exact-0.0 case
        offsupport=:(!(value > 0) || issubnormal(value)),
    ),
    _family_spec(
        :inversegamma;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:InverseGamma,
        params=(:shape, :scale),
        # subnormal boundary as in gamma (issues #345/#367)
        offsupport=:(!(value > 0) || issubnormal(value)),
    ),
    # weibull's accumulate loop stays hand-written (issue #86 x==0/shape==1
    # boundary), so no offsupport here; step/params still drive the scoring,
    # gradient-wrapper, and device-lowering generation
    _family_spec(
        :weibull;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:Weibull,
        params=(:shape, :scale),
    ),
    _family_spec(
        :beta;
        latent=:scalar,
        transform=:logit,
        backend=true,
        device=true,
        step=:Beta,
        params=(:alpha, :beta),
        offsupport=:(!(0 < value < 1)),
    ),
    _family_spec(:dirichlet; latent=:conditional, transform=:simplex, backend=true, device=true),
    _family_spec(:bernoulli; broadcastable=true, backend=true, device=true, step=:Bernoulli, params=(:probability,)),
    _family_spec(:bernoullilogit; broadcastable=true, backend=true, device=true),
    _family_spec(:binomial; backend=true, device=true, step=:Binomial, params=(:trials, :probability)),
    _family_spec(:betabinomial; backend=true),
    _family_spec(:multinomial),
    _family_spec(:discreteuniform; backend=true),
    _family_spec(:geometric; backend=true, device=true, step=:Geometric, params=(:probability,)),
    _family_spec(
        :negativebinomial;
        backend=true,
        device=true,
        step=:NegativeBinomial,
        params=(:successes, :probability),
    ),
    _family_spec(:poisson; broadcastable=true, backend=true, device=true, step=:Poisson, params=(:lambda,)),
    _family_spec(
        :studentt;
        broadcastable=true,
        latent=:scalar,
        transform=:identity,
        mixture_component=true,
        backend=true,
        device=true,
        step=:StudentT,
        params=(:nu, :mu, :sigma),
    ),
    _family_spec(:categorical; backend=true, device=true),
    _family_spec(:mvnormal; latent=:conditional, transform=:vector, backend=true, device=true),
    _family_spec(:truncatednormal; latent=:conditional, transform=:truncated, backend=true, device=true),
    _family_spec(:truncatedstudentt; latent=:conditional, transform=:truncated, backend=true, device=true),
    _family_spec(:mixture; latent=:conditional, transform=:mixture, backend=true, device=true),
    _family_spec(:mvnormaldense; latent=:conditional, transform=:vector, backend=true, device=true),
    _family_spec(:mvstudentt; latent=:conditional, transform=:vector, backend=true, device=true),
    _family_spec(:mvstudenttdense; latent=:conditional, transform=:vector, backend=true, device=true),
    _family_spec(:wishart; latent=:conditional, transform=:choleskycov),
    _family_spec(:inversewishart; latent=:conditional, transform=:choleskycov),
    _family_spec(:lkjcholesky; latent=:conditional, transform=:choleskycorr, backend=true, device=true),
    _family_spec(:gaussianprocess; known=false),
    _family_spec(:sparsegaussianprocess; known=false),
    _family_spec(:hmm; known=false),
    _family_spec(:orderedlogistic; known=false),
    _family_spec(:zeroinflatedpoisson; known=false),
    _family_spec(:zeroinflatednegativebinomial; known=false),
    _family_spec(:vonmises; known=false),
    _family_spec(:iid; known=false, macro_qualified=true),
    _family_spec(
        :cauchy;
        latent=:scalar,
        transform=:identity,
        backend=true,
        device=true,
        step=:Cauchy,
        params=(:mu, :sigma),
    ),
    _family_spec(
        :halfnormal;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:HalfNormal,
        params=(:sigma,),
        offsupport=:(value < 0),
    ),
    _family_spec(
        :halfcauchy;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:HalfCauchy,
        params=(:scale,),
        offsupport=:(value < 0),
    ),
    # uniform's accumulate loop stays hand-written (the partials kernel has no
    # value channel), so no offsupport here; step/params drive the scoring
    # generation only (not device-lowered)
    _family_spec(
        :uniform;
        latent=:conditional,
        transform=:bounded,
        backend=true,
        step=:Uniform,
        params=(:lower, :upper),
    ),
    _family_spec(
        :logistic;
        latent=:scalar,
        transform=:identity,
        backend=true,
        device=true,
        step=:Logistic,
        params=(:mu, :scale),
    ),
    _family_spec(
        :gumbel;
        latent=:scalar,
        transform=:identity,
        backend=true,
        device=true,
        step=:Gumbel,
        params=(:mu, :scale),
    ),
    _family_spec(
        :pareto;
        latent=:conditional,
        transform=:lowerbounded,
        backend=true,
        step=:Pareto,
        params=(:xm, :alpha),
        offsupport=:(!(xm > 0 && alpha > 0 && value >= xm)),
    ),
    _family_spec(
        :frechet;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:Frechet,
        params=(:shape, :scale),
        # subnormal boundary as in gamma (issues #345/#367)
        offsupport=:(!(shape > 0 && scale > 0 && value > 0) || issubnormal(value)),
    ),
    _family_spec(
        :rayleigh;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:Rayleigh,
        params=(:scale,),
        # subnormal boundary as in gamma (issues #345/#367)
        offsupport=:(!(scale > 0 && value > 0) || issubnormal(value)),
    ),
    _family_spec(
        :inversegaussian;
        latent=:scalar,
        transform=:log,
        backend=true,
        device=true,
        step=:InverseGaussian,
        params=(:mu, :lambda),
        # subnormal boundary as in gamma (issues #345/#367)
        offsupport=:(!(mu > 0 && lambda > 0 && value > 0) || issubnormal(value)),
    ),
)

# Family symbols of the table rows matching `pred`, in table order.
_distribution_families(pred) = Tuple(spec.family for spec in DISTRIBUTION_FAMILY_TABLE if pred(spec))

# The unique table row for `family` (include-time lookup for the stage-2
# generation loops; throws on a family the table does not carry).
function _distribution_family_spec(family::Symbol)
    for spec in DISTRIBUTION_FAMILY_TABLE
        spec.family === family && return spec
    end
    throw(ArgumentError("no DISTRIBUTION_FAMILY_TABLE row for family `$family`"))
end

# `@tea` splices the package-qualified constructor for these bare names
# (guard of `_qualify_builtin_distribution` in frontend.jl).
const _MACRO_QUALIFIED_FAMILIES = _distribution_families(spec -> spec.macro_qualified)

# Scalar latents with an unconditional parameter slot (the plain-membership
# arm of `_supported_distribution_family` in frontend.jl), split by the
# unconstrained parameterization used in frontend.jl `_parameter_transform_expr`
# and ir.jl `_parameter_transform` / `_iid_parameter_transform`.
const _IDENTITY_TRANSFORM_LATENT_FAMILIES =
    _distribution_families(spec -> spec.latent === :scalar && spec.transform === :identity)
const _LOG_TRANSFORM_LATENT_FAMILIES =
    _distribution_families(spec -> spec.latent === :scalar && spec.transform === :log)
const _LOGIT_TRANSFORM_LATENT_FAMILIES =
    _distribution_families(spec -> spec.latent === :scalar && spec.transform === :logit)
const _SCALAR_TRANSFORM_LATENT_FAMILIES = _distribution_families(spec -> spec.latent === :scalar)

# Real-line location-scale families eligible as latent mixture components.
const _MIXTURE_REAL_LINE_FAMILIES = _distribution_families(spec -> spec.mixture_component)
