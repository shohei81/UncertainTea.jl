# Device execution plan: an `isbits` mirror of the second-stage BackendExecutionPlan.
#
# Design notes
# ------------
# * Addresses are fully ERASED. Observed choice values are resolved on the host
#   (staging.jl) into a dense `observed[row, col]` matrix; the kernel walks the
#   plan maintaining an observation cursor that advances identically to staging.
# * Primitive operators are lifted to a type parameter (`DevicePrimitiveExpr{Op}`,
#   `Op::Symbol`) so the kernel dispatches at compile time with no runtime branch.
# * `Union{Nothing,Int}` is avoided everywhere (it is not an isbits layout). Slots
#   use `Int32` with `0` meaning "no binding". A choice step's `value_source::Int32`
#   encodes the value origin: `> 0` -> unconstrained parameter row (latent);
#   `< 0` -> observed (value comes from the staged observation cursor).
# * Loops are non-nested and carry a static `loop_id::Int32`; the trip count and
#   start index are staged per loop into device Int32 buffers.

# ---- device expressions --------------------------------------------------------

abstract type AbstractDeviceExpr end

struct DeviceLiteralExpr{T} <: AbstractDeviceExpr
    value::T
end

struct DeviceSlotExpr <: AbstractDeviceExpr
    slot::Int32
end

struct DevicePrimitiveExpr{Op,A<:Tuple} <: AbstractDeviceExpr
    args::A
end

DevicePrimitiveExpr(op::Symbol, args::Tuple) = DevicePrimitiveExpr{op,typeof(args)}(args)

# ---- device plan steps ---------------------------------------------------------

abstract type AbstractDevicePlanStep end
abstract type AbstractDeviceChoiceStep <: AbstractDevicePlanStep end

struct DeviceNormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# reparam=:noncentered normal: z read raw from the unconstrained params,
# scored against N(0, 1); the binding materializes theta = mu + sigma * z.
struct DeviceNoncenteredNormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    binding_slot::Int32
end

struct DeviceLognormalChoiceStep{M,S} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceExponentialChoiceStep{R} <: AbstractDeviceChoiceStep
    rate::R
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceGammaChoiceStep{SH,R} <: AbstractDeviceChoiceStep
    shape::SH
    rate::R
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceLaplaceChoiceStep{M,S} <: AbstractDeviceChoiceStep
    loc::M
    scale::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceBetaChoiceStep{A,B} <: AbstractDeviceChoiceStep
    alpha::A
    beta::B
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceStudentTChoiceStep{N,M,S} <: AbstractDeviceChoiceStep
    nu::N
    mu::M
    sigma::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceInverseGammaChoiceStep{SH,SC} <: AbstractDeviceChoiceStep
    shape::SH
    scale::SC
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceWeibullChoiceStep{SH,SC} <: AbstractDeviceChoiceStep
    shape::SH
    scale::SC
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceBernoulliChoiceStep{P} <: AbstractDeviceChoiceStep
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Logit-parameterized Bernoulli with a generic scalar `eta` (issues #135/#149):
# `bernoullilogit(scalar_expr)`. Scored in the stable `x*eta - log1p(exp(eta))`
# form. bernoullilogit latents are unsupported (backend rejects them), so
# `value_source` is always the staged observation (-1); the transform field is
# carried for the choice-step convention and stays identity.
struct DeviceBernoulliLogitChoiceStep{E<:AbstractDeviceExpr} <: AbstractDeviceChoiceStep
    eta::E
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Logit-parameterized Bernoulli whose eta is the fused GLM linear predictor
# `intercept + sum_d coef[d] * X[d, index]` (issues #135/#150 -- the device analog
# of the HOST batched analytic lowering). `D` is the compile-time coefficient
# dimension (the backend guarantees a VectorIdentity latent vector), unrolling the
# dot product in the kernel body. The covariate column `X[:, index]` is
# per-observation CONSTANT data staged into the observation buffer (like observed
# values, addresses erased): staging pushes `D` covariate rows then the observed y
# row, and the kernel reads them cursor-first, so the `index` expr disappears from
# the device plan entirely. `coef_value_source` is the coefficient latent's
# UNCONSTRAINED parameter row start (VectorIdentity: constrained == unconstrained,
# log-abs-det 0); the gradient kernel seeds derivative 1 there so forward-mode
# reproduces `d_eta/d_coef[d] = X[d, index]`, and `d_eta/d_intercept = 1` flows
# through the intercept sub-expr's slot dual. `value_source` is always -1
# (bernoullilogit is observed-only).
struct DeviceBernoulliLogitGLMChoiceStep{D,I<:AbstractDeviceExpr} <: AbstractDeviceChoiceStep
    intercept::I
    coef_value_source::Int32
    value_source::Int32
    binding_slot::Int32
end

DeviceBernoulliLogitGLMChoiceStep{D}(intercept::I, coef_value_source, value_source, binding_slot) where {D,I} =
    DeviceBernoulliLogitGLMChoiceStep{D,I}(
        intercept,
        Int32(coef_value_source),
        Int32(value_source),
        Int32(binding_slot),
    )

struct DeviceBinomialChoiceStep{N,P} <: AbstractDeviceChoiceStep
    trials::N
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceGeometricChoiceStep{P} <: AbstractDeviceChoiceStep
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceNegativeBinomialChoiceStep{R,P} <: AbstractDeviceChoiceStep
    successes::R
    probability::P
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceCategoricalChoiceStep{PS<:Tuple} <: AbstractDeviceChoiceStep
    probabilities::PS
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Finite mixture of normals (issue #50): the VALUE is a scalar (Identity latent
# or observed), so this follows the group-1 scalar conventions -- the binding
# IS materialized, no audit special-case -- with per-component expression
# tuples and a log-sum-exp fold.
struct DeviceMixtureNormalChoiceStep{W<:Tuple,M<:Tuple,S<:Tuple} <: AbstractDeviceChoiceStep
    weights::W
    mus::M
    sigmas::S
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DevicePoissonChoiceStep{L} <: AbstractDeviceChoiceStep
    lambda::L
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

# Diagonal multivariate normal (issue #12 group 3, phase 1). The dimension is
# compile-time (the argument tuples' arity); a latent reads `D` consecutive
# unconstrained rows starting at `value_source` (VectorIdentity transform:
# constrained == unconstrained, log-abs-det 0), an observation consumes `D`
# staged rows. The binding slot is carried but NEVER written -- the slots
# matrix holds one scalar per symbol -- so the read/write audit treats it as
# unmaterialized and honestly rejects any downstream read.
struct DeviceMvNormalChoiceStep{M<:Tuple,S<:Tuple} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    value_source::Int32
    binding_slot::Int32
end

# Dirichlet (issue #12 group 3, phase 2): the first dimension-changing vector
# step. A latent reads K-1 consecutive unconstrained rows starting at
# `value_source` and constrains them through the register-resident shifted
# softmax (log-abs-det included); an observation consumes K staged rows. The
# binding slot is carried but never written (see DeviceMvNormalChoiceStep).
struct DeviceDirichletChoiceStep{A<:Tuple} <: AbstractDeviceChoiceStep
    alpha::A
    value_source::Int32
    binding_slot::Int32
end

# Dense multivariate normal (issue #12 group 3, phase 3). The constant
# scale_tril factor rides the observation buffer: staging packs its lower
# triangle (column-major) into d(d+1)/2 cursor rows ahead of any observed
# value rows, so the kernel needs no extra buffer or signature change and
# per-batch factors work for free. mu is a compile-time expr tuple; the value
# follows the mvnormal latent/observed conventions.
struct DeviceMvNormalDenseChoiceStep{M<:Tuple} <: AbstractDeviceChoiceStep
    mu::M
    value_source::Int32
    binding_slot::Int32
end

# LKJ prior over a packed correlation Cholesky factor (issue #57, mirroring
# the CPU-native step from issue #49). `D` is the compile-time dimension (the
# backend step guarantees a macro-time literal); a latent reads d(d-1)/2
# unconstrained rows starting at `value_source` and constrains them through
# the register-resident tanh/stick construction (log-abs-det included), an
# observation consumes d(d+1)/2 staged packed rows. The binding slot is
# carried but never written (see DeviceMvNormalChoiceStep).
struct DeviceLKJCholeskyChoiceStep{D,E} <: AbstractDeviceChoiceStep
    eta::E
    value_source::Int32
    binding_slot::Int32
end

DeviceLKJCholeskyChoiceStep{D}(eta::E, value_source, binding_slot) where {D,E} =
    DeviceLKJCholeskyChoiceStep{D,E}(eta, Int32(value_source), Int32(binding_slot))

# Compile-time-unrolled tuple folds multiply the fused kernel body by D (and
# the gradient kernel again by parameter count); cap the dimension so Metal
# shader compilation stays inside its budget (see docs/device-vector-latents.md).
const DEVICE_MAX_VECTOR_DIMENSION = 16

# The dense forward substitution unrolls d(d+1)/2 fused multiply-adds per step
# (doubled again in the gradient kernel), so its cap is tighter.
const DEVICE_MAX_DENSE_DIMENSION = 8

# The unrolled cholesky-correlation constrain and row-norm support checks are
# likewise quadratic in the kernel body, so lkjcholesky shares the dense cap.
const DEVICE_MAX_CHOLESKY_DIMENSION = 8

# Truncated families are observed-only on the backend path (latents fall back at
# backend lowering: the bounded transform is unimplemented there), so the value
# source is always the staged observation; the fields stay generic regardless.
struct DeviceTruncatedNormalChoiceStep{M,S,L,U} <: AbstractDeviceChoiceStep
    mu::M
    sigma::S
    lower::L
    upper::U
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceTruncatedStudentTChoiceStep{N,M,S,L,U} <: AbstractDeviceChoiceStep
    nu::N
    mu::M
    sigma::S
    lower::L
    upper::U
    value_source::Int32
    transform::Int32
    binding_slot::Int32
end

struct DeviceDeterministicStep{E} <: AbstractDevicePlanStep
    expr::E
    binding_slot::Int32
end

struct DeviceLoopStep{B<:Tuple} <: AbstractDevicePlanStep
    loop_id::Int32
    iterator_slot::Int32
    body::B
end

struct DeviceExecutionPlan{T,S<:Tuple}
    steps::S
    slot_count::Int32
    loop_count::Int32
end

DeviceExecutionPlan{T}(steps::S, slot_count, loop_count) where {T,S<:Tuple} =
    DeviceExecutionPlan{T,S}(steps, Int32(slot_count), Int32(loop_count))

function Base.show(io::IO, plan::DeviceExecutionPlan{T}) where {T}
    print(
        io,
        "DeviceExecutionPlan{",
        T,
        "}(steps=",
        length(plan.steps),
        ", slots=",
        plan.slot_count,
        ", loops=",
        plan.loop_count,
        ")",
    )
end

# ---- lowering ------------------------------------------------------------------

const DEVICE_SUPPORTED_PRIMITIVES = (:+, :-, :*, :/, :^, :exp, :log, :log1p, :sqrt, :abs, :min, :max, :clamp)

function _device_issue!(issues::Vector{String}, message::String)
    push!(issues, message)
    return nothing
end

_device_slot32(slot::Nothing) = Int32(0)
_device_slot32(slot::Integer) = Int32(slot)

function _device_transform_code(transform::AbstractParameterTransform, issues::Vector{String}, context::String)
    if transform isa IdentityTransform
        return DEVICE_TRANSFORM_IDENTITY
    elseif transform isa LogTransform
        return DEVICE_TRANSFORM_LOG
    elseif transform isa LogitTransform
        return DEVICE_TRANSFORM_LOGIT
    end
    _device_issue!(
        issues,
        "device lowering does not support the $(nameof(typeof(transform))) parameter transform (vector transforms such as Simplex/VectorIdentity are unsupported) in $context",
    )
    return nothing
end

function _lower_device_expr(expr::BackendLiteralExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    value = expr.value
    if value isa Real && !(value isa Bool)
        return DeviceLiteralExpr(convert(T, value))
    end
    _device_issue!(issues, "device lowering only supports real numeric literals, got $(typeof(value)) in $context")
    return nothing
end

function _lower_device_expr(expr::BackendSlotExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    if generic_slots[expr.slot]
        _device_issue!(
            issues,
            "device lowering cannot feed generic (non-numeric) slot $(expr.slot) into a numeric expression in $context",
        )
        return nothing
    end
    return DeviceSlotExpr(Int32(expr.slot))
end

function _lower_device_expr(expr::BackendPrimitiveExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    if !(expr.op in DEVICE_SUPPORTED_PRIMITIVES)
        _device_issue!(issues, "device lowering does not support primitive `$(expr.op)` in $context")
        return nothing
    end
    lowered = map(arg -> _lower_device_expr(arg, generic_slots, T, issues, context), expr.arguments)
    any(isnothing, lowered) && return nothing
    return DevicePrimitiveExpr(expr.op, tuple(lowered...))
end

function _lower_device_expr(expr::BackendBlockExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    if length(expr.arguments) == 1
        return _lower_device_expr(first(expr.arguments), generic_slots, T, issues, context)
    end
    _device_issue!(issues, "device lowering does not support multi-statement block expressions in $context")
    return nothing
end

function _lower_device_expr(expr::AbstractBackendExpr, generic_slots, ::Type{T}, issues::Vector{String}, context::String) where {T}
    _device_issue!(issues, "device lowering does not support $(nameof(typeof(expr))) in $context")
    return nothing
end

# Latent value-source / transform, or observed marker. Returns (value_source, transform_code)
# or nothing (issue pushed).
function _device_choice_value_source(
    step::BackendChoicePlanStep,
    layout::ParameterLayout,
    in_loop::Bool,
    issues::Vector{String},
    family::String,
)
    if isnothing(step.parameter_slot)
        return (Int32(-1), DEVICE_TRANSFORM_IDENTITY)
    end
    if in_loop
        _device_issue!(issues, "device lowering does not support latent $family choices inside a loop")
        return nothing
    end
    # backend steps carry the slot's VALUE row (issue #36), so recover the
    # slot spec by that row rather than indexing by ordinal
    slot_position = findfirst(s -> s.value_index == step.parameter_slot, layout.slots)
    if isnothing(slot_position)
        _device_issue!(issues, "device lowering could not resolve the parameter slot for a $family latent")
        return nothing
    end
    slot = layout.slots[slot_position]
    if slot.value_length != 1 || slot.dimension != 1
        _device_issue!(issues, "device lowering only supports scalar latent parameters, got a vector $family latent")
        return nothing
    end
    tcode = _device_transform_code(slot.transform, issues, "$family latent")
    isnothing(tcode) && return nothing
    return (Int32(slot.index), tcode)
end

# ---- per-family choice lowering ----

function _lower_device_step!(
    out,
    step::BackendNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.mu, step.sigma, DeviceNormalChoiceStep, backend, layout, T, issues, in_loop, "normal")
end
function _lower_device_step!(
    out,
    step::BackendLognormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.mu,
        step.sigma,
        DeviceLognormalChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "lognormal",
    )
end
function _lower_device_step!(
    out,
    step::BackendGammaChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.shape, step.rate, DeviceGammaChoiceStep, backend, layout, T, issues, in_loop, "gamma")
end
function _lower_device_step!(
    out,
    step::BackendLaplaceChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.mu, step.scale, DeviceLaplaceChoiceStep, backend, layout, T, issues, in_loop, "laplace")
end
function _lower_device_step!(
    out,
    step::BackendBetaChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(out, step, step.alpha, step.beta, DeviceBetaChoiceStep, backend, layout, T, issues, in_loop, "beta")
end

function _lower_device_step!(
    out,
    step::BackendInverseGammaChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.shape,
        step.scale,
        DeviceInverseGammaChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "inversegamma",
    )
end
function _lower_device_step!(
    out,
    step::BackendWeibullChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.shape,
        step.scale,
        DeviceWeibullChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "weibull",
    )
end
function _lower_device_step!(
    out,
    step::BackendBinomialChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.trials,
        step.probability,
        DeviceBinomialChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "binomial",
    )
end
function _lower_device_step!(
    out,
    step::BackendNegativeBinomialChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    _lower_device_two_arg!(
        out,
        step,
        step.successes,
        step.probability,
        DeviceNegativeBinomialChoiceStep,
        backend,
        layout,
        T,
        issues,
        in_loop,
        "negativebinomial",
    )
end
function _lower_device_step!(
    out,
    step::BackendStudentTChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "studentt")
    nu = _lower_device_expr(step.nu, backend.generic_slots, T, issues, "studentt argument")
    mu = _lower_device_expr(step.mu, backend.generic_slots, T, issues, "studentt argument")
    sigma = _lower_device_expr(step.sigma, backend.generic_slots, T, issues, "studentt argument")
    (isnothing(src) || isnothing(nu) || isnothing(mu) || isnothing(sigma)) && return nothing
    value_source, tcode = src
    push!(out, DeviceStudentTChoiceStep(nu, mu, sigma, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end
function _lower_device_step!(
    out,
    step::BackendGeometricChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "geometric")
    p = _lower_device_expr(step.probability, backend.generic_slots, T, issues, "geometric argument")
    (isnothing(src) || isnothing(p)) && return nothing
    value_source, tcode = src
    push!(out, DeviceGeometricChoiceStep(p, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end
function _lower_device_step!(
    out,
    step::BackendCategoricalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "categorical")
    probabilities = map(
        probability -> _lower_device_expr(probability, backend.generic_slots, T, issues, "categorical argument"),
        step.probabilities,
    )
    (isnothing(src) || any(isnothing, probabilities)) && return nothing
    value_source, tcode = src
    push!(out, DeviceCategoricalChoiceStep(probabilities, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendTruncatedNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "truncatednormal")
    mu = _lower_device_expr(step.mu, backend.generic_slots, T, issues, "truncatednormal argument")
    sigma = _lower_device_expr(step.sigma, backend.generic_slots, T, issues, "truncatednormal argument")
    lower = _lower_device_expr(step.lower, backend.generic_slots, T, issues, "truncatednormal bound")
    upper = _lower_device_expr(step.upper, backend.generic_slots, T, issues, "truncatednormal bound")
    (isnothing(src) || isnothing(mu) || isnothing(sigma) || isnothing(lower) || isnothing(upper)) && return nothing
    value_source, tcode = src
    push!(
        out,
        DeviceTruncatedNormalChoiceStep(mu, sigma, lower, upper, value_source, tcode, _device_slot32(step.binding_slot)),
    )
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendTruncatedStudentTChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "truncatedstudentt")
    # backend lowering guarantees a literal nu (the analytic d/dnu is omitted)
    nu = _lower_device_expr(step.nu, backend.generic_slots, T, issues, "truncatedstudentt argument")
    mu = _lower_device_expr(step.mu, backend.generic_slots, T, issues, "truncatedstudentt argument")
    sigma = _lower_device_expr(step.sigma, backend.generic_slots, T, issues, "truncatedstudentt argument")
    lower = _lower_device_expr(step.lower, backend.generic_slots, T, issues, "truncatedstudentt bound")
    upper = _lower_device_expr(step.upper, backend.generic_slots, T, issues, "truncatedstudentt bound")
    (isnothing(src) || isnothing(nu) || isnothing(mu) || isnothing(sigma) || isnothing(lower) || isnothing(upper)) &&
        return nothing
    value_source, tcode = src
    push!(
        out,
        DeviceTruncatedStudentTChoiceStep(
            nu,
            mu,
            sigma,
            lower,
            upper,
            value_source,
            tcode,
            _device_slot32(step.binding_slot),
        ),
    )
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendMvNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    dimension = length(step.mu)
    if dimension > DEVICE_MAX_VECTOR_DIMENSION
        _device_issue!(
            issues,
            "device lowering caps vector dimensions at $DEVICE_MAX_VECTOR_DIMENSION (kernel compile-time budget), got an mvnormal of dimension $dimension",
        )
        return nothing
    end
    if isnothing(step.parameter_slot)
        value_source = Int32(-1)
    else
        if in_loop
            _device_issue!(issues, "device lowering does not support latent mvnormal choices inside a loop")
            return nothing
        end
        # vector backend steps carry the slot ORDINAL (scalar steps carry the
        # value row; see issue #36) -- index directly
        slot = layout.slots[step.parameter_slot]
        if !(slot.transform isa VectorIdentityTransform) || slot.dimension != dimension ||
           slot.value_length != dimension
            _device_issue!(issues, "device lowering could not resolve the mvnormal parameter slot")
            return nothing
        end
        value_source = Int32(slot.index)
    end
    mu = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mvnormal argument"), step.mu)
    sigma = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mvnormal argument"), step.sigma)
    (any(isnothing, mu) || any(isnothing, sigma)) && return nothing
    push!(out, DeviceMvNormalChoiceStep(mu, sigma, value_source, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendDirichletChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    dimension = length(step.alpha)
    if dimension > DEVICE_MAX_VECTOR_DIMENSION
        _device_issue!(
            issues,
            "device lowering caps vector dimensions at $DEVICE_MAX_VECTOR_DIMENSION (kernel compile-time budget), got a dirichlet of dimension $dimension",
        )
        return nothing
    end
    if isnothing(step.parameter_slot)
        value_source = Int32(-1)
    else
        if in_loop
            _device_issue!(issues, "device lowering does not support latent dirichlet choices inside a loop")
            return nothing
        end
        # vector backend steps carry the slot ORDINAL (see the mvnormal step)
        slot = layout.slots[step.parameter_slot]
        if !(slot.transform isa SimplexTransform) || slot.value_length != dimension ||
           slot.dimension != dimension - 1
            _device_issue!(issues, "device lowering could not resolve the dirichlet parameter slot")
            return nothing
        end
        value_source = Int32(slot.index)
    end
    alpha = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "dirichlet argument"), step.alpha)
    any(isnothing, alpha) && return nothing
    push!(out, DeviceDirichletChoiceStep(alpha, value_source, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendLKJCholeskyChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    dimension = step.d
    if dimension > DEVICE_MAX_CHOLESKY_DIMENSION
        _device_issue!(
            issues,
            "device lowering caps lkjcholesky at dimension $DEVICE_MAX_CHOLESKY_DIMENSION (the unrolled constrain is quadratic in the kernel body), got dimension $dimension",
        )
        return nothing
    end
    if isnothing(step.parameter_slot)
        value_source = Int32(-1)
    else
        if in_loop
            _device_issue!(issues, "device lowering does not support latent lkjcholesky choices inside a loop")
            return nothing
        end
        slot = layout.slots[step.parameter_slot] # vector steps carry the slot ORDINAL
        if !(slot.transform isa CholeskyCorrTransform) || slot.dimension != (dimension * (dimension - 1)) ÷ 2 ||
           slot.value_length != (dimension * (dimension + 1)) ÷ 2
            _device_issue!(issues, "device lowering could not resolve the lkjcholesky parameter slot")
            return nothing
        end
        value_source = Int32(slot.index)
    end
    eta = _lower_device_expr(step.eta, backend.generic_slots, T, issues, "lkjcholesky concentration")
    isnothing(eta) && return nothing
    push!(out, DeviceLKJCholeskyChoiceStep{dimension}(eta, value_source, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendMvNormalDenseChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    dimension = length(step.mu)
    if dimension > DEVICE_MAX_DENSE_DIMENSION
        _device_issue!(
            issues,
            "device lowering caps mvnormaldense at dimension $DEVICE_MAX_DENSE_DIMENSION (the unrolled forward substitution is quadratic in the kernel body), got dimension $dimension",
        )
        return nothing
    end
    if isnothing(step.parameter_slot)
        value_source = Int32(-1)
    else
        if in_loop
            _device_issue!(issues, "device lowering does not support latent mvnormaldense choices inside a loop")
            return nothing
        end
        slot = layout.slots[step.parameter_slot] # vector steps carry the slot ORDINAL
        if !(slot.transform isa VectorIdentityTransform) || slot.dimension != dimension ||
           slot.value_length != dimension
            _device_issue!(issues, "device lowering could not resolve the mvnormaldense parameter slot")
            return nothing
        end
        value_source = Int32(slot.index)
    end
    mu = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mvnormaldense argument"), step.mu)
    any(isnothing, mu) && return nothing
    # scale_tril stays a host-side generic slot; staging packs its lower
    # triangle into the observation buffer, so nothing lowers here
    push!(out, DeviceMvNormalDenseChoiceStep(mu, value_source, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendMixtureNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    components = length(step.weights)
    if components > DEVICE_MAX_VECTOR_DIMENSION
        _device_issue!(
            issues,
            "device lowering caps mixture components at $DEVICE_MAX_VECTOR_DIMENSION (kernel compile-time budget), got $components",
        )
        return nothing
    end
    src = _device_choice_value_source(step, layout, in_loop, issues, "mixture")
    weights = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mixture weight"), step.weights)
    mus = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mixture argument"), step.mus)
    sigmas = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mixture argument"), step.sigmas)
    (isnothing(src) || any(isnothing, weights) || any(isnothing, mus) || any(isnothing, sigmas)) && return nothing
    value_source, tcode = src
    push!(
        out,
        DeviceMixtureNormalChoiceStep(weights, mus, sigmas, value_source, tcode, _device_slot32(step.binding_slot)),
    )
    return nothing
end

function _lower_device_two_arg!(
    out,
    step,
    arg1_expr,
    arg2_expr,
    Ctor,
    backend,
    layout,
    ::Type{T},
    issues,
    in_loop,
    family,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, family)
    a1 = _lower_device_expr(arg1_expr, backend.generic_slots, T, issues, "$family argument")
    a2 = _lower_device_expr(arg2_expr, backend.generic_slots, T, issues, "$family argument")
    (isnothing(src) || isnothing(a1) || isnothing(a2)) && return nothing
    value_source, tcode = src
    push!(out, Ctor(a1, a2, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendExponentialChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "exponential")
    rate = _lower_device_expr(step.rate, backend.generic_slots, T, issues, "exponential argument")
    (isnothing(src) || isnothing(rate)) && return nothing
    value_source, tcode = src
    push!(out, DeviceExponentialChoiceStep(rate, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendBernoulliChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "bernoulli")
    p = _lower_device_expr(step.probability, backend.generic_slots, T, issues, "bernoulli argument")
    (isnothing(src) || isnothing(p)) && return nothing
    value_source, tcode = src
    push!(out, DeviceBernoulliChoiceStep(p, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

# Resolve the linear predictor coefficient's UNCONSTRAINED parameter row start
# from the backend expr's constrained value_index/length. The coefficient must be
# a VectorIdentity latent vector (the same guard the host analytic lowering #150
# uses); a non-identity coefficient makes the device fall back honestly.
function _device_linear_predictor_coef_source(eta::BackendLinearPredictorExpr, layout::ParameterLayout, issues::Vector{String})
    slot_position =
        findfirst(s -> s.value_index == eta.coef_value_index && s.value_length == eta.coef_length, layout.slots)
    if isnothing(slot_position)
        _device_issue!(issues, "device lowering could not resolve the linear predictor coefficient parameter slot")
        return nothing
    end
    slot = layout.slots[slot_position]
    if !(slot.transform isa VectorIdentityTransform) || slot.dimension != eta.coef_length
        _device_issue!(
            issues,
            "device lowering only supports identity-transform (VectorIdentity) linear predictor coefficients; non-identity coefficients fall back",
        )
        return nothing
    end
    return Int32(slot.index)
end

# bernoullilogit (issues #135/#149/#150). Latents are unsupported (the backend
# rejects them), so the value is always the staged observation. When `eta` is the
# fused GLM linear predictor, lower to the analytic device GLM step (covariate
# column staged into the observation buffer); otherwise lower the generic scalar
# eta expr.
function _lower_device_step!(
    out,
    step::BackendBernoulliLogitChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    if !isnothing(step.parameter_slot)
        _device_issue!(issues, "device lowering does not support bernoullilogit latent parameters")
        return nothing
    end
    if step.eta isa BackendLinearPredictorExpr
        eta = step.eta
        if eta.coef_length > DEVICE_MAX_VECTOR_DIMENSION
            _device_issue!(
                issues,
                "device lowering caps the linear predictor coefficient dimension at $DEVICE_MAX_VECTOR_DIMENSION (kernel compile-time budget), got $(eta.coef_length)",
            )
            return nothing
        end
        coef_source = _device_linear_predictor_coef_source(eta, layout, issues)
        isnothing(coef_source) && return nothing
        intercept =
            isnothing(eta.intercept) ? DeviceLiteralExpr(zero(T)) :
            _lower_device_expr(eta.intercept, backend.generic_slots, T, issues, "bernoullilogit linear predictor intercept")
        isnothing(intercept) && return nothing
        push!(
            out,
            DeviceBernoulliLogitGLMChoiceStep{eta.coef_length}(
                intercept,
                coef_source,
                Int32(-1),
                _device_slot32(step.binding_slot),
            ),
        )
        return nothing
    end
    eta = _lower_device_expr(step.eta, backend.generic_slots, T, issues, "bernoullilogit eta")
    isnothing(eta) && return nothing
    push!(out, DeviceBernoulliLogitChoiceStep(eta, Int32(-1), DEVICE_TRANSFORM_IDENTITY, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendPoissonChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, "poisson")
    lambda = _lower_device_expr(step.lambda, backend.generic_slots, T, issues, "poisson argument")
    (isnothing(src) || isnothing(lambda)) && return nothing
    value_source, tcode = src
    push!(out, DevicePoissonChoiceStep(lambda, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

# Any other supported-by-CPU-backend choice family that we deliberately do not
# yet lower to the device.
function _lower_device_step!(
    out,
    step::BackendNoncenteredNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    if in_loop
        _device_issue!(issues, "device lowering does not support latent noncentered normal choices inside a loop")
        return nothing
    end
    isnothing(step.parameter_slot) && begin
        _device_issue!(issues, "noncentered normal requires a parameter slot")
        return nothing
    end
    slot_position = findfirst(s -> s.value_index == step.parameter_slot, layout.slots)
    if isnothing(slot_position) || layout.slots[slot_position].dimension != 1
        _device_issue!(issues, "device lowering could not resolve the noncentered normal parameter slot")
        return nothing
    end
    mu = _lower_device_expr(step.mu, backend.generic_slots, T, issues, "noncentered normal argument")
    sigma = _lower_device_expr(step.sigma, backend.generic_slots, T, issues, "noncentered normal argument")
    (isnothing(mu) || isnothing(sigma)) && return nothing
    push!(
        out,
        DeviceNoncenteredNormalChoiceStep(
            mu,
            sigma,
            Int32(layout.slots[slot_position].index),
            _device_slot32(step.binding_slot),
        ),
    )
    return nothing
end

function _lower_device_step!(out, step::BackendChoicePlanStep, backend, layout, ::Type{T}, issues, loop_counter, in_loop) where {T}
    _device_issue!(issues, "device lowering does not support the $(nameof(typeof(step))) distribution family yet")
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendDeterministicPlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    slot = step.binding_slot
    if backend.numeric_slots[slot]
        expr = _lower_device_expr(step.expr, backend.generic_slots, T, issues, "deterministic assignment")
        isnothing(expr) && return nothing
        push!(out, DeviceDeterministicStep(expr, Int32(slot)))
    elseif backend.index_slots[slot]
        # Index deterministic slots are staged on the host for loop iterables and
        # addresses, but kernel expressions may read them too (e.g. binomial trials),
        # so emit them on device when the expression lowers; a host-only leftover
        # (e.g. a range iterable) is caught by the read/write audit below if a
        # kernel expression references it.
        probe_issues = String[]
        expr = _lower_device_expr(step.expr, backend.generic_slots, T, probe_issues, "deterministic assignment")
        isnothing(expr) || push!(out, DeviceDeterministicStep(expr, Int32(slot)))
    end
    # Generic deterministic slots only feed addresses, which are resolved on the
    # host during staging; the kernel does not need them.
    return nothing
end

# Statically evaluate a backend expression to a concrete value when it depends
# only on literals (captured constants); `nothing` if it references a runtime slot
# (model argument / binding) and is only knowable during host staging.
_device_static_value(::AbstractBackendExpr) = nothing
_device_static_value(expr::BackendLiteralExpr) = expr.value
function _device_static_value(expr::BackendPrimitiveExpr)
    (expr.op === Symbol(":") || expr.op === Symbol("=>")) && return nothing
    isempty(expr.arguments) && return nothing
    values = map(_device_static_value, expr.arguments)
    any(isnothing, values) && return nothing
    try
        return getfield(Base, expr.op)(values...)
    catch
        return nothing
    end
end
function _device_static_value(expr::BackendBlockExpr)
    value = nothing
    for argument in expr.arguments
        value = _device_static_value(argument)
    end
    return value
end

# The `:` range primitive backing a loop iterable (a block iterable ends in it).
_device_loop_range_expr(expr::BackendPrimitiveExpr) = expr.op === Symbol(":") ? expr : nothing
function _device_loop_range_expr(expr::BackendBlockExpr)
    for argument in Iterators.reverse(expr.arguments)
        range = _device_loop_range_expr(argument)
        isnothing(range) || return range
    end
    return nothing
end
_device_loop_range_expr(::AbstractBackendExpr) = nothing

_device_fits_int32(n::Integer) = typemin(Int32) <= n <= typemax(Int32)

# Enforce, IN THE REPORT, the same loop-range constraints device staging enforces
# (src/device/staging.jl): unit step and Int32-representable bounds/trip count. The
# report is a public contract, so a range it cannot encode must surface as
# supported=false here rather than as a staging error later (issue #83). Bounds
# that depend on runtime arguments stay unchecked here (staging resolves them).
function _device_check_loop_iterable!(issues, iterable)
    range = _device_loop_range_expr(iterable)
    isnothing(range) && return nothing
    args = range.arguments
    nargs = length(args)
    step_value = nargs == 3 ? _device_static_value(args[2]) : 1
    if step_value isa Integer && step_value != 1
        _device_issue!(
            issues,
            "device staging only supports unit-step loop ranges (see device_lowering_report); loop iterable has step $step_value",
        )
    end
    start_value = _device_static_value(args[1])
    stop_value = _device_static_value(args[nargs])
    if start_value isa Integer && !_device_fits_int32(start_value)
        _device_issue!(
            issues,
            "device staging requires Int32-representable loop bounds; loop start $start_value is out of Int32 range",
        )
    end
    if start_value isa Integer && stop_value isa Integer && step_value isa Integer && step_value != 0
        trip_count = length(start_value:step_value:stop_value)
        if !_device_fits_int32(trip_count)
            _device_issue!(
                issues,
                "device staging requires an Int32-representable loop trip count; got $trip_count",
            )
        end
    end
    return nothing
end

function _lower_device_step!(out, step::BackendLoopPlanStep, backend, layout, ::Type{T}, issues, loop_counter, in_loop) where {T}
    if in_loop
        _device_issue!(issues, "device lowering does not support nested loops")
        return nothing
    end
    _device_check_loop_iterable!(issues, step.iterable)
    loop_counter[] += Int32(1)
    loop_id = loop_counter[]
    body_vec = Any[]
    for inner in step.body
        _lower_device_step!(body_vec, inner, backend, layout, T, issues, loop_counter, true)
    end
    any(isnothing, body_vec) && return nothing
    push!(out, DeviceLoopStep(loop_id, Int32(step.iterator_slot), tuple(body_vec...)))
    return nothing
end

# Read/write audit over the lowered steps: which slots kernel expressions read,
# and which slots the kernel itself materializes. Reads and writes are grouped by
# loop context (`loop_id == 0` is top level) because a loop-body write only
# materializes a slot while that loop's body runs -- a zero-trip loop never
# writes it, so a read outside the loop cannot rely on it.
_device_expr_reads!(reads::BitSet, expr::DeviceSlotExpr) = (push!(reads, Int(expr.slot)); nothing)
function _device_expr_reads!(reads::BitSet, expr::DevicePrimitiveExpr)
    for arg in expr.args
        _device_expr_reads!(reads, arg)
    end
    return nothing
end
_device_expr_reads!(reads::BitSet, expr::AbstractDeviceExpr) = nothing

function _device_step_expr_reads!(reads::BitSet, step)
    for name in propertynames(step)
        value = getproperty(step, name)
        if value isa AbstractDeviceExpr
            _device_expr_reads!(reads, value)
        elseif value isa Tuple # categorical probabilities
            for element in value
                element isa AbstractDeviceExpr && _device_expr_reads!(reads, element)
            end
        end
    end
    return nothing
end

function _device_collect_expr_reads!(reads_by_loop::Dict{Int32,BitSet}, steps, loop_id::Int32)
    reads = get!(BitSet, reads_by_loop, loop_id)
    for step in steps
        if step isa DeviceLoopStep
            _device_collect_expr_reads!(reads_by_loop, step.body, step.loop_id)
        else
            _device_step_expr_reads!(reads, step)
        end
    end
    return nothing
end

# Vector choice steps carry their binding slot but never materialize it in the
# scalar slots matrix; the audit must NOT count it as written, so a downstream
# read is honestly rejected instead of reading uninitialized scratch.
_device_step_writes_binding(step::AbstractDevicePlanStep) = true
_device_step_writes_binding(::DeviceMvNormalChoiceStep) = false
_device_step_writes_binding(::DeviceDirichletChoiceStep) = false
_device_step_writes_binding(::DeviceMvNormalDenseChoiceStep) = false
_device_step_writes_binding(::DeviceLKJCholeskyChoiceStep) = false

function _device_collect_written_slots!(written_by_loop::Dict{Int32,BitSet}, steps, loop_id::Int32)
    written = get!(BitSet, written_by_loop, loop_id)
    for step in steps
        if step isa DeviceLoopStep
            # the iterator is only defined while the loop body runs
            body_written = get!(BitSet, written_by_loop, step.loop_id)
            step.iterator_slot > Int32(0) && push!(body_written, Int(step.iterator_slot))
            _device_collect_written_slots!(written_by_loop, step.body, step.loop_id)
        elseif _device_step_writes_binding(step) && step.binding_slot > Int32(0)
            push!(written, Int(step.binding_slot))
        end
    end
    return nothing
end

# Drop device-emitted index deterministics no live kernel expression reads: they
# only exist to feed kernel reads (host staging keeps its own evaluation), and an
# unread emission would needlessly trip the argument-rebinding audit for models
# that rebind an argument into a host-only loop bound. Liveness closes over
# chains of index deterministics but never over a pruned step's own expression
# (a self-referential rebind must not keep itself alive).
_device_is_index_deterministic(step, index_slots::BitVector) =
    step isa DeviceDeterministicStep && index_slots[Int(step.binding_slot)]

function _device_collect_reads_filtered!(reads::BitSet, steps, keep::F) where {F}
    for step in steps
        if step isa DeviceLoopStep
            _device_collect_reads_filtered!(reads, step.body, keep)
        elseif keep(step)
            _device_step_expr_reads!(reads, step)
        end
    end
    return nothing
end

function _device_live_reads(steps, index_slots::BitVector)
    live = BitSet()
    _device_collect_reads_filtered!(live, steps, step -> !_device_is_index_deterministic(step, index_slots))
    while true
        before = length(live)
        _device_collect_reads_filtered!(
            live,
            steps,
            step -> _device_is_index_deterministic(step, index_slots) && Int(step.binding_slot) in live,
        )
        length(live) == before && break
    end
    return live
end

function _device_prune_step(step::DeviceLoopStep, index_slots::BitVector, live::BitSet)
    body = Any[]
    for inner in step.body
        kept = _device_prune_step(inner, index_slots, live)
        isnothing(kept) || push!(body, kept)
    end
    return DeviceLoopStep(step.loop_id, step.iterator_slot, tuple(body...))
end
function _device_prune_step(step::DeviceDeterministicStep, index_slots::BitVector, live::BitSet)
    (_device_is_index_deterministic(step, index_slots) && !(Int(step.binding_slot) in live)) && return nothing
    return step
end
_device_prune_step(step, index_slots::BitVector, live::BitSet) = step

# ---- host-staging feasibility (backend-plan level) -------------------------------
# Staging resolves loop ranges and choice addresses on the host, where choice
# bindings (latent or observed) are never materialized; a range or address that
# depends on one -- directly or through deterministics -- cannot be staged.

_backend_expr_slot_refs!(refs::BitSet, expr::BackendSlotExpr) = (push!(refs, expr.slot); nothing)
function _backend_expr_slot_refs!(refs::BitSet, expr::Union{BackendPrimitiveExpr,BackendBlockExpr,BackendTupleExpr})
    for arg in expr.arguments
        _backend_expr_slot_refs!(refs, arg)
    end
    return nothing
end
_backend_expr_slot_refs!(refs::BitSet, expr::AbstractBackendExpr) = nothing

function _device_staging_taint(backend::BackendExecutionPlan)
    taint = BitSet()
    changed = Ref(true)
    while changed[]
        changed[] = false
        _device_staging_taint_pass!(taint, backend.steps, changed)
    end
    return taint
end

function _device_staging_taint_pass!(taint::BitSet, steps, changed::Ref{Bool})
    for step in steps
        if step isa BackendLoopPlanStep
            _device_staging_taint_pass!(taint, step.body, changed)
        elseif step isa BackendChoicePlanStep
            slot = step.binding_slot
            if !isnothing(slot) && !(slot in taint)
                push!(taint, slot)
                changed[] = true
            end
        elseif step isa BackendDeterministicPlanStep
            if !(step.binding_slot in taint)
                refs = BitSet()
                _backend_expr_slot_refs!(refs, step.expr)
                if !isempty(intersect(refs, taint))
                    push!(taint, step.binding_slot)
                    changed[] = true
                end
            end
        end
    end
    return nothing
end

function _device_check_staging_refs!(issues::Vector{String}, steps, taint::BitSet, symbols)
    for step in steps
        if step isa BackendLoopPlanStep
            refs = BitSet()
            _backend_expr_slot_refs!(refs, step.iterable)
            for slot in intersect(refs, taint)
                _device_issue!(
                    issues,
                    "device staging cannot resolve a loop range that depends on the random choice binding `$(symbols[slot])`",
                )
            end
            _device_check_staging_refs!(issues, step.body, taint, symbols)
        elseif step isa BackendChoicePlanStep
            for part in step.address.parts
                part isa BackendAddressExprPart || continue
                refs = BitSet()
                _backend_expr_slot_refs!(refs, part.expr)
                for slot in intersect(refs, taint)
                    _device_issue!(
                        issues,
                        "device staging cannot resolve a choice address that depends on the random choice binding `$(symbols[slot])`",
                    )
                end
            end
        end
    end
    return nothing
end

# When the audit rejects a read of a deterministic binding, recover the concrete
# reason its device emission was skipped (the emission probe discards issues) so
# the report points at the real blocker instead of a generic host-only message.
function _device_find_deterministic(steps, slot::Int)
    for step in steps
        if step isa BackendDeterministicPlanStep && step.binding_slot == slot
            return step
        elseif step isa BackendLoopPlanStep
            found = _device_find_deterministic(step.body, slot)
            isnothing(found) || return found
        end
    end
    return nothing
end

function _device_probe_deterministic_issue(backend::BackendExecutionPlan, slot::Int)
    step = _device_find_deterministic(backend.steps, slot)
    isnothing(step) && return nothing
    probe_issues = String[]
    expr = _lower_device_expr(step.expr, backend.generic_slots, Float64, probe_issues, "its defining expression")
    (isnothing(expr) && !isempty(probe_issues)) && return first(probe_issues)
    return nothing
end

# Device lowering is keyed off the conditioning signature (issue #95, PR-4): the
# backend plan and parameter layout come from `resolved` (a signature-specific
# `ExecutionPlan`), so the observed/latent split -- the dense observed matrix and
# the latent slot spans -- matches the CPU signature layout by construction.
function _lower_device_plan(model::TeaModel, resolved::ResolvedSignaturePlan, ::Type{T}) where {T}
    issues = String[]
    lowering = _signature_backend_lowering(model, resolved)
    backend = lowering.plan
    if isnothing(backend)
        report = lowering.report
        if isempty(report.issues)
            _device_issue!(issues, "model $(model.name) is not representable in the second-stage backend")
        else
            append!(issues, report.issues)
        end
        return issues, nothing
    end

    layout = resolved.plan.parameter_layout
    loop_counter = Ref(Int32(0))
    out = Any[]
    for step in backend.steps
        _lower_device_step!(out, step, backend, layout, T, issues, loop_counter, false)
    end

    if !isempty(issues)
        return issues, nothing
    end

    environment_layout = resolved.plan.environment_layout
    symbols = environment_layout.symbols

    # loop ranges and choice addresses are resolved by host staging, where choice
    # bindings are never materialized; reject those dependencies here instead of
    # leaking a staging error out of workspace construction.
    _device_check_staging_refs!(issues, backend.steps, _device_staging_taint(backend), symbols)
    if !isempty(issues)
        return issues, nothing
    end

    # drop emitted index deterministics no live kernel expression reads (host
    # staging keeps its own evaluation of them)
    live_reads = _device_live_reads(out, backend.index_slots)
    pruned = Any[]
    for step in out
        kept = _device_prune_step(step, backend.index_slots, live_reads)
        isnothing(kept) || push!(pruned, kept)
    end
    out = pruned

    # every slot a kernel expression reads must be materialized on the device:
    # written by a kernel step in scope (choice/deterministic binding; the loop
    # iterator and loop-body writes only while that loop's body runs) or staged
    # from a model argument (issue #38). Host-only slots (index/generic
    # deterministics that did not lower, e.g. range iterables) would otherwise
    # be read as uninitialized scratch.
    reads_by_loop = Dict{Int32,BitSet}()
    _device_collect_expr_reads!(reads_by_loop, out, Int32(0))
    written_by_loop = Dict{Int32,BitSet}()
    _device_collect_written_slots!(written_by_loop, out, Int32(0))
    top_available = union(
        BitSet(environment_layout.argument_slots),
        get(written_by_loop, Int32(0), BitSet()),
    )
    for (loop_id, reads) in reads_by_loop
        available =
            loop_id == Int32(0) ? top_available :
            union(top_available, get(written_by_loop, loop_id, BitSet()))
        for slot in setdiff(reads, available)
            message = "device lowering cannot read binding `$(symbols[slot])` (slot $slot): it is not materialized on the device at that point"
            probe = _device_probe_deterministic_issue(backend, slot)
            isnothing(probe) || (message *= " ($probe)")
            _device_issue!(issues, message)
        end
    end
    if !isempty(issues)
        return issues, nothing
    end

    # argument slots are staged once per workspace (issue #38); a model that
    # rebinds an argument symbol would have kernels overwrite that slot and
    # break workspace reuse, so reject the rebinding shape outright
    argument_slots = BitSet(environment_layout.argument_slots)
    if !isempty(argument_slots) && _device_steps_rebind_argument(out, argument_slots)
        _device_issue!(
            issues,
            "device lowering does not support rebinding a model argument symbol; rename the binding",
        )
        return issues, nothing
    end

    slot_count = length(environment_layout.symbols)
    steps = tuple(out...)
    plan = DeviceExecutionPlan{T}(steps, slot_count, loop_counter[])
    return issues, plan
end

"""
    device_lowering_report(model; precision=Float64) -> (supported::Bool, issues::Vector{String})

Reports whether `model` can be lowered to the device (KernelAbstractions) logjoint
path. `issues` is empty iff `supported` is `true`; otherwise each entry is a precise
explanation of what is not representable.
"""
function _device_steps_rebind_argument(steps, argument_slots::BitSet)
    for step in steps
        if _device_step_writes_binding(step)
            binding = hasproperty(step, :binding_slot) ? step.binding_slot : Int32(-1)
            binding isa Integer && Int(binding) in argument_slots && return true
        end
        hasproperty(step, :body) && _device_steps_rebind_argument(step.body, argument_slots) && return true
    end
    return false
end

function device_lowering_report(model::TeaModel; constraints::ChoiceMap=choicemap(), precision::Type=Float64)
    resolved = _resolve_signature_plan(model, _representative_constraints(constraints))
    issues, plan = _lower_device_plan(model, resolved, precision)
    return (isempty(issues) && !isnothing(plan), issues)
end
