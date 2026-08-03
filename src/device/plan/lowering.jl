# ---- lowering ------------------------------------------------------------------

const DEVICE_SUPPORTED_PRIMITIVES = (
    :+, :-, :*, :/, :^, :exp, :log, :log1p, :sqrt, :abs, :min, :max, :clamp,
    # issue #286: trig/hyperbolic/expm1 -- Metal-supported Base intrinsics with
    # DeviceDual/DeviceGradN rules, mirrored in _device_apply below
    :sin, :cos, :tan, :tanh, :sinh, :cosh, :atan, :expm1,
)

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
#
# The pure one-/two-argument delegate wrappers are @eval-generated from
# DISTRIBUTION_FAMILY_TABLE below the `_lower_device_two_arg!` /
# `_lower_device_one_arg!` helpers (issue #331 stage 2); only the families
# with extra structure keep hand-written wrappers here.
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
    if dimension > DEVICE_MAX_VECTOR_DIMENSION
        # The compile-time `Val(D)` fold would blow the shader budget past D = 16.
        # Lower the IID LATENT case -- every component shares one mu/sigma expr, e.g.
        # a large GLM's coefficient prior -- to the runtime-length loop step (issue
        # #221). A heterogeneous prior or an observation mvnormal stays capped.
        if value_source > Int32(0) && allequal(mu) && allequal(sigma)
            push!(
                out,
                DeviceDiagNormalChoiceStepDyn(
                    first(mu),
                    first(sigma),
                    Int32(dimension),
                    value_source,
                    _device_slot32(step.binding_slot),
                ),
            )
            return nothing
        end
        _device_issue!(
            issues,
            "device lowering caps vector dimensions at $DEVICE_MAX_VECTOR_DIMENSION (kernel compile-time budget), got an mvnormal of dimension $dimension" *
            (value_source > Int32(0) ? " with non-iid components (only an iid latent prior lowers past the cap)" : ""),
        )
        return nothing
    end
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
    step::BackendMvStudentTChoicePlanStep,
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
            "device lowering caps vector dimensions at $DEVICE_MAX_VECTOR_DIMENSION (kernel compile-time budget), got an mvstudentt of dimension $dimension",
        )
        return nothing
    end
    if isnothing(step.parameter_slot)
        value_source = Int32(-1)
    else
        if in_loop
            _device_issue!(issues, "device lowering does not support latent mvstudentt choices inside a loop")
            return nothing
        end
        slot = layout.slots[step.parameter_slot]
        if !(slot.transform isa VectorIdentityTransform) || slot.dimension != dimension ||
           slot.value_length != dimension
            _device_issue!(issues, "device lowering could not resolve the mvstudentt parameter slot")
            return nothing
        end
        value_source = Int32(slot.index)
    end
    nu = _lower_device_expr(step.nu, backend.generic_slots, T, issues, "mvstudentt degrees of freedom")
    mu = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mvstudentt argument"), step.mu)
    sigma = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mvstudentt argument"), step.sigma)
    (isnothing(nu) || any(isnothing, mu) || any(isnothing, sigma)) && return nothing
    push!(out, DeviceMvStudentTChoiceStep(nu, mu, sigma, value_source, _device_slot32(step.binding_slot)))
    return nothing
end

function _lower_device_step!(
    out,
    step::BackendMvStudentTDenseChoicePlanStep,
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
            "device lowering caps mvstudenttdense at dimension $DEVICE_MAX_DENSE_DIMENSION (the unrolled forward substitution is quadratic in the kernel body), got dimension $dimension",
        )
        return nothing
    end
    if isnothing(step.parameter_slot)
        value_source = Int32(-1)
    else
        if in_loop
            _device_issue!(issues, "device lowering does not support latent mvstudenttdense choices inside a loop")
            return nothing
        end
        slot = layout.slots[step.parameter_slot]
        if !(slot.transform isa VectorIdentityTransform) || slot.dimension != dimension ||
           slot.value_length != dimension
            _device_issue!(issues, "device lowering could not resolve the mvstudenttdense parameter slot")
            return nothing
        end
        value_source = Int32(slot.index)
    end
    nu = _lower_device_expr(step.nu, backend.generic_slots, T, issues, "mvstudenttdense degrees of freedom")
    mu = map(expr -> _lower_device_expr(expr, backend.generic_slots, T, issues, "mvstudenttdense argument"), step.mu)
    (isnothing(nu) || any(isnothing, mu)) && return nothing
    push!(out, DeviceMvStudentTDenseChoiceStep(nu, mu, value_source, _device_slot32(step.binding_slot)))
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

# Single-argument analogue of `_lower_device_two_arg!` for the one-parameter
# families (issue #331 stage 2).
function _lower_device_one_arg!(
    out,
    step,
    arg_expr,
    Ctor,
    backend,
    layout,
    ::Type{T},
    issues,
    in_loop,
    family,
) where {T}
    src = _device_choice_value_source(step, layout, in_loop, issues, family)
    a1 = _lower_device_expr(arg_expr, backend.generic_slots, T, issues, "$family argument")
    (isnothing(src) || isnothing(a1)) && return nothing
    value_source, tcode = src
    push!(out, Ctor(a1, value_source, tcode, _device_slot32(step.binding_slot)))
    return nothing
end

# ---- table-generated per-family lowering wrappers (issue #331 stage 2) --------
#
# Every family whose `_lower_device_step!` body is a pure one-/two-argument
# delegate is @eval-generated from DISTRIBUTION_FAMILY_TABLE (`step`/`params`
# fields); the families with extra structure (studentt's three arguments,
# categorical's probability tuple, the vector/truncated/mixture/marginalize
# steps, bernoullilogit's GLM fusion, noncentered normal) stay hand-written.
for family in (
    :normal,
    :lognormal,
    :gamma,
    :laplace,
    :cauchy,
    :logistic,
    :gumbel,
    :frechet,
    :inversegaussian,
    :beta,
    :inversegamma,
    :weibull,
    :binomial,
    :negativebinomial,
    :halfnormal,
    :halfcauchy,
    :rayleigh,
    :exponential,
    :bernoulli,
    :geometric,
    :poisson,
)
    spec = _distribution_family_spec(family)
    step_type = Symbol("Backend", spec.step, "ChoicePlanStep")
    device_step_type = Symbol("Device", spec.step, "ChoiceStep")
    helper = length(spec.params) == 2 ? :_lower_device_two_arg! : :_lower_device_one_arg!
    argument_fields = [:(step.$parameter) for parameter in spec.params]
    family_name = String(family)
    @eval function _lower_device_step!(
        out,
        step::$step_type,
        backend,
        layout,
        ::Type{T},
        issues,
        loop_counter,
        in_loop,
    ) where {T}
        $helper(
            out,
            step,
            $(argument_fields...),
            $device_step_type,
            backend,
            layout,
            T,
            issues,
            in_loop,
            $family_name,
        )
    end
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
        coef_source = _device_linear_predictor_coef_source(eta, layout, issues)
        isnothing(coef_source) && return nothing
        intercept =
            isnothing(eta.intercept) ? DeviceLiteralExpr(zero(T)) :
            _lower_device_expr(eta.intercept, backend.generic_slots, T, issues, "bernoullilogit linear predictor intercept")
        isnothing(intercept) && return nothing
        # D <= 16 keeps the compile-time-unrolled `{D}` step (measured fast); D > 16
        # uses the runtime-length loop step with the analytic gradient, lifting the
        # unroll-by-D compile-budget cap for the GLM class only (issue #221).
        if eta.coef_length > DEVICE_MAX_VECTOR_DIMENSION
            push!(
                out,
                DeviceBernoulliLogitGLMChoiceStepDyn(
                    intercept,
                    coef_source,
                    eta.coef_length,
                    Int32(-1),
                    _device_slot32(step.binding_slot),
                ),
            )
        else
            push!(
                out,
                DeviceBernoulliLogitGLMChoiceStep{eta.coef_length}(
                    intercept,
                    coef_source,
                    Int32(-1),
                    _device_slot32(step.binding_slot),
                ),
            )
        end
        return nothing
    end
    eta = _lower_device_expr(step.eta, backend.generic_slots, T, issues, "bernoullilogit eta")
    isnothing(eta) && return nothing
    push!(out, DeviceBernoulliLogitChoiceStep(eta, Int32(-1), DEVICE_TRANSFORM_IDENTITY, _device_slot32(step.binding_slot)))
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

# ---- broadcast normal (issue #134) ----
#
# A broadcast mu/sigma leaf slot is either SCALAR (a numeric or index slot: a
# scalar latent binding, a scalar model argument, a loop iterator -- read once from
# the `slots` buffer and broadcast across every element, matching the host
# `_eval_backend_broadcast_numeric!` numeric/index branch) or a per-element
# COVARIATE VECTOR (everything else: a generic-storage argument read per element,
# matching the host generic-storage branch). Collect the DISTINCT covariate slots
# in mu-then-sigma, left-to-right first-encounter order; the position in this list
# is the covariate's observation-block offset. Staging walks the same backend exprs
# in the same order, so the offsets it emits line up with the ones lowered here.
_device_broadcast_leaf_is_covariate(slot::Int, numeric_slots::BitVector, index_slots::BitVector) =
    !(numeric_slots[slot] || index_slots[slot])

function _collect_broadcast_covariate_slots!(slots::Vector{Int}, expr::BackendSlotExpr, numeric_slots, index_slots)
    if _device_broadcast_leaf_is_covariate(expr.slot, numeric_slots, index_slots) && !(expr.slot in slots)
        push!(slots, expr.slot)
    end
    return nothing
end
function _collect_broadcast_covariate_slots!(
    slots::Vector{Int},
    expr::Union{BackendPrimitiveExpr,BackendBlockExpr},
    numeric_slots,
    index_slots,
)
    for argument in expr.arguments
        _collect_broadcast_covariate_slots!(slots, argument, numeric_slots, index_slots)
    end
    return nothing
end
_collect_broadcast_covariate_slots!(slots::Vector{Int}, ::AbstractBackendExpr, numeric_slots, index_slots) = nothing

function _broadcast_covariate_slots(mu, sigma, numeric_slots::BitVector, index_slots::BitVector)
    slots = Int[]
    _collect_broadcast_covariate_slots!(slots, mu, numeric_slots, index_slots)
    _collect_broadcast_covariate_slots!(slots, sigma, numeric_slots, index_slots)
    return slots
end

# Lower a broadcast argument expression, replacing each per-element covariate slot
# leaf with a `DeviceObservedColumnExpr` at its assigned observation-block offset;
# scalar leaves reuse the standard scalar expr lowering (`DeviceSlotExpr` etc.).
function _lower_device_broadcast_expr(
    expr::BackendSlotExpr,
    covariate_offsets::Dict{Int,Int32},
    backend,
    ::Type{T},
    issues::Vector{String},
    context::String,
) where {T}
    offset = get(covariate_offsets, expr.slot, Int32(-1))
    offset >= Int32(0) && return DeviceObservedColumnExpr(offset)
    return _lower_device_expr(expr, backend.generic_slots, T, issues, context)
end
function _lower_device_broadcast_expr(
    expr::BackendPrimitiveExpr,
    covariate_offsets::Dict{Int,Int32},
    backend,
    ::Type{T},
    issues::Vector{String},
    context::String,
) where {T}
    if !(expr.op in DEVICE_SUPPORTED_PRIMITIVES)
        _device_issue!(issues, "device lowering does not support primitive `$(expr.op)` in $context")
        return nothing
    end
    lowered = map(arg -> _lower_device_broadcast_expr(arg, covariate_offsets, backend, T, issues, context), expr.arguments)
    any(isnothing, lowered) && return nothing
    return DevicePrimitiveExpr(expr.op, tuple(lowered...))
end
function _lower_device_broadcast_expr(
    expr::BackendBlockExpr,
    covariate_offsets::Dict{Int,Int32},
    backend,
    ::Type{T},
    issues::Vector{String},
    context::String,
) where {T}
    if length(expr.arguments) == 1
        return _lower_device_broadcast_expr(first(expr.arguments), covariate_offsets, backend, T, issues, context)
    end
    _device_issue!(issues, "device lowering does not support multi-statement block expressions in $context")
    return nothing
end
function _lower_device_broadcast_expr(
    expr::AbstractBackendExpr,
    covariate_offsets::Dict{Int,Int32},
    backend,
    ::Type{T},
    issues::Vector{String},
    context::String,
) where {T}
    # a covariate leaf is only ever a slot; every other leaf lowers as a scalar
    return _lower_device_expr(expr, backend.generic_slots, T, issues, context)
end

function _lower_device_step!(
    out,
    step::BackendBroadcastNormalChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    # broadcast latents are rejected at backend lowering, so a step reaching here is
    # observed-only; guard anyway for honesty.
    if !isnothing(step.parameter_slot)
        _device_issue!(issues, "device lowering does not support broadcast normal latent parameters")
        return nothing
    end
    # A broadcast observation carries a dynamic per-workspace element count `M`,
    # resolved by staging like a loop trip count; nesting it inside a loop would make
    # `M` depend on the loop iterate, which staging does not encode. Scope to the
    # top-level flagship form.
    if in_loop
        _device_issue!(issues, "device lowering does not support broadcast normal observations inside a loop")
        return nothing
    end
    covariate_slots = _broadcast_covariate_slots(step.mu, step.sigma, backend.numeric_slots, backend.index_slots)
    if length(covariate_slots) > DEVICE_MAX_VECTOR_DIMENSION
        _device_issue!(
            issues,
            "device lowering caps broadcast covariate vectors at $DEVICE_MAX_VECTOR_DIMENSION (kernel compile-time budget), got $(length(covariate_slots))",
        )
        return nothing
    end
    covariate_offsets = Dict{Int,Int32}()
    for (position, slot) in enumerate(covariate_slots)
        covariate_offsets[slot] = Int32(position - 1)
    end
    mu = _lower_device_broadcast_expr(step.mu, covariate_offsets, backend, T, issues, "broadcast normal mean")
    sigma = _lower_device_broadcast_expr(step.sigma, covariate_offsets, backend, T, issues, "broadcast normal scale")
    (isnothing(mu) || isnothing(sigma)) && return nothing
    # take a shared trip-count slot (like a loop): staging fills it with `M`.
    loop_counter[] += Int32(1)
    count_id = loop_counter[]
    stride = Int32(length(covariate_slots) + 1) # C covariate rows + the y row
    push!(out, DeviceBroadcastNormalChoiceStep(mu, sigma, count_id, stride, Int32(-1), _device_slot32(step.binding_slot)))
    return nothing
end

# ---- marginalize=:enumerate (issue #67) ----
#
# A marginalize suffix that owns a loop or a broadcast observation would allocate
# loop-id / trip-count slots inside the branch scan, breaking the pre-order
# alignment between lowering and host staging (which walk the suffix identically);
# it would also multiply the unrolled kernel body by the loop trip count. The
# supported acceptance class is loop-free suffixes (scalar choices/deterministics,
# possibly nested marginalize), so reject the dynamic forms honestly.
function _device_marg_body_has_dynamic(steps)
    for step in steps
        (step isa DeviceLoopStep || step isa DeviceBroadcastNormalChoiceStep) && return true
        step isa DeviceMarginalizeChoiceStep && _device_marg_body_has_dynamic(step.body) && return true
    end
    return false
end

function _lower_device_step!(
    out,
    step::BackendMarginalizeChoicePlanStep,
    backend,
    layout,
    ::Type{T},
    issues,
    loop_counter,
    in_loop,
) where {T}
    if in_loop
        # loop-scoped marginalized choices are rejected at BACKEND lowering already;
        # guard for honesty (a marginalize suffix carrying `M` per-iterate would need
        # per-iterate staging the device does not encode).
        _device_issue!(issues, "device lowering does not support marginalize=:enumerate choices inside a loop")
        return nothing
    end
    family = step.family
    if !(family === :bernoulli || family === :categorical)
        _device_issue!(
            issues,
            "device lowering supports marginalize=:enumerate for bernoulli and categorical only, got `$family`",
        )
        return nothing
    end
    probabilities =
        map(p -> _lower_device_expr(p, backend.generic_slots, T, issues, "marginalize $family probability"), step.probabilities)
    any(isnothing, probabilities) && return nothing
    # lower the suffix body recursively (the nested-step precedent of DeviceLoopStep)
    body_vec = Any[]
    for inner in step.body
        _lower_device_step!(body_vec, inner, backend, layout, T, issues, loop_counter, in_loop)
    end
    any(isnothing, body_vec) && return nothing
    body = tuple(body_vec...)
    if _device_marg_body_has_dynamic(body)
        _device_issue!(
            issues,
            "device lowering does not support a marginalize=:enumerate suffix containing a loop or broadcast observation",
        )
        return nothing
    end
    # binding VALUES the branches materialize: bernoulli false/true -> 0/1, the
    # categorical category is its own index. The backend caps the total support
    # product at 32, bounding the unrolled kernel body.
    support = family === :bernoulli ? (Int32(0), Int32(1)) : ntuple(i -> Int32(i), length(step.support))
    push!(
        out,
        DeviceMarginalizeChoiceStep{family}(tuple(probabilities...), support, body, _device_slot32(step.binding_slot)),
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
        elseif step isa DeviceMarginalizeChoiceStep
            # the marginalize step's own pmf argument reads live in this scope; the
            # suffix body is scored INLINE in the same kernel scope (no separate loop
            # id), so its reads join this scope and see the branch binding + any
            # fresh suffix bindings the body materializes.
            _device_step_expr_reads!(reads, step)
            _device_collect_expr_reads!(reads_by_loop, step.body, loop_id)
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
_device_step_writes_binding(::DeviceDiagNormalChoiceStepDyn) = false
_device_step_writes_binding(::DeviceDirichletChoiceStep) = false
_device_step_writes_binding(::DeviceMvNormalDenseChoiceStep) = false
_device_step_writes_binding(::DeviceMvStudentTChoiceStep) = false
_device_step_writes_binding(::DeviceMvStudentTDenseChoiceStep) = false
_device_step_writes_binding(::DeviceLKJCholeskyChoiceStep) = false
# broadcast normal binds the whole observed VECTOR (issue #134); the scalar slots
# matrix cannot hold it, so like the mvnormal families the binding is never
# materialized and a downstream read is honestly rejected by the audit.
_device_step_writes_binding(::DeviceBroadcastNormalChoiceStep) = false

function _device_collect_written_slots!(written_by_loop::Dict{Int32,BitSet}, steps, loop_id::Int32)
    written = get!(BitSet, written_by_loop, loop_id)
    for step in steps
        if step isa DeviceLoopStep
            # the iterator is only defined while the loop body runs
            body_written = get!(BitSet, written_by_loop, step.loop_id)
            step.iterator_slot > Int32(0) && push!(body_written, Int(step.iterator_slot))
            _device_collect_written_slots!(written_by_loop, step.body, step.loop_id)
        elseif step isa DeviceMarginalizeChoiceStep
            # the branch binding is materialized per branch, and the suffix runs in
            # the enclosing scope: record the binding and the suffix's writes here so
            # the suffix's own reads resolve (a suffix rebind of a PRE-EXISTING slot
            # is rejected separately, see _device_check_marginalize_rebinds!).
            step.binding_slot > Int32(0) && push!(written, Int(step.binding_slot))
            _device_collect_written_slots!(written_by_loop, step.body, loop_id)
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
        elseif step isa BackendMarginalizeChoicePlanStep
            slot = step.binding_slot
            if !isnothing(slot) && !(slot in taint)
                push!(taint, slot)
                changed[] = true
            end
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
        elseif step isa BackendMarginalizeChoicePlanStep
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

# Broadcast observation covariates ride the observation buffer, so staging must be
# able to resolve them on the host: a per-element covariate vector must be a MODEL
# ARGUMENT (a direct argument or captured constant vector stored in generic env
# storage), never a random-choice or deterministic binding computed later. Reject
# anything else honestly here rather than leaking a staging error (issue #134).
function _device_check_broadcast_covariates!(
    issues::Vector{String},
    steps,
    argument_slots::BitSet,
    numeric_slots::BitVector,
    index_slots::BitVector,
    symbols,
)
    for step in steps
        if step isa BackendLoopPlanStep
            _device_check_broadcast_covariates!(issues, step.body, argument_slots, numeric_slots, index_slots, symbols)
        elseif step isa BackendBroadcastNormalChoicePlanStep
            for slot in _broadcast_covariate_slots(step.mu, step.sigma, numeric_slots, index_slots)
                slot in argument_slots || _device_issue!(
                    issues,
                    "device lowering only supports broadcast normal covariates that are model arguments; `$(symbols[slot])` is not a model argument",
                )
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
    # broadcast covariates ride the observation buffer, so they must be host-
    # stageable model arguments (issue #134)
    _device_check_broadcast_covariates!(
        issues,
        backend.steps,
        BitSet(environment_layout.argument_slots),
        backend.numeric_slots,
        backend.index_slots,
        symbols,
    )
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

    # a marginalize suffix that rebinds a pre-existing slot leaks across the
    # sequential enumeration branches (no per-branch restore on device); reject it
    _device_check_marginalize_rebinds!(issues, out, BitSet(environment_layout.argument_slots), symbols)
    if !isempty(issues)
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
# Slots a marginalize suffix (recursively) MATERIALIZES: choice/deterministic
# bindings, nested-marginalize bindings, loop iterators. Used to reject a suffix
# that rebinds a pre-existing slot (see below).
function _device_marg_body_writes!(w::BitSet, steps)
    for step in steps
        if step isa DeviceLoopStep
            step.iterator_slot > Int32(0) && push!(w, Int(step.iterator_slot))
            _device_marg_body_writes!(w, step.body)
        elseif step isa DeviceMarginalizeChoiceStep
            step.binding_slot > Int32(0) && push!(w, Int(step.binding_slot))
            _device_marg_body_writes!(w, step.body)
        elseif _device_step_writes_binding(step) && step.binding_slot > Int32(0)
            push!(w, Int(step.binding_slot))
        end
    end
    return nothing
end

# ENV DISCIPLINE (issue #67 point 3): the CPU path snapshots/restores the
# environment per enumeration branch. The device scores the branches sequentially
# in one kernel with NO restore, reusing the `slots` matrix. That is correct
# whenever every slot the suffix reads is either pre-existing (never written by the
# suffix, so stable) or written-before-read WITHIN the branch (fresh each branch by
# program order). The one unsound shape is a suffix that REBINDS a slot already
# materialized before the marginalize step (a leaked write would carry into the
# next branch -- e.g. `s = 1.0; z ~ bernoulli(..; marginalize); s = s + 1.0`), so
# match the CPU honest-rejection precedent and reject it here rather than restore.
function _device_check_marginalize_rebinds!(issues::Vector{String}, steps, before::BitSet, symbols)
    accumulated = copy(before)
    for step in steps
        if step isa DeviceMarginalizeChoiceStep
            body_writes = BitSet()
            _device_marg_body_writes!(body_writes, step.body)
            for slot in intersect(body_writes, accumulated)
                _device_issue!(
                    issues,
                    "device lowering does not support a marginalize=:enumerate suffix that rebinds the pre-existing binding `$(symbols[slot])` (slot $slot); the per-branch write would leak across enumeration branches",
                )
            end
            inner_before = copy(accumulated)
            step.binding_slot > Int32(0) && push!(inner_before, Int(step.binding_slot))
            _device_check_marginalize_rebinds!(issues, step.body, inner_before, symbols)
        elseif step isa DeviceLoopStep
            inner_before = copy(accumulated)
            step.iterator_slot > Int32(0) && push!(inner_before, Int(step.iterator_slot))
            _device_check_marginalize_rebinds!(issues, step.body, inner_before, symbols)
        end
        if step isa DeviceLoopStep
            step.iterator_slot > Int32(0) && push!(accumulated, Int(step.iterator_slot))
        elseif _device_step_writes_binding(step) && step.binding_slot > Int32(0)
            push!(accumulated, Int(step.binding_slot))
        end
    end
    return nothing
end

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
