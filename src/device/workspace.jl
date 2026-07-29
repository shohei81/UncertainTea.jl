# Device-resident workspace: holds the lowered plan, the staged observation buffers,
# and reusable device buffers for params / slots / totals. Buffers are allocated via
# `KernelAbstractions.allocate(backend, T, dims...)`, so the same code path serves the
# CPU reference backend and any GPU backend (e.g. Metal) unchanged.

# Precision traits. `default_device_precision` names the natural precision of a
# backend; `_device_supports_float64` gates the Float64 request. The Metal extension
# overrides both for MetalBackend (Float32-only).
default_device_precision(::KernelAbstractions.Backend) = Float64
_device_supports_float64(::KernelAbstractions.Backend) = true

function _check_device_precision(backend::KernelAbstractions.Backend, precision::Type)
    if precision === Float64 && !_device_supports_float64(backend)
        throw(
            ArgumentError(
                "backend $(typeof(backend)) does not support Float64 arithmetic; " *
                "request precision=$(default_device_precision(backend)) instead",
            ),
        )
    end
    return nothing
end

function _device_unsupported_message(model::TeaModel, issues::Vector{String})
    io = IOBuffer()
    print(io, "model $(model.name) cannot be lowered to the device logjoint path.")
    if !isempty(issues)
        print(io, " Issues:")
        for issue in issues
            print(io, "\n  - ", issue)
        end
    end
    print(io, "\nSee device_lowering_report(model) for details.")
    return String(take!(io))
end

mutable struct DeviceBatchedWorkspace{T,B<:KernelAbstractions.Backend,P<:DeviceExecutionPlan{T}}
    model::TeaModel
    backend::B
    plan::P
    batch_size::Int
    parameter_count::Int
    slot_count::Int
    params_device::Any
    slots_device::Any
    observed_device::Any
    observed_int_device::Any
    totals_device::Any
    trip_counts_device::Any
    loop_starts_device::Any
    # model arguments, kept for staging into the lazily allocated gradient
    # slot scratch (issue #38)
    args::Any
    # Gradient buffers (allocated lazily on the first gradient call). `grad_slots_device`
    # is a `DeviceDual{T}` scratch laid out (slot_count x parameter_count x batch).
    gradients_device::Any
    grad_slots_device::Any
    # Observation-parallel tiled gradient (issue #153): the descriptor when the plan
    # has a tileable observation loop large enough to tile (else `nothing`, keeping
    # the serial `(P, C)` scan), plus the per-tile partial value/gradient scratch
    # (allocated lazily with the gradient buffers).
    tiled_gradient::Any
    tile_partial_grads::Any
    tile_partial_totals::Any
    # Signature layout + representative constraints kept so a params-width
    # mismatch can name the conditioning (issue #95, PR-5).
    layout::ParameterLayout
    signature_constraints::ChoiceMap
end

function DeviceBatchedWorkspace(
    model::TeaModel,
    batch_size::Integer;
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=Float64,
    args=(),
    constraints=choicemap(),
)
    _check_device_precision(backend, precision)
    # fill missing trailing model arguments from the model's defaults, so the
    # device path agrees with generate and the CPU scoring entry points
    args =
        args isa Tuple ? _complete_model_args(model, args) :
        Tuple[_complete_model_args(model, batch_args) for batch_args in args]
    T = precision
    # Resolve the conditioning-signature plan once and derive the device plan,
    # observation staging, and parameter count from it (issue #95, PR-4).
    resolved = _resolve_signature_plan(model, _representative_constraints(constraints))
    issues, plan = _lower_device_plan(model, resolved, T)
    isnothing(plan) && throw(ArgumentError(_device_unsupported_message(model, issues)))

    batch_size = Int(batch_size)
    backend_plan = _signature_backend_plan(model, resolved)
    # Detect a tileable observation loop before staging so staging captures the
    # loop's base-cursor / per-iteration stride anchors (issue #153).
    tileable = _device_detect_tileable_loop(plan)
    tile_loop_id = isnothing(tileable) ? 0 : Int(tileable.loop_id)
    bundle =
        _stage_device_observations(model, backend_plan, plan, args, constraints, batch_size; tile_loop_id=tile_loop_id)

    parameter_count = parametercount(resolved.plan.parameter_layout)
    slot_count = Int(plan.slot_count)

    # Shared observations stage a single column and broadcast (issue #153); a
    # per-chain (batched-argument) staging keeps `C` columns. The device buffers
    # mirror the staged column count.
    obs_cols = size(bundle.observed, 2)

    params_device = KernelAbstractions.allocate(backend, T, parameter_count, batch_size)
    slots_device = KernelAbstractions.allocate(backend, T, slot_count, batch_size)
    _device_stage_arguments!(slots_device, model, args, batch_size, T)
    observed_device = KernelAbstractions.allocate(backend, T, size(bundle.observed, 1), obs_cols)
    copyto!(observed_device, bundle.observed)
    # The exact-integer mirror is only read by count-family (binomial) steps; drop
    # it to a 1x1 dummy when the plan has none, so the (potentially large) Int64
    # buffer is neither materialized per chain nor uploaded (issue #71/#153).
    if _device_plan_has_count_family(plan)
        observed_int_device = KernelAbstractions.allocate(backend, Int64, size(bundle.observed_int, 1), obs_cols)
        copyto!(observed_int_device, bundle.observed_int)
    else
        observed_int_device = fill!(KernelAbstractions.allocate(backend, Int64, 1, 1), Int64(0))
    end
    # Build the tiled-gradient descriptor when the flagged loop is large enough that
    # the occupancy win outweighs the extra kernel launches; below the threshold the
    # serial scan is kept so small models stay bitwise identical to the untiled path.
    tiled_gradient = nothing
    if !isnothing(tileable) && bundle.tile_stride > Int32(0)
        n_obs = bundle.trip_counts[tile_loop_id]
        if n_obs >= DEVICE_GRADIENT_TILE_MIN_OBS
            tiled_gradient = _build_device_tiled_gradient(plan, tileable, bundle)
        end
    end
    totals_device = KernelAbstractions.allocate(backend, T, batch_size)
    trip_counts_device = KernelAbstractions.allocate(backend, Int32, length(bundle.trip_counts))
    loop_starts_device = KernelAbstractions.allocate(backend, Int32, length(bundle.loop_starts))
    copyto!(trip_counts_device, bundle.trip_counts)
    copyto!(loop_starts_device, bundle.loop_starts)

    return DeviceBatchedWorkspace{T,typeof(backend),typeof(plan)}(
        model,
        backend,
        plan,
        batch_size,
        parameter_count,
        slot_count,
        params_device,
        slots_device,
        observed_device,
        observed_int_device,
        totals_device,
        trip_counts_device,
        loop_starts_device,
        args,
        nothing,
        nothing,
        tiled_gradient,
        nothing,
        nothing,
        resolved.plan.parameter_layout,
        _representative_constraints(constraints),
    )
end

# Allocates (once) the device buffers the gradient kernel needs: a plain-`T`
# `parameter_count x batch` gradient matrix and a `DeviceDual{T}` slot scratch laid
# out `(slot_count, parameter_count, batch)` so each `(parameter, batch)` thread owns
# its own slot column.
function _device_ensure_gradient_buffers!(workspace::DeviceBatchedWorkspace{T}) where {T}
    if isnothing(workspace.gradients_device)
        workspace.gradients_device =
            KernelAbstractions.allocate(workspace.backend, T, workspace.parameter_count, workspace.batch_size)
    end
    if isnothing(workspace.grad_slots_device)
        workspace.grad_slots_device = KernelAbstractions.allocate(
            workspace.backend,
            DeviceDual{T},
            workspace.slot_count,
            workspace.parameter_count,
            workspace.batch_size,
        )
        _device_stage_gradient_arguments!(
            workspace.grad_slots_device,
            workspace.model,
            workspace.args,
            workspace.parameter_count,
            workspace.batch_size,
            T,
        )
    end
    if !isnothing(workspace.tiled_gradient) && isnothing(workspace.tile_partial_grads)
        tg = workspace.tiled_gradient
        workspace.tile_partial_grads = KernelAbstractions.allocate(
            workspace.backend,
            T,
            workspace.parameter_count,
            Int(tg.ntiles),
            workspace.batch_size,
        )
        workspace.tile_partial_totals =
            KernelAbstractions.allocate(workspace.backend, T, Int(tg.ntiles), workspace.batch_size)
    end
    return nothing
end

# Kernels never write argument slots, so staging their values once covers
# every launch (issue #38). Non-Real arguments are skipped: expressions
# referencing them are not device-lowerable in the first place, and integer
# loop bounds are consumed host-side through the trip-count staging.
function _device_stage_arguments!(slots_device, model::TeaModel, args, batch_size::Int, ::Type{T}) where {T}
    argument_slots = executionplan(model).environment_layout.argument_slots
    isempty(argument_slots) && return nothing
    staged = Matrix{T}(undef, 1, batch_size)
    for (argument_index, slot) in enumerate(argument_slots)
        if args isa Tuple
            value = args[argument_index]
            value isa Real || continue
            fill!(staged, convert(T, value))
        else
            all(args[batch_index][argument_index] isa Real for batch_index = 1:batch_size) || continue
            for batch_index = 1:batch_size
                staged[1, batch_index] = convert(T, args[batch_index][argument_index])
            end
        end
        copyto!(view(slots_device, slot:slot, :), staged)
    end
    return nothing
end

function _device_stage_gradient_arguments!(
    grad_slots_device,
    model::TeaModel,
    args,
    parameter_count::Int,
    batch_size::Int,
    ::Type{T},
) where {T}
    argument_slots = executionplan(model).environment_layout.argument_slots
    isempty(argument_slots) && return nothing
    staged = Array{DeviceDual{T}}(undef, 1, parameter_count, batch_size)
    for (argument_index, slot) in enumerate(argument_slots)
        if args isa Tuple
            value = args[argument_index]
            value isa Real || continue
            fill!(staged, DeviceDual{T}(convert(T, value), zero(T)))
        else
            all(args[batch_index][argument_index] isa Real for batch_index = 1:batch_size) || continue
            for batch_index = 1:batch_size
                dual = DeviceDual{T}(convert(T, args[batch_index][argument_index]), zero(T))
                for parameter_index = 1:parameter_count
                    staged[1, parameter_index, batch_index] = dual
                end
            end
        end
        copyto!(view(grad_slots_device, slot:slot, :, :), staged)
    end
    return nothing
end

function _device_upload_params!(workspace::DeviceBatchedWorkspace{T}, params::AbstractMatrix) where {T}
    if eltype(params) === T
        copyto!(workspace.params_device, params)
    else
        copyto!(workspace.params_device, convert(Array{T}, Array(params)))
    end
    return nothing
end

"""
    device_batched_logjoint!(workspace, params) -> Vector

Runs the fused device kernel for `params` (unconstrained, `parameter_count × batch`)
reusing the workspace's staged observations and device buffers, and returns the
per-column unconstrained logjoint (including the transform log-abs-det). Successive
calls with different `params` are independent (staging is not mutated).
"""
function device_batched_logjoint!(workspace::DeviceBatchedWorkspace{T}, params::AbstractMatrix) where {T}
    size(params, 1) == workspace.parameter_count || throw(
        _signature_length_error(
            workspace.model,
            workspace.layout,
            workspace.signature_constraints,
            workspace.parameter_count,
            size(params, 1),
        ),
    )
    size(params, 2) == workspace.batch_size ||
        throw(DimensionMismatch("expected $(workspace.batch_size) batch elements, got $(size(params, 2))"))

    _device_upload_params!(workspace, params)
    kernel = _device_logjoint_kernel!(workspace.backend)
    kernel(
        workspace.totals_device,
        workspace.plan,
        workspace.params_device,
        workspace.observed_device,
        workspace.observed_int_device,
        workspace.slots_device,
        workspace.trip_counts_device,
        workspace.loop_starts_device;
        ndrange=workspace.batch_size,
    )
    KernelAbstractions.synchronize(workspace.backend)

    result = Vector{T}(undef, workspace.batch_size)
    copyto!(result, workspace.totals_device)
    return result
end
