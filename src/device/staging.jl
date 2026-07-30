# One-time host staging. Given (args, constraints), resolve every observed choice
# value into a dense `observed::Matrix{T}` (row per choice instance in kernel
# pre-order, column per batch element), and resolve each loop's trip count / start.
#
# The traversal walks the second-stage BackendExecutionPlan in the SAME pre-order
# the device kernel uses, so the observation cursor in the kernel lines up with the
# rows produced here. It reuses the batched-backend helpers (`_batched_backend_address_parts`,
# `_batched_choice_numeric_values!`, `_batched_index_iterable_reference`) so the
# resolution logic is identical to the CPU batched path.

struct DeviceStagingBundle{T}
    observed::Matrix{T}
    # Exact integer mirror of `observed`, row-for-row: integer-valued entries
    # (observation counts, binomial trials) as `Int64`. Discrete-count kernels
    # read integer inputs from here so they stay exact regardless of the compute
    # precision `T` (issue #71). Non-integral / out-of-range entries map to the
    # sentinel -1 so a count kernel reading one lands out of support (k >= 0
    # fails), matching the CPU rejection of a non-integral count; rows that no
    # count kernel reads (continuous observations) simply ignore the sentinel.
    observed_int::Matrix{Int64}
    trip_counts::Vector{Int32}
    loop_starts::Vector{Int32}
    # When one column carries the whole batch (issue #153): the observation stream
    # is identical in every chain (shared model arguments + a single constraint
    # set), so `observed`/`observed_int` hold ONE column and the kernels broadcast
    # it. `false` keeps the per-chain `C`-column staging for genuinely per-chain
    # constraints (batched argument tuples).
    shared::Bool
    # Observation-parallel tiling anchors for the detected tileable loop (issue
    # #153); zero when no loop was flagged. `base_cursor` is the number of observed
    # rows emitted before the loop body (0-based), `stride` the observed rows one
    # body iteration consumes, so iteration `t` (0-based) reads at 1-based cursor
    # `base_cursor + 1 + t*stride`.
    tile_base_cursor::Int32
    tile_stride::Int32
end

# Mutable staging cursor threaded through the observation walk: the running loop
# counter plus, for the one loop flagged tileable (issue #153), the captured
# base-cursor / per-iteration stride. `tile_loop_id == 0` disables capture.
mutable struct DeviceStageLoopState
    count::Int32
    tile_loop_id::Int32
    base_cursor::Int32
    stride::Int32
end
DeviceStageLoopState(tile_loop_id::Integer) = DeviceStageLoopState(Int32(0), Int32(tile_loop_id), Int32(0), Int32(0))

# Round to an exact Int64 when the (Float64-staged) value is integral and within
# the Int64-exact range; otherwise the out-of-support sentinel -1.
@inline function _device_exact_int(v::Real)
    (isfinite(v) && v == round(v) && abs(v) <= 9.007199254740992e15) || return Int64(-1)
    return round(Int64, v)
end

function _device_staging_environment(
    model::TeaModel,
    backend::BackendExecutionPlan,
    args,
    batch_size::Int,
    ::Type{T},
) where {T}
    layout = executionplan(model).environment_layout
    env = BatchedPlanEnvironment(
        layout,
        backend.numeric_slots,
        backend.index_slots,
        backend.generic_slots,
        batch_size,
        T,
    )
    fill!(env.assigned, false)

    argument_slots = layout.argument_slots
    argument_count = length(argument_slots)
    if args isa Tuple
        length(args) == argument_count ||
            throw(DimensionMismatch("expected $argument_count model arguments, got $(length(args))"))
        for (slot, value) in zip(argument_slots, args)
            _batched_environment_set_shared!(env, slot, value)
        end
    else
        length(args) == batch_size ||
            throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
        values = Vector{Any}(undef, batch_size)
        for argument_index = 1:argument_count
            slot = argument_slots[argument_index]
            for batch_index = 1:batch_size
                batch_args = args[batch_index]
                length(batch_args) == argument_count ||
                    throw(DimensionMismatch("expected $argument_count model arguments, got $(length(batch_args))"))
                values[batch_index] = batch_args[argument_index]
            end
            _batched_environment_set!(env, slot, values)
        end
    end
    return env
end

function _stage_observed_row!(rows, step, env, constraints, dummy_params, ::Type{T}) where {T}
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    row = Vector{T}(undef, size(dummy_params, 2))
    _batched_choice_numeric_values!(row, step.parameter_slot, dummy_params, constraints, address_parts)
    push!(rows, row)
    return nothing
end

function _stage_deterministic!(step, env)
    slot = step.binding_slot
    try
        if env.index_slots[slot]
            _eval_backend_index_value_expr!(view(env.index_values, slot, :), env, step.expr)
            env.assigned[slot] = true
        elseif env.numeric_slots[slot]
            _eval_backend_numeric_expr!(view(env.numeric_values, slot, :), env, step.expr)
            env.assigned[slot] = true
        end
    catch err
        # A deterministic that depends on latent parameters (unavailable during
        # staging) or on a generic value simply cannot be staged; it is only needed
        # here if it feeds a loop iterable / address, in which case that evaluation
        # raises its own precise error below.
        (err isa BatchedBackendFallback || err isa ArgumentError) || rethrow()
    end
    return nothing
end

function _stage_step!(
    rows,
    step::BackendChoicePlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    isnothing(step.parameter_slot) || return nothing # latent: no observed row
    _stage_observed_row!(rows, step, env, constraints, dummy_params, T)
    return nothing
end

# binomial trials is an integer, parameter-independent argument. Stage it as a
# LEADING row (exact through the Float64 staging traversal) so the device reads
# the exact `n` from the integer observation buffer regardless of the compute
# precision `T` (issue #71). The observed count `k` (if any) stages next, exactly
# as the generic choice step; the device binomial step consumes both.
#
# When `trials` is a direct argument / constant its value resolves here exactly.
# When it is a deterministic binding the host env cannot always resolve (its
# in-kernel value is computed later), the row is staged as NaN -> integer
# sentinel -1, and the device binomial step falls back to evaluating `step.trials`
# in-kernel from the float slots (the pre-issue-#71 behavior).
function _stage_step!(
    rows,
    step::BackendBinomialChoicePlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    trials_row = Vector{T}(undef, size(dummy_params, 2))
    resolved = true
    try
        _eval_backend_numeric_expr!(trials_row, env, step.trials)
    catch err
        (err isa BatchedBackendFallback || err isa ArgumentError) || rethrow()
        resolved = false
    end
    resolved || fill!(trials_row, T(NaN))
    push!(rows, trials_row)
    isnothing(step.parameter_slot) || return nothing # latent value: no count row
    _stage_observed_row!(rows, step, env, constraints, dummy_params, T)
    return nothing
end

# mvnormaldense: the constant scale_tril factor rides the observation buffer.
# Its column-major packed lower triangle stages as d(d+1)/2 rows FIRST, then an
# observed value stages d more; the kernel consumes them in the same order.
function _stage_step!(
    rows,
    step::BackendMvNormalDenseChoicePlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    d = step.value_length
    env.assigned[step.scale_tril.slot] || throw(
        ArgumentError(
            "device staging requires the mvnormaldense scale_tril binding to be resolvable on the host (a model argument or captured constant)",
        ),
    )
    storage = env.generic_values[step.scale_tril.slot]
    batch_size = size(dummy_params, 2)
    for col = 1:d, row = col:d
        packed_row = Vector{T}(undef, batch_size)
        for batch_index = 1:batch_size
            scale = _backend_mvnormaldense_scale_matrix(storage[batch_index], d)
            packed_row[batch_index] = T(scale[row, col])
        end
        push!(rows, packed_row)
    end
    isnothing(step.parameter_slot) || return nothing # latent: no observed value rows
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    block = Matrix{T}(undef, d, batch_size)
    _batched_choice_vector_values!(block, nothing, d, dummy_params, constraints, address_parts)
    for component_index = 1:d
        push!(rows, block[component_index, :])
    end
    return nothing
end

# bernoullilogit whose eta is the fused GLM linear predictor (issues #135/#150):
# the covariate column `X[:, index]` is per-observation CONSTANT data (like
# observed values), so it rides the observation buffer with addresses erased.
# Staging pushes `D` covariate rows FIRST (one per coefficient dimension,
# resolving the loop-indexed column on the host), then the observed y row; the
# device GLM step consumes them in exactly that cursor order. A generic scalar
# eta stages only the observed y row (no leading covariate rows).
function _stage_step!(
    rows,
    step::BackendBernoulliLogitChoicePlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    step.eta isa BackendLinearPredictorExpr &&
        _stage_linear_predictor_covariates!(rows, step.eta, env, dummy_params, T)
    isnothing(step.parameter_slot) || return nothing # latent: no observed row (bernoullilogit is observed-only)
    _stage_observed_row!(rows, step, env, constraints, dummy_params, T)
    return nothing
end

function _stage_linear_predictor_covariates!(rows, eta::BackendLinearPredictorExpr, env, dummy_params, ::Type{T}) where {T}
    env.assigned[eta.matrix_slot] || throw(
        ArgumentError(
            "device staging requires the linear-predictor covariate matrix to be a resolvable model argument (a captured constant or direct argument)",
        ),
    )
    # the covariate column index is the loop iterator (already set by the loop
    # staging walk); it is shared across the batch, so resolve it once
    column = _eval_backend_index_value_expr(env, eta.index, 1)
    matrix_storage = env.generic_values[eta.matrix_slot]
    batch_size = size(dummy_params, 2)
    for d = 1:eta.coef_length
        covariate_row = Vector{T}(undef, batch_size)
        for batch_index = 1:batch_size
            matrix = matrix_storage[batch_index]
            covariate_row[batch_index] = T(matrix[d, column])
        end
        push!(rows, covariate_row)
    end
    return nothing
end

# Broadcast (vectorized) normal observation (issue #134): a VECTOR observation
# `y[1..M]` scored per element against `normal(mu_i, sigma)`. Like a loop it carries
# a dynamic element count `M`, so it takes a trip-count slot -- kept aligned with the
# device plan's `count_id` by the shared pre-order loop counter, since both walks
# cover `backend.steps` in the same order -- filled here with `M`. Its per-observation
# covariate vectors ride the observation buffer (addresses erased) exactly like the
# GLM covariate column: for each element `i`, push the `C` covariate values (distinct
# covariate slots in the mu-then-sigma first-encounter order the lowering assigns
# offsets) followed by the observed `y[i]`, so one element consumes `C + 1` cursor
# rows and the device handler reads them cursor-first. A covariate slot is read from
# generic env storage and indexed per element (a scalar generic broadcasts),
# mirroring the host `_eval_backend_broadcast_numeric!` generic-storage branch.
function _stage_step!(
    rows,
    step::BackendBroadcastNormalChoicePlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    loop_counter.count += Int32(1)
    lid = loop_counter.count
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    observed = _batched_broadcast_observed_values(env, address_parts, constraints)
    n = _broadcast_uniform_length(observed)
    trip_counts[lid] = Int32(n)
    covariate_slots = _broadcast_covariate_slots(step.mu, step.sigma, env.numeric_slots, env.index_slots)
    batch_size = size(dummy_params, 2)
    for element_index = 1:n
        for slot in covariate_slots
            covariate_row = Vector{T}(undef, batch_size)
            for batch_index = 1:batch_size
                value = env.generic_values[slot][batch_index]
                scalar = value isa AbstractVector ? value[element_index] : value
                covariate_row[batch_index] = T(_require_numeric_value(env, scalar, "broadcast covariate"))
            end
            push!(rows, covariate_row)
        end
        y_row = Vector{T}(undef, batch_size)
        for batch_index = 1:batch_size
            y_row[batch_index] = T(observed[batch_index][element_index])
        end
        push!(rows, y_row)
    end
    return nothing
end

# A vector observation stages as `value_length` consecutive rows in component
# order; the kernel-side cursor advances by the step's compile-time dimension,
# preserving the pre-order alignment invariant with a stride. lkjcholesky
# observations stage their packed lower triangle the same way (value_length =
# d(d+1)/2 rows).
function _stage_step!(
    rows,
    step::Union{BackendMvNormalChoicePlanStep,BackendDirichletChoicePlanStep,BackendLKJCholeskyChoicePlanStep},
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    isnothing(step.parameter_slot) || return nothing # latent: no observed rows
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)
    block = Matrix{T}(undef, step.value_length, size(dummy_params, 2))
    _batched_choice_vector_values!(block, nothing, step.value_length, dummy_params, constraints, address_parts)
    for component_index = 1:step.value_length
        push!(rows, block[component_index, :])
    end
    return nothing
end

function _stage_step!(
    rows,
    step::BackendDeterministicPlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    _stage_deterministic!(step, env)
    return nothing
end

function _stage_step!(
    rows,
    step::BackendLoopPlanStep,
    env,
    constraints,
    dummy_params,
    trip_counts,
    loop_starts,
    loop_counter,
    ::Type{T},
) where {T}
    loop_counter.count += Int32(1)
    lid = loop_counter.count
    reference = _batched_index_iterable_reference(env, step.iterable)
    Base.step(reference) == 1 ||
        throw(ArgumentError("device staging only supports unit-step loop ranges; see device_lowering_report(model)"))
    trip_counts[lid] = Int32(length(reference))
    loop_starts[lid] = isempty(reference) ? Int32(0) : Int32(first(reference))

    capture = lid == loop_counter.tile_loop_id
    capture && (loop_counter.base_cursor = Int32(length(rows)))
    had_previous = env.assigned[step.iterator_slot]
    previous = had_previous ? copy(env.index_values[step.iterator_slot, :]) : Int[]
    first_iteration = true
    for item in reference
        _batched_environment_set_shared!(env, step.iterator_slot, item)
        _stage_steps!(rows, step.body, env, constraints, dummy_params, trip_counts, loop_starts, loop_counter, T)
        if capture && first_iteration
            loop_counter.stride = Int32(length(rows)) - loop_counter.base_cursor
            first_iteration = false
        end
    end
    _batched_environment_restore!(env, step.iterator_slot, previous, had_previous)
    return nothing
end

# Function barrier for the recursive step walk: iterate the heterogeneous
# compiled-step tuple as a `Vector{Any}` so type inference/codegen does not
# specialize the whole (nested) tuple recursion. On the Julia 1.10 aarch64 build
# that specialization degrades into a SIGILL codegen crash (issue #131); the same
# over-specialization class was handled the same way for #88. This is the
# one-time host staging path, so the dynamic dispatch cost is immaterial.
function _stage_steps!(rows, steps, env, constraints, dummy_params, trip_counts, loop_starts, loop_counter, ::Type{T}) where {T}
    for step in collect(Any, steps)
        _stage_step!(rows, step, env, constraints, dummy_params, trip_counts, loop_starts, loop_counter, T)
    end
    return nothing
end

function _stage_device_observations(
    model::TeaModel,
    backend::BackendExecutionPlan,
    plan::DeviceExecutionPlan{T},
    args,
    constraints,
    batch_size::Int;
    tile_loop_id::Integer=0,
) where {T}
    # `backend` is the SIGNATURE-specific backend plan (issue #95, PR-4): a step
    # is observed iff it carries no parameter slot in this signature, so a
    # bound-but-constrained choice stages an observed row and a bound-but-
    # unconstrained choice does not -- matching the CPU signature layout.
    # Stage the observation stream in Float64 (exact for integers up to 2^53) so
    # integer inputs survive before splitting into the compute-precision float
    # buffer and the exact-integer mirror (issue #71). Float64 -> T is identical
    # to a direct T stage for every representable value, so the float path is
    # unchanged.
    env = _device_staging_environment(model, backend, args, batch_size, Float64)
    dummy_params = Matrix{Float64}(undef, 0, batch_size)
    rows = Vector{Vector{Float64}}()
    trip_counts = zeros(Int32, Int(plan.loop_count))
    loop_starts = zeros(Int32, Int(plan.loop_count))
    loop_counter = DeviceStageLoopState(tile_loop_id)
    _stage_steps!(rows, backend.steps, env, constraints, dummy_params, trip_counts, loop_starts, loop_counter, Float64)

    # The observation stream is identical in every chain -- so it can be staged as
    # ONE column and broadcast (issue #153) -- exactly when BOTH the model arguments
    # AND the constraints are shared across the batch: a single argument tuple (not a
    # per-chain vector of tuples) and a single constraint set (not a per-column vector
    # of choicemaps). Either being per-chain keeps the full `C`-column staging.
    shared = (args isa Tuple) && !(constraints isa AbstractVector)
    ncols = shared ? 1 : batch_size
    observed = Matrix{T}(undef, length(rows), ncols)
    observed_int = Matrix{Int64}(undef, length(rows), ncols)
    for (row_index, row) in enumerate(rows)
        for col = 1:ncols
            @inbounds v = row[col]
            @inbounds observed[row_index, col] = T(v)
            @inbounds observed_int[row_index, col] = _device_exact_int(v)
        end
    end
    return DeviceStagingBundle{T}(
        observed,
        observed_int,
        trip_counts,
        loop_starts,
        shared,
        loop_counter.base_cursor,
        loop_counter.stride,
    )
end
