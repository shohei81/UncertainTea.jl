mutable struct BatchedLogjointWorkspace{BP,CP,E}
    backend_plan::BP
    compiled_plan::CP
    environment::E
    # Signature-specific execution plan and its parameter layout (issue #95,
    # PR-4): the observed/latent split (and therefore the slot layout, the backend
    # plan, and the compiled plan above) is derived from the conditioning
    # signature, matching the CPU `logjoint` path. Static per signature.
    plan::ExecutionPlan
    layout::ParameterLayout
    parameter_count::Int
    constrained_parameter_count::Int
    argument_count::Int
    argument_slots::Vector{Int}
    constrained_buffer::Base.RefValue{Any}
    batched_environment::Base.RefValue{Any}
    batched_totals_buffer::Base.RefValue{Any}
    batched_constrained_buffer::Base.RefValue{Any}
    batched_logabsdet_buffer::Base.RefValue{Any}
    batched_argument_buffer::Base.RefValue{Any}
end

struct BatchedGradientObjective{M,W}
    model::M
    workspace::W
    args::Any
    constraints::ChoiceMap
end

function (objective::BatchedGradientObjective)(theta)
    return _logjoint_unconstrained_with_workspace!(
        objective.model,
        objective.workspace,
        theta,
        objective.args,
        objective.constraints,
    )
end

struct BatchedGradientColumnCache{F,C,V}
    objective::F
    config::C
    buffer::V
end

struct BatchedFlatGradientObjective{M,W,A,C}
    model::M
    workspace::W
    args::A
    constraints::C
    parameter_count::Int
    batch_size::Int
end

function (objective::BatchedFlatGradientObjective)(theta)
    params = reshape(theta, objective.parameter_count, objective.batch_size)
    totals = _batched_totals_buffer!(objective.workspace, objective.batch_size, eltype(theta))
    _batched_logjoint_unconstrained_with_workspace!(
        totals,
        objective.model,
        objective.workspace,
        params,
        objective.args,
        objective.constraints,
    )

    total = zero(eltype(theta))
    for value in totals
        total += value
    end
    return total
end

struct BatchedFlatGradientCache{O,C,B}
    objective::O
    config::C
    flat_buffer::B
end

struct BatchedBackendGradientCache{W,S,G,A,C}
    workspace::W
    slot_gradients::S
    gradient_scratch::G
    args::A
    constraints::C
    # constrained VALUE row -> unconstrained gradient-seed row (issue #36): a
    # scalar step's `parameter_slot` is its slot's value row (correct for
    # reading the constrained matrix), but gradient buffers are indexed by
    # UNCONSTRAINED rows, and the two drift apart after any dimension-changing
    # (simplex/cholesky) slot. Entries with no unconstrained counterpart (the
    # extra simplex/cholesky value rows) hold 0.
    seed_rows::Vector{Int}
    # observed-loop observation staging (issue #141): loop step -> (iterable,
    # gathered observation vector). The cache's constraints are fixed for its
    # lifetime, so the per-address constraint lookups run once per (step,
    # iterable) instead of once per gradient call; entries are revalidated
    # against the loop's current iterable before reuse.
    observed_loop_values::IdDict{Any,Any}
    # observed-loop sufficient statistics (issue #146): loop step -> (iterable,
    # family stats or nothing). For an iid exponential-family loop with
    # loop-invariant parameters the whole observation reduction collapses to a
    # few cached numbers; `nothing` records that the staged data cannot take
    # the closed form (so the O(observations) scan runs once, not per call).
    observed_loop_stats::IdDict{Any,Any}
end

function _backend_gradient_seed_rows(layout::ParameterLayout)
    seed_rows = zeros(Int, layout.value_count)
    for slot in layout.slots
        # only dimension-preserving slots map 1:1; the dimension-changing
        # transforms handle their own seeding through explicit parameter rows
        slot.dimension == slot.value_length || continue
        for component = 0:(slot.value_length-1)
            seed_rows[slot.value_index+component] = slot.index + component
        end
    end
    return seed_rows
end

# Chain-block threading plan for the analytic backend gradient (issue #143).
# `ranges` partitions the batch columns into contiguous, disjoint blocks;
# `caches` holds one INDEPENDENT `BatchedBackendGradientCache` per block, each
# sized for its block width (its own workspace scratch, slot gradients, and
# observation staging) so blocks can run under `Threads.@threads` writing
# disjoint slices of the shared totals/gradient buffers. `nothing` on the
# gradient cache means the work-size gate kept this problem serial. Built once
# at construction; the partition is fixed for the cache's lifetime.
struct BatchedGradientThreadPlan{C}
    ranges::Vector{UnitRange{Int}}
    caches::C
    reject_invalid_parameters::Bool
end

struct BatchedLogjointGradientCache{C,B,F,G<:AbstractMatrix,T}
    model::TeaModel
    column_caches::C
    backend_cache::B
    flat_cache::F
    gradient_buffer::G
    parameter_count::Int
    batch_size::Int
    # analytic-gradient chain-block threading plan, or `nothing` when serial
    # (issue #143). Only the `backend_cache` tier is threaded.
    thread_plan::T
    # Lane-compaction scratch (issue #160). Masked batched NUTS runs a full-width
    # gradient every leapfrog leaf even when most chains have finished/diverged;
    # once the active fraction drops below ~50% the sampler gathers the active
    # columns into `compact_params[:, 1:k]`, evaluates the analytic backend
    # gradient over just those k columns into `compact_logjoint[1:k]` /
    # `compact_gradient[:, 1:k]`, and scatters back to the active lanes (see
    # `_batched_compact_logjoint_and_gradient!`). `compact_index[slot]` records the
    # original column of the `slot`-th gathered lane. All are pre-sized to the full
    # batch width so the gather/scatter layer never allocates per step; only the
    # analytic tier is batch-width-independent enough to be compacted BITWISE, so
    # these are unused for the ForwardDiff tiers.
    compact_params::G
    compact_gradient::G
    compact_logjoint::Vector{Float64}
    compact_index::Vector{Int}
end

# `reject_invalid_parameters=true` puts the workspace's compiled-plan walk in
# Stan-style reject mode (issue #157): invalid distribution parameters score
# -Inf instead of throwing. Sampler-owned workspaces enable it; the public
# batched logjoint/gradient APIs keep the throwing default.
function BatchedLogjointWorkspace(model::TeaModel, constraints=choicemap(); reject_invalid_parameters::Bool=false)
    resolved = _resolve_signature_plan(model, _representative_constraints(constraints))
    plan = resolved.plan
    layout = plan.parameter_layout
    return BatchedLogjointWorkspace(
        _signature_backend_plan(model, resolved),
        resolved.compiled,
        PlanEnvironment(plan.environment_layout; reject_invalid_parameters=reject_invalid_parameters),
        plan,
        layout,
        parametercount(layout),
        parametervaluecount(layout),
        length(modelspec(model).arguments),
        copy(plan.environment_layout.argument_slots),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
    )
end

# Parameter-vector length now follows the CONDITIONING SIGNATURE, not the
# syntactic default layout (issue #95, PR-4): constraining a bound choice drops
# its slot; leaving an unbound choice unconstrained adds one. (PR-5 owns naming
# the signature in the message.)
function _batched_signature_layout(model::TeaModel, constraints)
    return _resolve_signature_plan(model, _representative_constraints(constraints)).plan.parameter_layout
end

function _validate_batched_unconstrained_params(model::TeaModel, params::AbstractMatrix, constraints=choicemap())
    representative = _representative_constraints(constraints)
    layout = _batched_signature_layout(model, constraints)
    expected = parametercount(layout)
    size(params, 1) == expected ||
        throw(_signature_length_error(model, layout, representative, expected, size(params, 1)))
    return size(params, 2)
end

function _validate_batched_constrained_params(model::TeaModel, params::AbstractMatrix, constraints=choicemap())
    representative = _representative_constraints(constraints)
    layout = _batched_signature_layout(model, constraints)
    expected = parametervaluecount(layout)
    size(params, 1) == expected || throw(
        _signature_length_error(
            model,
            layout,
            representative,
            expected,
            size(params, 1);
            space="constrained-space parameters",
        ),
    )
    return size(params, 2)
end

# Validation-only variant (no model in scope): checks the container shape but
# does not complete default arguments.
_validate_batched_args(args::Tuple, batch_size::Int) = args

function _validate_batched_args(args::AbstractVector, batch_size::Int)
    length(args) == batch_size || throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
    for batch_args in args
        batch_args isa Tuple || throw(ArgumentError("batched args must be a tuple or a vector of tuples"))
    end
    return args
end

function _validate_batched_args(model::TeaModel, args::Tuple, batch_size::Int)
    return _complete_model_args(model, args)
end

function _validate_batched_args(model::TeaModel, args::AbstractVector, batch_size::Int)
    length(args) == batch_size || throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
    completed = Vector{Tuple}(undef, length(args))
    for (index, batch_args) in enumerate(args)
        batch_args isa Tuple || throw(ArgumentError("batched args must be a tuple or a vector of tuples"))
        completed[index] = _complete_model_args(model, batch_args)
    end
    return completed
end

function _validate_batched_constraints(constraints::ChoiceMap, batch_size::Int)
    return constraints
end

function _validate_batched_constraints(constraints::AbstractVector, batch_size::Int)
    length(constraints) == batch_size ||
        throw(DimensionMismatch("expected $batch_size batched choicemaps, got $(length(constraints))"))
    for batch_constraints in constraints
        batch_constraints isa ChoiceMap || throw(ArgumentError("batched constraints must be a ChoiceMap or a vector of ChoiceMaps"))
    end
    return constraints
end

_batched_args(args::Tuple, index::Int) = args
_batched_args(args::AbstractVector, index::Int) = args[index]

_batched_constraints(constraints::ChoiceMap, index::Int) = constraints
_batched_constraints(constraints::AbstractVector, index::Int) = constraints[index]

function _reset_environment!(env::PlanEnvironment)
    fill!(env.assigned, false)
    return env
end

function _prepare_environment!(workspace::BatchedLogjointWorkspace, args::Tuple)
    length(args) == workspace.argument_count ||
        throw(DimensionMismatch("expected $(workspace.argument_count) model arguments, got $(length(args))"))

    env = workspace.environment
    _reset_environment!(env)
    for (slot, value) in zip(workspace.argument_slots, args)
        _environment_set!(env, slot, value)
    end
    return env
end

function _batched_environment!(workspace::BatchedLogjointWorkspace, batch_size::Int, ::Type{T}=Float64) where {T<:Real}
    env = workspace.batched_environment[]
    if !(env isa BatchedPlanEnvironment{T}) || env.batch_size != batch_size
        env = BatchedPlanEnvironment(
            workspace.environment.layout,
            workspace.backend_plan.numeric_slots,
            workspace.backend_plan.index_slots,
            workspace.backend_plan.generic_slots,
            batch_size,
            T,
        )
        workspace.batched_environment[] = env
    end
    return env
end

function _batched_argument_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    buffer = workspace.batched_argument_buffer[]
    if !(buffer isa Vector{Any}) || length(buffer) != batch_size
        buffer = Vector{Any}(undef, batch_size)
        workspace.batched_argument_buffer[] = buffer
    end
    return buffer
end

function _prepare_batched_environment!(
    workspace::BatchedLogjointWorkspace,
    args,
    batch_size::Int,
    ::Type{T}=Float64,
) where {T<:Real}
    env = _batched_environment!(workspace, batch_size, T)
    fill!(env.assigned, false)

    if args isa Tuple
        length(args) == workspace.argument_count ||
            throw(DimensionMismatch("expected $(workspace.argument_count) model arguments, got $(length(args))"))
        for (slot, value) in zip(workspace.argument_slots, args)
            _batched_environment_set_shared!(env, slot, value)
        end
    else
        length(args) == batch_size ||
            throw(DimensionMismatch("expected $batch_size batched argument tuples, got $(length(args))"))
        values = _batched_argument_buffer!(workspace, batch_size)
        for argument_index = 1:workspace.argument_count
            slot = workspace.argument_slots[argument_index]
            for batch_index = 1:batch_size
                batch_args = args[batch_index]
                length(batch_args) == workspace.argument_count ||
                    throw(DimensionMismatch("expected $(workspace.argument_count) model arguments, got $(length(batch_args))"))
                values[batch_index] = batch_args[argument_index]
            end
            _batched_environment_set!(env, slot, values)
        end
    end

    return env
end

function _constrained_buffer!(workspace::BatchedLogjointWorkspace, params::AbstractVector)
    buffer = workspace.constrained_buffer[]
    if !(buffer isa AbstractVector) || length(buffer) != workspace.constrained_parameter_count || eltype(buffer) != eltype(params)
        buffer = similar(params, workspace.constrained_parameter_count)
        workspace.constrained_buffer[] = buffer
    end
    return buffer
end

function _batched_totals_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    return _batched_totals_buffer!(workspace, batch_size, Float64)
end

function _batched_totals_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int, ::Type{T}) where {T<:Real}
    buffer = workspace.batched_totals_buffer[]
    if !(buffer isa Vector{T}) || length(buffer) != batch_size
        buffer = zeros(T, batch_size)
        workspace.batched_totals_buffer[] = buffer
    else
        fill!(buffer, zero(T))
    end
    return buffer
end

function _batched_constrained_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    return _batched_constrained_buffer!(workspace, workspace.constrained_parameter_count, batch_size, Float64)
end

function _batched_constrained_buffer!(
    workspace::BatchedLogjointWorkspace,
    constrained_parameter_count::Int,
    batch_size::Int,
    ::Type{T},
) where {T<:Real}
    buffer = workspace.batched_constrained_buffer[]
    if !(buffer isa Matrix{T}) || size(buffer) != (constrained_parameter_count, batch_size)
        buffer = Matrix{T}(undef, constrained_parameter_count, batch_size)
        workspace.batched_constrained_buffer[] = buffer
    end
    return buffer
end

function _batched_logabsdet_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int)
    return _batched_logabsdet_buffer!(workspace, batch_size, Float64)
end

function _batched_logabsdet_buffer!(workspace::BatchedLogjointWorkspace, batch_size::Int, ::Type{T}) where {T<:Real}
    buffer = workspace.batched_logabsdet_buffer[]
    if !(buffer isa Vector{T}) || length(buffer) != batch_size
        buffer = zeros(T, batch_size)
        workspace.batched_logabsdet_buffer[] = buffer
    else
        fill!(buffer, zero(T))
    end
    return buffer
end

function _logjoint_with_workspace!(
    workspace::BatchedLogjointWorkspace,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    length(params) == workspace.constrained_parameter_count ||
        throw(DimensionMismatch("expected $(workspace.constrained_parameter_count) parameters, got $(length(params))"))
    env = _prepare_environment!(workspace, args)
    if !isnothing(workspace.backend_plan)
        try
            return _score_backend_steps(workspace.backend_plan.steps, env, params, constraints)
        catch err
            # a scalar-backend capability gap (index-typed slot holding a
            # Pair/Tuple argument, an unsupported conditioning value, ...)
            # drops to the compiled plan, which scores those natively; the
            # environment is re-prepared because the aborted backend pass may
            # have partially mutated it. In reject mode (issue #157) the
            # backend's own parameter-validation throws (ArgumentError /
            # DomainError, e.g. poisson lambda <= 0) also drop to the compiled
            # plan, whose walk then scores the invalid step as -Inf.
            if !(
                err isa BatchedBackendFallback ||
                (env.reject_invalid_parameters && (err isa ArgumentError || err isa DomainError))
            )
                rethrow()
            end
            env = _prepare_environment!(workspace, args)
        end
    end
    return _score_compiled_steps(workspace.compiled_plan.steps, env, params, constraints)
end

function _logjoint_with_batched_backend!(
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    batch_size = size(params, 2)
    env = _prepare_batched_environment!(workspace, args, batch_size, eltype(params))
    totals = _batched_totals_buffer!(workspace, batch_size, eltype(params))
    _score_backend_steps!(totals, workspace.backend_plan.steps, env, params, constraints)
    return totals
end

function _fallback_batched_logjoint!(
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    batch_size = size(params, 2)
    values = Vector{float(eltype(params))}(undef, batch_size)
    for batch_index = 1:batch_size
        values[batch_index] = _logjoint_with_workspace!(
            workspace,
            view(params, :, batch_index),
            _batched_args(args, batch_index),
            _batched_constraints(constraints, batch_index),
        )
    end
    return values
end

function _logjoint_unconstrained_with_workspace!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    length(params) == workspace.parameter_count ||
        throw(DimensionMismatch("expected $(workspace.parameter_count) parameters, got $(length(params))"))

    layout = workspace.layout
    constrained = _constrained_buffer!(workspace, params)
    if _has_dependent_transforms(layout)
        # dependent transforms (reparam=:noncentered) need the plan walk; the
        # per-slot loop below would throw on their marker transforms. The walk
        # runs against the SIGNATURE plan/compiled plan so an observed value that
        # feeds a noncentered loc/scale resolves from the constraint (PR-3/PR-4).
        logabsdet = _dependent_transform_walk!(
            constrained,
            model,
            workspace.plan,
            workspace.compiled_plan,
            params,
            args,
            false,
            constraints;
            reject_invalid_parameters=workspace.environment.reject_invalid_parameters,
        )
        return _logjoint_with_workspace!(workspace, constrained, args, constraints) + logabsdet
    end
    logabsdet = workspace.parameter_count == 0 ? zero(float(eltype(params))) : zero(params[firstindex(params)])
    for slot in layout.slots
        logabsdet += _transform_slot_to_constrained!(constrained, slot, params)
    end
    return _logjoint_with_workspace!(workspace, constrained, args, constraints) + logabsdet
end

function _logjoint_unconstrained_batched_backend!(
    destination::AbstractVector,
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    parameter_count, batch_size = size(params)
    length(destination) == batch_size ||
        throw(DimensionMismatch("expected unconstrained batched destination of length $batch_size, got $(length(destination))"))
    layout = workspace.layout
    value_type = eltype(destination)
    constrained = _batched_constrained_buffer!(workspace, workspace.constrained_parameter_count, batch_size, value_type)
    logabsdet = _batched_logabsdet_buffer!(workspace, batch_size, value_type)
    for slot in layout.slots
        source_indices = parameterindices(slot)
        destination_indices = parametervalueindices(slot)
        if slot.transform isa IdentityTransform
            copyto!(view(constrained, destination_indices, :), view(params, source_indices, :))
        elseif slot.transform isa NoncenteredTransform
            # z-space plan: the noncentered step scores N(z; 0, 1) and
            # materializes theta itself, so the pre-pass passes z through with
            # no Jacobian term
            copyto!(view(constrained, destination_indices, :), view(params, source_indices, :))
        elseif slot.transform isa VectorIdentityTransform
            copyto!(view(constrained, destination_indices, :), view(params, source_indices, :))
        elseif slot.transform isa VectorLogTransform
            for batch_index = 1:batch_size
                for (source_index, destination_index) in zip(source_indices, destination_indices)
                    unconstrained_value = params[source_index, batch_index]
                    constrained[destination_index, batch_index] = exp(unconstrained_value)
                    logabsdet[batch_index] += unconstrained_value
                end
            end
        elseif slot.transform isa VectorLogitTransform
            for batch_index = 1:batch_size
                for (source_index, destination_index) in zip(source_indices, destination_indices)
                    unconstrained_value = params[source_index, batch_index]
                    constrained_value = to_constrained(LogitTransform(), unconstrained_value)
                    constrained[destination_index, batch_index] = constrained_value
                    logabsdet[batch_index] += logabsdetjac(LogitTransform(), unconstrained_value)
                end
            end
        elseif slot.transform isa LogTransform
            for batch_index = 1:batch_size
                unconstrained_value = params[first(source_indices), batch_index]
                constrained[first(destination_indices), batch_index] = exp(unconstrained_value)
                logabsdet[batch_index] += unconstrained_value
            end
        elseif slot.transform isa LogitTransform
            for batch_index = 1:batch_size
                unconstrained_value = params[first(source_indices), batch_index]
                constrained_value = to_constrained(slot.transform, unconstrained_value)
                constrained[first(destination_indices), batch_index] = constrained_value
                logabsdet[batch_index] += logabsdetjac(slot.transform, unconstrained_value)
            end
        elseif slot.transform isa SimplexTransform
            for batch_index = 1:batch_size
                constrained_view = view(constrained, destination_indices, batch_index)
                unconstrained_view = view(params, source_indices, batch_index)
                _to_constrained_simplex!(constrained_view, slot.transform, unconstrained_view)
                logabsdet[batch_index] += _simplex_logabsdet(constrained_view)
            end
        elseif slot.transform isa CholeskyCorrTransform
            for batch_index = 1:batch_size
                constrained_view = view(constrained, destination_indices, batch_index)
                unconstrained_view = view(params, source_indices, batch_index)
                logabsdet[batch_index] +=
                    _to_constrained_cholesky_corr!(constrained_view, slot.transform, unconstrained_view)
            end
        else
            throw(ArgumentError("unsupported parameter transform $(typeof(slot.transform))"))
        end
    end
    totals = _logjoint_with_batched_backend!(workspace, constrained, args, constraints)
    for batch_index = 1:batch_size
        destination[batch_index] = totals[batch_index] + logabsdet[batch_index]
    end
    return destination
end

function _logjoint_unconstrained_batched_backend!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    values = Vector{eltype(params)}(undef, size(params, 2))
    return _logjoint_unconstrained_batched_backend!(values, model, workspace, params, args, constraints)
end

function _fallback_batched_logjoint_unconstrained!(
    destination::AbstractVector,
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    batch_size = size(params, 2)
    length(destination) == batch_size ||
        throw(DimensionMismatch("expected unconstrained batched destination of length $batch_size, got $(length(destination))"))
    for batch_index = 1:batch_size
        destination[batch_index] = _logjoint_unconstrained_with_workspace!(
            model,
            workspace,
            view(params, :, batch_index),
            _batched_args(args, batch_index),
            _batched_constraints(constraints, batch_index),
        )
    end
    return destination
end

function _fallback_batched_logjoint_unconstrained!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    values = Vector{eltype(params)}(undef, size(params, 2))
    return _fallback_batched_logjoint_unconstrained!(values, model, workspace, params, args, constraints)
end

function _batched_logjoint_unconstrained_with_workspace!(
    destination::AbstractVector,
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    if !isnothing(workspace.backend_plan)
        try
            return _logjoint_unconstrained_batched_backend!(destination, model, workspace, params, args, constraints)
        catch err
            # reject mode (issue #157): a vectorized-backend parameter-validation
            # throw for ONE lane must not kill the batch -- drop to the per-column
            # fallback, where only the offending column scores -Inf
            if !(
                err isa BatchedBackendFallback ||
                (
                    workspace.environment.reject_invalid_parameters &&
                    (err isa ArgumentError || err isa DomainError)
                )
            )
                rethrow()
            end
        end
    end
    return _fallback_batched_logjoint_unconstrained!(destination, model, workspace, params, args, constraints)
end

function _batched_logjoint_unconstrained_with_workspace!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    params::AbstractMatrix,
    args,
    constraints,
)
    values = Vector{eltype(params)}(undef, size(params, 2))
    return _batched_logjoint_unconstrained_with_workspace!(values, model, workspace, params, args, constraints)
end
