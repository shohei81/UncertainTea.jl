abstract type AbstractTemperedNUTSSchedulerIR end
abstract type AbstractTemperedNUTSSchedulerBlock end
abstract type AbstractTemperedNUTSSchedulerDescriptor end

struct TemperedNUTSIdleIR <: AbstractTemperedNUTSSchedulerIR end

struct TemperedNUTSExpandIR <: AbstractTemperedNUTSSchedulerIR
    active_depth::Int
    active_depth_count::Int
    remaining_steps::Int
    active_particles::BitVector
    directions::Vector{Int}
end

struct TemperedNUTSMergeIR <: AbstractTemperedNUTSSchedulerIR
    active_depth::Int
    active_depth_count::Int
    active_particles::BitVector
end

struct TemperedNUTSDoneIR <: AbstractTemperedNUTSSchedulerIR end

struct TemperedNUTSIdleBlock <: AbstractTemperedNUTSSchedulerBlock
    ir::TemperedNUTSIdleIR
end

struct TemperedNUTSExpandBlock <: AbstractTemperedNUTSSchedulerBlock
    ir::TemperedNUTSExpandIR
    active_particles::BitVector
    directions::Vector{Int}
end

struct TemperedNUTSMergeBlock <: AbstractTemperedNUTSSchedulerBlock
    ir::TemperedNUTSMergeIR
    active_particles::BitVector
end

struct TemperedNUTSDoneBlock <: AbstractTemperedNUTSSchedulerBlock
    ir::TemperedNUTSDoneIR
end

struct TemperedNUTSIdleDescriptor <: AbstractTemperedNUTSSchedulerDescriptor
    block::TemperedNUTSIdleBlock
end

struct TemperedNUTSExpandDescriptor <: AbstractTemperedNUTSSchedulerDescriptor
    block::TemperedNUTSExpandBlock
    remaining_steps::Int
    active_particles::BitVector
    directions::Vector{Int}
end

struct TemperedNUTSMergeDescriptor <: AbstractTemperedNUTSSchedulerDescriptor
    block::TemperedNUTSMergeBlock
    active_particles::BitVector
end

struct TemperedNUTSDoneDescriptor <: AbstractTemperedNUTSSchedulerDescriptor
    block::TemperedNUTSDoneBlock
end

function _begin_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    max_tree_depth::Int,
    rng::AbstractRNG,
)
    scheduler = workspace.scheduler
    active_depth, active_depth_count = _select_tempered_nuts_depth_cohort!(
        workspace,
        continuations,
        max_tree_depth,
    )
    if active_depth_count == 0
        fill!(scheduler.cohort_active, false)
        scheduler.phase = TemperedNUTSSchedulerDone
        scheduler.remaining_steps = 0
        return scheduler.phase
    end

    copyto!(
        scheduler.cohort_active,
        _activate_tempered_nuts_depth_cohort!(workspace, continuations, active_depth),
    )
    scheduler.phase = TemperedNUTSSchedulerExpand
    scheduler.remaining_steps = 1 << active_depth
    _initialize_tempered_nuts_depth_cohort!(workspace, continuations, active_depth, rng)
    return scheduler.phase
end

function _tempered_nuts_scheduler_ir(workspace::TemperedNUTSMoveWorkspace)
    scheduler = workspace.scheduler
    if scheduler.phase === TemperedNUTSSchedulerIdle
        return TemperedNUTSIdleIR()
    elseif scheduler.phase === TemperedNUTSSchedulerExpand
        return TemperedNUTSExpandIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            scheduler.remaining_steps,
            copy(scheduler.cohort_active),
            copy(workspace.control.directions),
        )
    elseif scheduler.phase === TemperedNUTSSchedulerMerge
        return TemperedNUTSMergeIR(
            scheduler.active_depth,
            scheduler.active_depth_count,
            copy(scheduler.cohort_active),
        )
    end
    return TemperedNUTSDoneIR()
end

_tempered_nuts_scheduler_block(workspace::TemperedNUTSMoveWorkspace) =
    _tempered_nuts_scheduler_block(_tempered_nuts_scheduler_ir(workspace))

_tempered_nuts_scheduler_block(ir::TemperedNUTSIdleIR) = TemperedNUTSIdleBlock(ir)

function _tempered_nuts_scheduler_block(ir::TemperedNUTSExpandIR)
    return TemperedNUTSExpandBlock(ir, copy(ir.active_particles), copy(ir.directions))
end

function _tempered_nuts_scheduler_block(ir::TemperedNUTSMergeIR)
    return TemperedNUTSMergeBlock(ir, copy(ir.active_particles))
end

_tempered_nuts_scheduler_block(ir::TemperedNUTSDoneIR) = TemperedNUTSDoneBlock(ir)

_tempered_nuts_scheduler_descriptor(block::TemperedNUTSIdleBlock) = TemperedNUTSIdleDescriptor(block)

function _tempered_nuts_scheduler_descriptor(block::TemperedNUTSExpandBlock)
    return TemperedNUTSExpandDescriptor(
        block,
        block.ir.remaining_steps,
        copy(block.active_particles),
        copy(block.directions),
    )
end

function _tempered_nuts_scheduler_descriptor(block::TemperedNUTSMergeBlock)
    return TemperedNUTSMergeDescriptor(block, copy(block.active_particles))
end

_tempered_nuts_scheduler_descriptor(block::TemperedNUTSDoneBlock) = TemperedNUTSDoneDescriptor(block)

function _execute_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    ::TemperedNUTSIdleDescriptor,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    _begin_tempered_nuts_cohort_scheduler!(workspace, continuations, max_tree_depth, rng)
    return workspace.scheduler.phase
end

function _execute_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    descriptor::TemperedNUTSExpandDescriptor,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    ir = descriptor.block.ir
    scheduler = workspace.scheduler
    scheduler.active_depth = ir.active_depth
    scheduler.active_depth_count = ir.active_depth_count
    scheduler.remaining_steps = descriptor.remaining_steps
    copyto!(scheduler.cohort_active, descriptor.active_particles)
    copyto!(workspace.control.active, descriptor.active_particles)
    copyto!(workspace.control.directions, descriptor.directions)
    _expand_tempered_nuts_depth_cohort!(
        workspace,
        model,
        cache,
        args,
        constraints,
        proposal_location,
        proposal_log_scale,
        beta,
        step_size,
        divergence_threshold,
        initial_hamiltonian,
        inverse_mass_matrix,
        rng,
    )
    scheduler.phase = TemperedNUTSSchedulerMerge
    scheduler.remaining_steps = 0
    return scheduler.phase
end

function _execute_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    descriptor::TemperedNUTSMergeDescriptor,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    ir = descriptor.block.ir
    scheduler = workspace.scheduler
    scheduler.active_depth = ir.active_depth
    scheduler.active_depth_count = ir.active_depth_count
    copyto!(scheduler.cohort_active, descriptor.active_particles)
    copyto!(workspace.control.active, descriptor.active_particles)
    _merge_tempered_nuts_depth_cohort!(workspace, continuations, inverse_mass_matrix, rng)
    _begin_tempered_nuts_cohort_scheduler!(workspace, continuations, max_tree_depth, rng)
    return scheduler.phase
end

function _execute_tempered_nuts_cohort_scheduler!(
    workspace::TemperedNUTSMoveWorkspace,
    ::TemperedNUTSDoneDescriptor,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    workspace.scheduler.phase = TemperedNUTSSchedulerDone
    workspace.scheduler.remaining_steps = 0
    return workspace.scheduler.phase
end

function _continue_batched_tempered_nuts_cohorts!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    model::TeaModel,
    cache::BatchedLogjointGradientCache,
    args::Tuple,
    constraints::ChoiceMap,
    proposal_location::AbstractVector,
    proposal_log_scale::AbstractVector,
    beta::Float64,
    step_size::Float64,
    max_tree_depth::Int,
    divergence_threshold::Float64,
    initial_hamiltonian::AbstractVector,
    inverse_mass_matrix::AbstractVector,
    rng::AbstractRNG,
)
    _begin_tempered_nuts_cohort_scheduler!(workspace, continuations, max_tree_depth, rng)
    while workspace.scheduler.phase !== TemperedNUTSSchedulerDone
        block = _tempered_nuts_scheduler_block(workspace)
        descriptor = _tempered_nuts_scheduler_descriptor(block)
        _execute_tempered_nuts_cohort_scheduler!(
            workspace,
            descriptor,
            continuations,
            model,
            cache,
            args,
            constraints,
            proposal_location,
            proposal_log_scale,
            beta,
            step_size,
            max_tree_depth,
            divergence_threshold,
            initial_hamiltonian,
            inverse_mass_matrix,
            rng,
        )
    end
    return continuations
end
