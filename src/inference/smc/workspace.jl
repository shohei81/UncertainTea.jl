@enum TemperedNUTSSchedulerPhase::UInt8 begin
    TemperedNUTSSchedulerIdle = 0
    TemperedNUTSSchedulerExpand = 1
    TemperedNUTSSchedulerMerge = 2
    TemperedNUTSSchedulerDone = 3
end

mutable struct TemperedNUTSCohortWorkspace
    current_position::Matrix{Float64}
    current_momentum::Matrix{Float64}
    current_gradient::Matrix{Float64}
    current_logjoint::Vector{Float64}
    next_position::Matrix{Float64}
    next_momentum::Matrix{Float64}
    next_gradient::Matrix{Float64}
    next_logjoint::Vector{Float64}
    left_position::Matrix{Float64}
    left_momentum::Matrix{Float64}
    left_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    right_position::Matrix{Float64}
    right_momentum::Matrix{Float64}
    right_gradient::Matrix{Float64}
    right_logjoint::Vector{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    proposal_logjoint::Vector{Float64}
    logjoint_values::Vector{Float64}
    logjoint_gradient::Matrix{Float64}
    logproposal_values::Vector{Float64}
    logproposal_gradient::Matrix{Float64}
    proposal_noise::Matrix{Float64}
    subtree_log_weight::Vector{Float64}
    subtree_accept_stat_sum::Vector{Float64}
    subtree_accept_stat_count::Vector{Int}
    subtree_integration_steps::Vector{Int}
    subtree_proposal_energy::Vector{Float64}
    subtree_proposal_energy_error::Vector{Float64}
    subtree_turning::BitVector
    subtree_divergent::BitVector
    checkpoint_positions::Array{Float64,3}
    checkpoint_momenta::Array{Float64,3}
end

function TemperedNUTSCohortWorkspace(
    parameter_total::Int,
    num_particles::Int,
    max_tree_depth::Int=1,
)
    checkpoint_slots = max(max_tree_depth + 1, 1)
    return TemperedNUTSCohortWorkspace(
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        fill(-Inf, num_particles),
        zeros(Float64, num_particles),
        zeros(Int, num_particles),
        zeros(Int, num_particles),
        fill(Inf, num_particles),
        fill(Inf, num_particles),
        falses(num_particles),
        falses(num_particles),
        zeros(Float64, parameter_total, checkpoint_slots, num_particles),
        zeros(Float64, parameter_total, checkpoint_slots, num_particles),
    )
end

mutable struct TemperedNUTSCohortControlState
    active::BitVector
    subtree_active::BitVector
    valid::BitVector
    directions::Vector{Int}
end

function TemperedNUTSCohortControlState(num_particles::Int)
    return TemperedNUTSCohortControlState(
        falses(num_particles),
        falses(num_particles),
        falses(num_particles),
        Vector{Int}(undef, num_particles),
    )
end

mutable struct TemperedNUTSSchedulerState
    continuation_active::BitVector
    cohort_active::BitVector
    active_depth::Int
    active_depth_count::Int
    phase::TemperedNUTSSchedulerPhase
    remaining_steps::Int
end

function TemperedNUTSSchedulerState(num_particles::Int)
    return TemperedNUTSSchedulerState(
        falses(num_particles),
        falses(num_particles),
        0,
        0,
        TemperedNUTSSchedulerIdle,
        0,
    )
end

mutable struct TemperedNUTSMoveWorkspace{C,TW<:NUTSSubtreeWorkspace,CS<:NUTSContinuationState}
    parameter_total::Int
    num_particles::Int
    cache::C
    momentum::Matrix{Float64}
    proposal_particles::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    current_logjoint_gradient::Matrix{Float64}
    current_logproposal_gradient::Matrix{Float64}
    current_tempered_gradient::Matrix{Float64}
    current_tempered_values::Vector{Float64}
    proposal_logjoint_values::Vector{Float64}
    proposal_logproposal_values::Vector{Float64}
    proposal_tempered_values::Vector{Float64}
    proposal_logjoint_gradient::Matrix{Float64}
    proposal_logproposal_gradient::Matrix{Float64}
    proposal_tempered_gradient::Matrix{Float64}
    proposal_noise::Matrix{Float64}
    current_hamiltonian::Vector{Float64}
    directions::Vector{Int}
    valid::BitVector
    inverse_mass::Vector{Float64}
    sqrt_inverse_mass::Vector{Float64}
    depth_counts::Vector{Int}
    tree_workspaces::Vector{TW}
    continuations::Vector{CS}
    control::TemperedNUTSCohortControlState
    scheduler::TemperedNUTSSchedulerState
    cohort::TemperedNUTSCohortWorkspace
end

function TemperedNUTSMoveWorkspace(
    model::TeaModel,
    particles::AbstractMatrix,
    args=(),
    constraints=choicemap(),
    max_tree_depth::Int=1;
    adtype::Symbol=:auto,
)
    parameter_total, num_particles = size(particles)
    cache = BatchedLogjointGradientCache(model, particles, args, constraints; adtype=adtype)
    return TemperedNUTSMoveWorkspace(
        parameter_total,
        num_particles,
        cache,
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Vector{Float64}(undef, num_particles),
        Vector{Float64}(undef, num_particles),
        Vector{Float64}(undef, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Matrix{Float64}(undef, parameter_total, num_particles),
        Vector{Float64}(undef, num_particles),
        Vector{Int}(undef, num_particles),
        trues(num_particles),
        Vector{Float64}(undef, parameter_total),
        Vector{Float64}(undef, parameter_total),
        Int[],
        [NUTSSubtreeWorkspace(parameter_total, 0) for _ = 1:num_particles],
        [NUTSContinuationState(parameter_total) for _ = 1:num_particles],
        TemperedNUTSCohortControlState(num_particles),
        TemperedNUTSSchedulerState(num_particles),
        TemperedNUTSCohortWorkspace(parameter_total, num_particles, max_tree_depth),
    )
end

function _validate_tempered_nuts_move_workspace!(
    workspace::TemperedNUTSMoveWorkspace,
    particles::AbstractMatrix,
    inverse_mass_matrix::AbstractVector,
)
    size(particles, 1) == workspace.parameter_total ||
        throw(DimensionMismatch("expected particle matrix with $(workspace.parameter_total) rows, got $(size(particles, 1))"))
    size(particles, 2) == workspace.num_particles ||
        throw(DimensionMismatch("expected particle matrix with $(workspace.num_particles) columns, got $(size(particles, 2))"))
    length(inverse_mass_matrix) == workspace.parameter_total ||
        throw(
            DimensionMismatch(
                "expected inverse mass matrix of length $(workspace.parameter_total), got $(length(inverse_mass_matrix))",
            ),
        )
    for parameter_index = 1:workspace.parameter_total
        inverse_mass_value = Float64(inverse_mass_matrix[parameter_index])
        workspace.inverse_mass[parameter_index] = inverse_mass_value
        workspace.sqrt_inverse_mass[parameter_index] = sqrt(inverse_mass_value)
    end
    return workspace
end

function _tempered_nuts_depth_counts!(
    workspace::TemperedNUTSMoveWorkspace,
    max_tree_depth::Int,
)
    target_length = max(max_tree_depth - 1, 0)
    if length(workspace.depth_counts) != target_length
        resize!(workspace.depth_counts, target_length)
    end
    fill!(workspace.depth_counts, 0)
    return workspace.depth_counts
end

function _tempered_nuts_active_depth!(
    counts::AbstractVector{Int},
    continuations::AbstractVector{<:NUTSContinuationState},
    max_tree_depth::Int,
)
    max_tree_depth > 1 || return 0, 0
    length(counts) >= max_tree_depth - 1 ||
        throw(DimensionMismatch("expected depth-count scratch of length at least $(max_tree_depth - 1), got $(length(counts))"))
    fill!(counts, 0)
    active_depth = 0
    active_depth_count = 0
    for continuation in continuations
        if !_nuts_continuation_active(
            continuation.tree_depth,
            max_tree_depth,
            continuation.divergent,
            continuation.turning,
        )
            continue
        end
        depth = continuation.tree_depth
        counts[depth] += 1
        if counts[depth] > active_depth_count
            active_depth = depth
            active_depth_count = counts[depth]
        end
    end
    return active_depth, active_depth_count
end

function _select_tempered_nuts_depth_cohort!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    max_tree_depth::Int,
)
    scheduler = workspace.scheduler
    counts = _tempered_nuts_depth_counts!(workspace, max_tree_depth)
    fill!(counts, 0)
    fill!(scheduler.continuation_active, false)
    scheduler.active_depth = 0
    scheduler.active_depth_count = 0
    max_tree_depth > 1 || return 0, 0

    for particle_index in eachindex(continuations)
        continuation = continuations[particle_index]
        is_active = _nuts_continuation_active(
            continuation.tree_depth,
            max_tree_depth,
            continuation.divergent,
            continuation.turning,
        )
        scheduler.continuation_active[particle_index] = is_active
        is_active || continue
        depth = continuation.tree_depth
        counts[depth] += 1
        if counts[depth] > scheduler.active_depth_count
            scheduler.active_depth = depth
            scheduler.active_depth_count = counts[depth]
        end
    end

    return scheduler.active_depth, scheduler.active_depth_count
end

function _activate_tempered_nuts_depth_cohort!(
    workspace::TemperedNUTSMoveWorkspace,
    continuations::AbstractVector{<:NUTSContinuationState},
    cohort_depth::Int,
)
    active = workspace.control.active
    continuation_active = workspace.scheduler.continuation_active
    fill!(active, false)
    for particle_index in eachindex(continuations)
        active[particle_index] =
            continuation_active[particle_index] &&
            continuations[particle_index].tree_depth == cohort_depth
    end
    return active
end

function _reset_tempered_nuts_cohort_statistics!(
    workspace::TemperedNUTSMoveWorkspace,
)
    control = workspace.control
    cohort = workspace.cohort
    copyto!(control.subtree_active, control.active)
    fill!(cohort.subtree_log_weight, -Inf)
    fill!(cohort.subtree_accept_stat_sum, 0.0)
    fill!(cohort.subtree_accept_stat_count, 0)
    fill!(cohort.subtree_integration_steps, 0)
    fill!(cohort.subtree_proposal_energy, Inf)
    fill!(cohort.subtree_proposal_energy_error, Inf)
    fill!(cohort.subtree_turning, false)
    fill!(cohort.subtree_divergent, false)
    return workspace
end

function _tempered_nuts_active_depth(
    continuations::AbstractVector{<:NUTSContinuationState},
    max_tree_depth::Int,
)
    counts = zeros(Int, max(max_tree_depth - 1, 0))
    return _tempered_nuts_active_depth!(counts, continuations, max_tree_depth)
end
