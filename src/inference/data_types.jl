struct HMCMassAdaptationWindowSummary
    window_index::Int
    iteration_start::Int
    iteration_end::Int
    window_length::Int
    pooled_samples::Int
    weight_sum::Float64
    effective_count::Float64
    mean_weight::Float64
    clip_scale_start::Float64
    clip_scale_end::Float64
    updated::Bool
    mass_mean::Float64
    mass_min::Float64
    mass_max::Float64
end

struct HMCMassAdaptationSummary
    window_index::Int
    iteration_start::Int
    iteration_end::Int
    window_length::Int
    chains::Int
    num_updated::Int
    mean_pooled_samples::Float64
    mean_weight_sum::Float64
    mean_effective_count::Float64
    min_effective_count::Float64
    max_effective_count::Float64
    mean_weight::Float64
    mean_clip_scale_end::Float64
    mean_mass::Float64
    min_mass::Float64
    max_mass::Float64
end

struct HMCDiagnosticsSummary
    acceptance_rate::Float64
    divergence_rate::Float64
    step_sizes::Vector{Float64}
    mean_step_size::Float64
    mass_adaptation_windows::Vector{HMCMassAdaptationSummary}
end

"""
    HMCChain

A single sampled chain: the constrained and unconstrained draws plus per-draw
diagnostics (acceptance, energy, divergences, tree depths / integration steps)
and the adapted step size and mass matrix. Held per-chain inside an
[`HMCChains`](@ref) — every sampler entry point (including the single-chain
[`hmc`](@ref) / [`nuts`](@ref), issue #337) returns `HMCChains`; index into it
(`chains[1]`, `first(chains)`) to reach the per-chain diagnostics.
"""
struct HMCChain
    sampler::Symbol
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
    logjoint_values::Vector{Float64}
    acceptance_stats::Vector{Float64}
    energies::Vector{Float64}
    energy_errors::Vector{Float64}
    accepted::BitVector
    divergent::BitVector
    step_size::Float64
    mass_matrix::Vector{Float64}
    num_leapfrog_steps::Int
    max_tree_depth::Int
    tree_depths::Vector{Int}
    integration_steps::Vector{Int}
    target_accept::Float64
    mass_adaptation_windows::Vector{HMCMassAdaptationWindowSummary}
    # Dense metric (metric=:dense single-chain runs): the estimated posterior
    # covariance Sigma = M^{-1}. `nothing` for diagonal runs; when set, its
    # diagonal matches `mass_matrix`.
    dense_mass_matrix::Union{Nothing,Matrix{Float64}}
end

"""
    HMCChains

A collection of [`HMCChain`](@ref)s from the same model and conditioning, as
returned by [`hmc`](@ref), [`nuts`](@ref) (one chain), [`hmc_chains`](@ref),
[`nuts_chains`](@ref), [`batched_nuts`](@ref), [`batched_hmc`](@ref), and
[`batched_chees`](@ref). Pass it to [`summarize`](@ref), `rhat`, `ess`,
`predict`, `loo`, and the export helpers (`posterior_array`, `to_arviz_dict`,
`to_mcmcchains`).
"""
struct HMCChains{A,C}
    model::TeaModel
    args::A
    constraints::C
    chains::Vector{HMCChain}
end

# MH-within-Gibbs result (docs/mh-within-gibbs.md): the continuous block's
# samples and frozen adaptation, plus per-site discrete samples in trace
# order. `constraints` holds the user observations only; the discrete values
# live in `discrete_samples` (sites x samples, Bool sites stored as 0/1).
"""
    GibbsChain

Result of [`gibbs`](@ref) (Metropolis-within-Gibbs): the continuous block's
constrained/unconstrained draws and frozen adaptation, plus per-site discrete
samples in trace order and their acceptance rates. `constraints` holds only the
user observations; sampled discrete values live in `discrete_samples`.
"""
struct GibbsChain
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
    logjoint_values::Vector{Float64}
    acceptance_stats::Vector{Float64}
    accepted::BitVector
    divergent::BitVector
    step_size::Float64
    mass_matrix::Vector{Float64}
    # metric=:dense runs: the adapted dense inverse mass; `nothing` otherwise
    dense_mass_matrix::Union{Nothing,Matrix{Float64}}
    discrete_addresses::Vector{Any}
    discrete_samples::Matrix{Int}
    # per-site MH acceptance rate over ALL sweeps (warmup included)
    discrete_acceptance::Vector{Float64}
end

struct HMCParameterSummary
    index::Int
    binding::Symbol
    address::Any
    mean::Float64
    sd::Float64
    quantiles::Vector{Float64}
    rhat::Float64
    ess::Float64
    mcse_mean::Float64
    ess_tail::Float64
    mcse_quantiles::Vector{Float64}
    per_chain_means::Union{Nothing,Vector{Float64}}
    per_chain_sds::Union{Nothing,Vector{Float64}}
end

struct SamplerWarnings
    num_divergent::Vector{Int}
    ebfmi::Vector{Float64}
    treedepth_hits::Vector{Int}
    low_ess_parameters::Vector{String}
    high_rhat_parameters::Vector{String}
    # Threshold below which a chain's E-BFMI counts as a warning; recorded so
    # `has_warnings`/`show` honor the `check_diagnostics` kwarg.
    ebfmi_threshold::Float64
end

struct HMCSummary
    model::TeaModel
    space::Symbol
    quantile_probs::Vector{Float64}
    diagnostics::HMCDiagnosticsSummary
    parameters::Vector{HMCParameterSummary}
    warnings::SamplerWarnings
end

mutable struct DualAveragingState
    target_accept::Float64
    gamma::Float64
    t0::Float64
    kappa::Float64
    mu::Float64
    log_step_size::Float64
    log_step_size_avg::Float64
    hbar::Float64
    iteration::Int
end

mutable struct RunningVarianceState
    mean::Vector{Float64}
    m2::Vector{Float64}
    clipped_sample::Vector{Float64}
    window_length::Int
    count::Int
    weight_sum::Float64
    weight_square_sum::Float64
end

const _RUNNING_VARIANCE_CLIP_START = 4
const _RUNNING_VARIANCE_CLIP_SCALE_EARLY = 8.0
const _RUNNING_VARIANCE_CLIP_SCALE_LATE = 5.0
const _RUNNING_VARIANCE_REJECTION_WEIGHT_EARLY = 1.0
const _RUNNING_VARIANCE_FLOOR = 1e-3

struct WarmupSchedule
    initial_buffer::Int
    slow_window_ends::Vector{Int}
    terminal_buffer::Int
end

mutable struct NUTSState{P<:AbstractVector{Float64},M<:AbstractVector{Float64},G<:AbstractVector{Float64}}
    position::P
    momentum::M
    logjoint::Float64
    gradient::G
end

mutable struct NUTSSubtreeMetadataState
    log_weight::Float64
    accept_stat_sum::Float64
    accept_stat_count::Int
    integration_steps::Int
    proposed_energy::Float64
    delta_energy::Float64
    proposal_energy::Float64
    proposal_energy_error::Float64
    accept_prob::Float64
    candidate_log_weight::Float64
    combined_log_weight::Float64
    turning::Bool
    divergent::Bool
end

mutable struct NUTSSubtreeWorkspace{
    C<:NUTSState,
    N<:NUTSState,
    L<:NUTSState,
    R<:NUTSState,
    P<:NUTSState,
    S<:NUTSSubtreeMetadataState,
}
    current::C
    next::N
    left::L
    right::R
    proposal::P
    summary::S
    checkpoint_positions::Matrix{Float64}
    checkpoint_momenta::Matrix{Float64}
end

mutable struct NUTSContinuationState{L<:NUTSState,R<:NUTSState,P<:NUTSState}
    left::L
    right::R
    proposal::P
    proposal_energy::Float64
    proposal_energy_error::Float64
    log_weight::Float64
    accept_stat_sum::Float64
    accept_stat_count::Int
    integration_steps::Int
    tree_depth::Int
    turning::Bool
    divergent::Bool
end

@enum BatchedNUTSSchedulerPhase::UInt8 begin
    NUTSSchedulerIdle = 0
    NUTSSchedulerExpand = 1
    NUTSSchedulerMerge = 2
    NUTSSchedulerDone = 3
end

mutable struct BatchedNUTSSchedulerState
    continuation_active::BitVector
    subtree_started::BitVector
    active_depth::Int
    active_depth_count::Int
    phase::BatchedNUTSSchedulerPhase
    remaining_steps::Int
end

mutable struct BatchedNUTSControlState
    accepted_step::BitVector
    divergent_step::BitVector
    continuation_turning::BitVector
    step_valid::BitVector
    step_direction::Vector{Int}
    tree_depths::Vector{Int}
    integration_steps::Vector{Int}
    scheduler::BatchedNUTSSchedulerState
end

abstract type AbstractBatchedNUTSControlIR end

struct BatchedNUTSIdleIR <: AbstractBatchedNUTSControlIR end

struct BatchedNUTSExpandIR <: AbstractBatchedNUTSControlIR
    active_depth::Int
    active_depth_count::Int
    remaining_steps::Int
    active_chains::BitVector
    step_direction::Vector{Int}
end

struct BatchedNUTSMergeIR <: AbstractBatchedNUTSControlIR
    active_depth::Int
    active_depth_count::Int
    started_chains::BitVector
    merge_active::BitVector
end

struct BatchedNUTSDoneIR <: AbstractBatchedNUTSControlIR end

abstract type AbstractBatchedNUTSControlBlock end

struct BatchedNUTSIdleControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSIdleIR
end

struct BatchedNUTSExpandControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSExpandIR
    active_chains::BitVector
    step_direction::Vector{Int}
end

struct BatchedNUTSMergeControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSMergeIR
    started_chains::BitVector
    merge_active::BitVector
end

struct BatchedNUTSDoneControlBlock <: AbstractBatchedNUTSControlBlock
    ir::BatchedNUTSDoneIR
end

abstract type AbstractBatchedNUTSStepDescriptor end

struct BatchedNUTSIdleStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSIdleControlBlock
end

struct BatchedNUTSExpandStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSExpandControlBlock
    copy_left::BitVector
    copy_right::BitVector
    select_proposal::BitVector
    turning::BitVector
end

struct BatchedNUTSMergeStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSMergeControlBlock
    select_proposal::BitVector
    merged_turning::BitVector
end

struct BatchedNUTSDoneStepDescriptor <: AbstractBatchedNUTSStepDescriptor
    block::BatchedNUTSDoneControlBlock
end

abstract type AbstractBatchedNUTSStepState end

struct BatchedNUTSIdleStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSIdleStepDescriptor
end

struct BatchedNUTSExpandStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSExpandStepDescriptor
    log_weight::Vector{Float64}
    proposed_energy::Vector{Float64}
    delta_energy::Vector{Float64}
    proposal_energy::Vector{Float64}
    proposal_energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    candidate_log_weight::Vector{Float64}
    combined_log_weight::Vector{Float64}
end

struct BatchedNUTSMergeStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSMergeStepDescriptor
    proposal_energy::Vector{Float64}
    proposal_energy_error::Vector{Float64}
    candidate_log_weight::Vector{Float64}
    combined_log_weight::Vector{Float64}
end

struct BatchedNUTSDoneStepState <: AbstractBatchedNUTSStepState
    descriptor::BatchedNUTSDoneStepDescriptor
end

abstract type AbstractBatchedNUTSKernelFrame end

struct BatchedNUTSIdleKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSIdleStepState
end

struct BatchedNUTSExpandKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSExpandStepState
    current_position::Matrix{Float64}
    current_momentum::Matrix{Float64}
    current_gradient::Matrix{Float64}
    current_logjoint::Vector{Float64}
    next_position::Matrix{Float64}
    next_momentum::Matrix{Float64}
    next_gradient::Matrix{Float64}
    proposed_logjoint::Vector{Float64}
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
    current_energy::Vector{Float64}
end

struct BatchedNUTSMergeKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSMergeStepState
    left_position::Matrix{Float64}
    left_momentum::Matrix{Float64}
    left_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    right_position::Matrix{Float64}
    right_momentum::Matrix{Float64}
    right_gradient::Matrix{Float64}
    right_logjoint::Vector{Float64}
    tree_proposal_position::Matrix{Float64}
    tree_proposal_momentum::Matrix{Float64}
    tree_proposal_gradient::Matrix{Float64}
    tree_proposal_logjoint::Vector{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    proposed_logjoint::Vector{Float64}
    continuation_proposal_logjoint::Vector{Float64}
    continuation_log_weight::Vector{Float64}
    current_energy::Vector{Float64}
end

struct BatchedNUTSDoneKernelFrame <: AbstractBatchedNUTSKernelFrame
    state::BatchedNUTSDoneStepState
end

abstract type AbstractBatchedNUTSKernelAccess end

abstract type AbstractBatchedNUTSKernelProgram end

@enum BatchedNUTSKernelOp::UInt8 begin
    NUTSKernelReloadControl = 0
    NUTSKernelLeapfrog = 1
    NUTSKernelHamiltonian = 2
    NUTSKernelAdvance = 3
    NUTSKernelActivateMerge = 4
    NUTSKernelMerge = 5
    NUTSKernelTransitionPhase = 6
end

mutable struct BatchedNUTSKernelExecutionState
    any_active::Bool
end

abstract type AbstractBatchedNUTSKernelStep end

struct BatchedNUTSReloadControlStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSLeapfrogStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSHamiltonianStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSAdvanceStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSActivateMergeStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSMergeStep <: AbstractBatchedNUTSKernelStep end
struct BatchedNUTSTransitionPhaseStep <: AbstractBatchedNUTSKernelStep end

struct BatchedNUTSIdleKernelProgram{A<:AbstractBatchedNUTSKernelAccess} <: AbstractBatchedNUTSKernelProgram
    access::A
    ops::NTuple{1,BatchedNUTSKernelOp}
end

struct BatchedNUTSExpandKernelProgram{A<:AbstractBatchedNUTSKernelAccess} <: AbstractBatchedNUTSKernelProgram
    access::A
    ops::NTuple{5,BatchedNUTSKernelOp}
end

struct BatchedNUTSMergeKernelProgram{A<:AbstractBatchedNUTSKernelAccess} <: AbstractBatchedNUTSKernelProgram
    access::A
    ops::NTuple{4,BatchedNUTSKernelOp}
end

struct BatchedNUTSDoneKernelProgram{A<:AbstractBatchedNUTSKernelAccess} <: AbstractBatchedNUTSKernelProgram
    access::A
    ops::NTuple{1,BatchedNUTSKernelOp}
end

const _BATCHED_NUTS_IDLE_KERNEL_OPS = (NUTSKernelReloadControl,)
const _BATCHED_NUTS_EXPAND_KERNEL_OPS = (
    NUTSKernelReloadControl,
    NUTSKernelLeapfrog,
    NUTSKernelHamiltonian,
    NUTSKernelAdvance,
    NUTSKernelTransitionPhase,
)
const _BATCHED_NUTS_MERGE_KERNEL_OPS = (
    NUTSKernelReloadControl,
    NUTSKernelActivateMerge,
    NUTSKernelMerge,
    NUTSKernelTransitionPhase,
)
const _BATCHED_NUTS_DONE_KERNEL_OPS = (NUTSKernelReloadControl,)
const _BATCHED_NUTS_IDLE_KERNEL_STEPS = (BatchedNUTSReloadControlStep(),)
const _BATCHED_NUTS_EXPAND_KERNEL_STEPS = (
    BatchedNUTSReloadControlStep(),
    BatchedNUTSLeapfrogStep(),
    BatchedNUTSHamiltonianStep(),
    BatchedNUTSAdvanceStep(),
    BatchedNUTSTransitionPhaseStep(),
)
const _BATCHED_NUTS_MERGE_KERNEL_STEPS = (
    BatchedNUTSReloadControlStep(),
    BatchedNUTSActivateMergeStep(),
    BatchedNUTSMergeStep(),
    BatchedNUTSTransitionPhaseStep(),
)
const _BATCHED_NUTS_DONE_KERNEL_STEPS = (BatchedNUTSReloadControlStep(),)

mutable struct BatchedHMCWorkspace
    logjoint_workspace::BatchedLogjointWorkspace
    gradient_cache::BatchedLogjointGradientCache
    current_gradient::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    momentum::Matrix{Float64}
    proposal_position::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    valid::BitVector
    current_hamiltonian::Vector{Float64}
    proposed_hamiltonian::Vector{Float64}
    proposed_logjoint::Vector{Float64}
    log_accept_ratio::Vector{Float64}
    energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    accepted_step::BitVector
    divergent_step::BitVector
    mass_adaptation_weights::Vector{Float64}
    constrained_position::Matrix{Float64}
    sqrt_inverse_mass_matrix::Vector{Float64}
end

mutable struct BatchedNUTSWorkspace
    gradient_cache::BatchedLogjointGradientCache
    current_gradient::Matrix{Float64}
    tree_current_position::Matrix{Float64}
    tree_next_position::Matrix{Float64}
    tree_left_position::Matrix{Float64}
    tree_right_position::Matrix{Float64}
    tree_proposal_position::Matrix{Float64}
    left_position::Matrix{Float64}
    proposal_position::Matrix{Float64}
    right_position::Matrix{Float64}
    tree_current_momentum::Matrix{Float64}
    tree_next_momentum::Matrix{Float64}
    tree_left_momentum::Matrix{Float64}
    tree_right_momentum::Matrix{Float64}
    tree_proposal_momentum::Matrix{Float64}
    left_momentum::Matrix{Float64}
    current_momentum::Matrix{Float64}
    proposal_momentum::Matrix{Float64}
    right_momentum::Matrix{Float64}
    tree_current_gradient::Matrix{Float64}
    tree_next_gradient::Matrix{Float64}
    tree_left_gradient::Matrix{Float64}
    tree_right_gradient::Matrix{Float64}
    tree_proposal_gradient::Matrix{Float64}
    tree_current_logjoint::Vector{Float64}
    tree_left_logjoint::Vector{Float64}
    tree_right_logjoint::Vector{Float64}
    tree_proposal_logjoint::Vector{Float64}
    left_gradient::Matrix{Float64}
    proposal_gradient::Matrix{Float64}
    right_gradient::Matrix{Float64}
    left_logjoint::Vector{Float64}
    continuation_proposal_logjoint::Vector{Float64}
    right_logjoint::Vector{Float64}
    proposed_logjoint::Vector{Float64}
    current_energy::Vector{Float64}
    proposed_energy::Vector{Float64}
    continuation_log_weight::Vector{Float64}
    continuation_accept_stat_sum::Vector{Float64}
    continuation_accept_stat_count::Vector{Int}
    continuation_proposed_energy::Vector{Float64}
    continuation_delta_energy::Vector{Float64}
    continuation_accept_prob::Vector{Float64}
    continuation_candidate_log_weight::Vector{Float64}
    continuation_combined_log_weight::Vector{Float64}
    continuation_select_proposal::BitVector
    subtree_log_weight::Vector{Float64}
    subtree_accept_stat_sum::Vector{Float64}
    subtree_accept_stat_count::Vector{Int}
    subtree_integration_steps::Vector{Int}
    subtree_proposed_energy::Vector{Float64}
    subtree_delta_energy::Vector{Float64}
    subtree_proposal_energy::Vector{Float64}
    subtree_proposal_energy_error::Vector{Float64}
    subtree_accept_prob::Vector{Float64}
    subtree_candidate_log_weight::Vector{Float64}
    subtree_combined_log_weight::Vector{Float64}
    energy_error::Vector{Float64}
    accept_prob::Vector{Float64}
    subtree_turning::BitVector
    subtree_merged_turning::BitVector
    subtree_divergent::BitVector
    subtree_active::BitVector
    subtree_copy_left::BitVector
    subtree_copy_right::BitVector
    subtree_select_proposal::BitVector
    control::BatchedNUTSControlState
    mass_adaptation_weights::Vector{Float64}
    constrained_position::Matrix{Float64}
    # Per-iteration buffers (issue #142): sqrt inverse-mass for momentum
    # refreshes (shared vector / per-chain columns) and the all-chains-active
    # mask, previously reallocated on every draw.
    sqrt_inverse_mass::Vector{Float64}
    sqrt_inverse_mass_columns::Matrix{Float64}
    all_chains_active::BitVector
    column_gradient_caches::Vector{LogjointGradientCache}
    column_tree_workspaces::Vector{NUTSSubtreeWorkspace}
    column_continuation_states::Vector{NUTSContinuationState}
end

function BatchedNUTSSchedulerState(num_chains::Int)
    return BatchedNUTSSchedulerState(
        falses(num_chains),
        falses(num_chains),
        0,
        0,
        NUTSSchedulerIdle,
        0,
    )
end

function BatchedNUTSControlState(num_chains::Int)
    return BatchedNUTSControlState(
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        zeros(Int, num_chains),
        zeros(Int, num_chains),
        zeros(Int, num_chains),
        BatchedNUTSSchedulerState(num_chains),
    )
end

function BatchedHMCWorkspace(
    model::TeaModel,
    position::AbstractMatrix,
    args=(),
    constraints=choicemap(),
    inverse_mass_matrix::AbstractVector=ones(size(position, 1)),
)
    num_params, num_chains = size(position)
    batch_args = _validate_batched_args(model, args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    # constrained-space width follows the conditioning signature (issue #95)
    constrained_num_params = parametervaluecount(_batched_signature_layout(model, batch_constraints))
    length(inverse_mass_matrix) == num_params ||
        throw(DimensionMismatch("expected inverse mass matrix of length $num_params, got $(length(inverse_mass_matrix))"))

    # sampler-owned evaluation: Stan-style reject semantics (issue #157) --
    # invalid distribution parameters mid-trajectory score -Inf (per-chain
    # divergence) instead of killing the whole batched run
    return BatchedHMCWorkspace(
        BatchedLogjointWorkspace(model, batch_constraints; reject_invalid_parameters=true),
        BatchedLogjointGradientCache(model, position, batch_args, batch_constraints; reject_invalid_parameters=true),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        Matrix{Float64}(undef, num_params, num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        falses(num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Matrix{Float64}(undef, constrained_num_params, num_chains),
        sqrt.(Float64.(inverse_mass_matrix)),
    )
end

function BatchedNUTSWorkspace(
    model::TeaModel,
    position::AbstractMatrix,
    args=(),
    constraints=choicemap(),
    max_tree_depth::Int=10;
    adtype::Symbol=:auto,
)
    num_params, num_chains = size(position)
    checkpoint_columns = max(max_tree_depth + 1, 1)
    batch_args = _validate_batched_args(model, args, num_chains)
    batch_constraints = _validate_batched_constraints(constraints, num_chains)
    # constrained-space width follows the conditioning signature (issue #95),
    # matching BatchedHMCWorkspace; the syntactic default layout would mis-size
    # the constrained-sample buffer under a bound-observation signature.
    constrained_num_params = parametervaluecount(_batched_signature_layout(model, batch_constraints))
    # sampler-owned evaluation: Stan-style reject semantics (issue #157). `adtype`
    # selects the host gradient backend (issue #268, A2).
    gradient_cache = BatchedLogjointGradientCache(
        model, position, batch_args, batch_constraints; reject_invalid_parameters=true, adtype=adtype,
    )
    tree_current_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_next_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_left_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_right_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_proposal_position = Matrix{Float64}(undef, num_params, num_chains)
    tree_current_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_next_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_left_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_right_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_proposal_momentum = Matrix{Float64}(undef, num_params, num_chains)
    tree_current_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_next_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_left_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_right_gradient = Matrix{Float64}(undef, num_params, num_chains)
    tree_proposal_gradient = Matrix{Float64}(undef, num_params, num_chains)
    column_gradient_caches = _batched_nuts_column_gradient_caches(
        model,
        position,
        batch_args,
        batch_constraints,
        tree_next_gradient,
    )
    column_tree_workspaces = [
        NUTSSubtreeWorkspace(
            NUTSState(
                view(tree_current_position, :, chain_index),
                view(tree_current_momentum, :, chain_index),
                0.0,
                view(tree_current_gradient, :, chain_index),
            ),
            NUTSState(
                view(tree_next_position, :, chain_index),
                view(tree_next_momentum, :, chain_index),
                0.0,
                view(tree_next_gradient, :, chain_index),
            ),
            NUTSState(
                view(tree_left_position, :, chain_index),
                view(tree_left_momentum, :, chain_index),
                0.0,
                view(tree_left_gradient, :, chain_index),
            ),
            NUTSState(
                view(tree_right_position, :, chain_index),
                view(tree_right_momentum, :, chain_index),
                0.0,
                view(tree_right_gradient, :, chain_index),
            ),
            NUTSState(
                view(tree_proposal_position, :, chain_index),
                view(tree_proposal_momentum, :, chain_index),
                0.0,
                view(tree_proposal_gradient, :, chain_index),
            ),
            NUTSSubtreeMetadataState(-Inf, 0.0, 0, 0, Inf, Inf, Inf, Inf, 0.0, -Inf, -Inf, false, false),
            zeros(num_params, checkpoint_columns),
            zeros(num_params, checkpoint_columns),
        ) for chain_index = 1:num_chains
    ]
    left_position = Matrix{Float64}(undef, num_params, num_chains)
    right_position = Matrix{Float64}(undef, num_params, num_chains)
    left_momentum = Matrix{Float64}(undef, num_params, num_chains)
    right_momentum = Matrix{Float64}(undef, num_params, num_chains)
    left_gradient = Matrix{Float64}(undef, num_params, num_chains)
    right_gradient = Matrix{Float64}(undef, num_params, num_chains)
    proposal_position = Matrix{Float64}(undef, num_params, num_chains)
    proposal_momentum = Matrix{Float64}(undef, num_params, num_chains)
    proposal_gradient = Matrix{Float64}(undef, num_params, num_chains)
    proposed_logjoint = Vector{Float64}(undef, num_chains)
    control = BatchedNUTSControlState(num_chains)
    column_continuation_states = [
        NUTSContinuationState(
            NUTSState(
                view(left_position, :, chain_index),
                view(left_momentum, :, chain_index),
                0.0,
                view(left_gradient, :, chain_index),
            ),
            NUTSState(
                view(right_position, :, chain_index),
                view(right_momentum, :, chain_index),
                0.0,
                view(right_gradient, :, chain_index),
            ),
            _batched_nuts_state(proposal_position, proposal_momentum, proposed_logjoint, proposal_gradient, chain_index),
            Inf,
            Inf,
            -Inf,
            0.0,
            0,
            0,
            0,
            false,
            false,
        ) for chain_index = 1:num_chains
    ]
    return BatchedNUTSWorkspace(
        gradient_cache,
        Matrix{Float64}(undef, num_params, num_chains),
        tree_current_position,
        tree_next_position,
        tree_left_position,
        tree_right_position,
        tree_proposal_position,
        left_position,
        proposal_position,
        right_position,
        tree_current_momentum,
        tree_next_momentum,
        tree_left_momentum,
        tree_right_momentum,
        tree_proposal_momentum,
        left_momentum,
        Matrix{Float64}(undef, num_params, num_chains),
        proposal_momentum,
        right_momentum,
        tree_current_gradient,
        tree_next_gradient,
        tree_left_gradient,
        tree_right_gradient,
        tree_proposal_gradient,
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        left_gradient,
        proposal_gradient,
        right_gradient,
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        proposed_logjoint,
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        zeros(Int, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        falses(num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        zeros(Int, num_chains),
        zeros(Int, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        Vector{Float64}(undef, num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        falses(num_chains),
        control,
        Vector{Float64}(undef, num_chains),
        Matrix{Float64}(undef, constrained_num_params, num_chains),
        Vector{Float64}(undef, num_params),
        Matrix{Float64}(undef, num_params, num_chains),
        trues(num_chains),
        column_gradient_caches,
        column_tree_workspaces,
        column_continuation_states,
    )
end

function _batched_nuts_column_gradient_caches(
    model::TeaModel,
    position::AbstractMatrix,
    batch_args::Tuple,
    batch_constraints::ChoiceMap,
    gradient_buffer::AbstractMatrix,
)
    num_chains = size(position, 2)
    num_chains == 0 && return LogjointGradientCache[]
    first_cache = _logjoint_gradient_cache(
        model,
        collect(view(position, :, 1)),
        batch_args,
        batch_constraints,
        view(gradient_buffer, :, 1);
        reject_invalid_parameters=true,
    )
    caches = Vector{typeof(first_cache)}(undef, num_chains)
    caches[1] = first_cache
    for chain_index = 2:num_chains
        caches[chain_index] = LogjointGradientCache(
            first_cache.objective,
            first_cache.config,
            view(gradient_buffer, :, chain_index),
        )
    end
    return caches
end

function _batched_nuts_column_gradient_caches(
    model::TeaModel,
    position::AbstractMatrix,
    batch_args,
    batch_constraints,
    gradient_buffer::AbstractMatrix,
)
    num_chains = size(position, 2)
    caches = Vector{LogjointGradientCache}(undef, num_chains)
    for chain_index = 1:num_chains
        caches[chain_index] = _logjoint_gradient_cache(
            model,
            collect(view(position, :, chain_index)),
            _batched_args(batch_args, chain_index),
            _batched_constraints(batch_constraints, chain_index),
            view(gradient_buffer, :, chain_index);
            reject_invalid_parameters=true,
        )
    end
    return caches
end

Base.length(chain::HMCChain) = size(chain.unconstrained_samples, 2)
Base.length(chains::HMCChains) = length(chains.chains)
Base.length(summary::HMCSummary) = length(summary.parameters)
Base.getindex(chains::HMCChains, index::Int) = chains.chains[index]
Base.getindex(summary::HMCSummary, index::Int) = summary.parameters[index]
Base.firstindex(chains::HMCChains) = firstindex(chains.chains)
Base.firstindex(summary::HMCSummary) = firstindex(summary.parameters)
Base.lastindex(chains::HMCChains) = lastindex(chains.chains)
Base.lastindex(summary::HMCSummary) = lastindex(summary.parameters)
Base.iterate(chains::HMCChains, state...) = iterate(chains.chains, state...)
Base.iterate(summary::HMCSummary, state...) = iterate(summary.parameters, state...)

_sampler_label(chain::HMCChain) = uppercase(String(chain.sampler))
