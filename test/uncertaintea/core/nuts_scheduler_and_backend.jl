@testset "nuts_scheduler_and_backend" begin
    unsupported_backend_params = reshape(Float64[-0.3, 0.2], 1, 2)
    unsupported_backend_constraints = [
        choicemap((:y, -0.1f0)),
        choicemap((:y, 0.4f0)),
    ]
    unsupported_backend_logjoint = batched_logjoint(
        unsupported_backend_model,
        unsupported_backend_params,
        (),
        unsupported_backend_constraints,
    )
    gaussian_backend_report = backend_report(gaussian_mean)
    gaussian_backend_plan = backend_execution_plan(gaussian_mean)
    iid_backend_report = backend_report(iid_model)
    iid_backend_plan = backend_execution_plan(iid_model)
    shifted_backend_report = backend_report(shifted_iid_model)
    shifted_backend_plan = backend_execution_plan(shifted_iid_model)
    offset_backend_report = backend_report(offset_iid_model)
    offset_backend_plan = backend_execution_plan(offset_iid_model)
    indexed_scale_plan = executionplan(indexed_scale_model)
    indexed_scale_backend_report = backend_report(indexed_scale_model)
    indexed_scale_backend_plan = backend_execution_plan(indexed_scale_model)
    coin_backend_report = backend_report(observed_coin)
    coin_backend_plan = backend_execution_plan(observed_coin)
    # PR-4: signature-keyed workspace, built with the conditioning it will score
    # (`:y` observed), so the coin choice is an observation with no parameter slot.
    coin_workspace = UncertainTea.BatchedLogjointWorkspace(observed_coin, choicemap((:y, true)))
    coin_batch_logjoint = batched_logjoint(
        observed_coin,
        zeros(0, 3),
        (),
        [
            choicemap((:y, true)),
            choicemap((:y, false)),
            choicemap((:y, true)),
        ],
    )
    deterministic_backend_report = backend_report(deterministic_scale)
    deterministic_backend_plan = backend_execution_plan(deterministic_scale)
    unsupported_backend_report = backend_report(unsupported_backend_model)
    short_warmup_schedule = UncertainTea._warmup_schedule(12)
    long_warmup_schedule = UncertainTea._warmup_schedule(150)

    gaussian_cohort_scheduler_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_cohort_scheduler_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(103),
    )
    cohort_rng = MersenneTwister(104)
    @test gaussian_cohort_scheduler_workspace.control isa UncertainTea.BatchedNUTSControlState
    @test gaussian_cohort_scheduler_workspace.control.scheduler isa
          UncertainTea.BatchedNUTSSchedulerState
    idle_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test idle_ir isa UncertainTea.BatchedNUTSIdleIR
    idle_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test idle_block isa UncertainTea.BatchedNUTSIdleControlBlock
    idle_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test idle_descriptor isa UncertainTea.BatchedNUTSIdleStepDescriptor
    idle_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test idle_state isa UncertainTea.BatchedNUTSIdleStepState
    idle_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test idle_frame isa UncertainTea.BatchedNUTSIdleKernelFrame
    idle_program = UncertainTea._batched_nuts_kernel_program(gaussian_cohort_scheduler_workspace)
    @test idle_program isa UncertainTea.BatchedNUTSIdleKernelProgram
    @test UncertainTea._batched_nuts_kernel_ops(idle_program) ==
          (UncertainTea.NUTSKernelReloadControl,)
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(idle_program)) ==
          (UncertainTea.BatchedNUTSReloadControlStep,)
    @test !UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerIdle
    @test UncertainTea._begin_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        4,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerExpand
    @test gaussian_cohort_scheduler_workspace.control.scheduler.active_depth == 1
    @test gaussian_cohort_scheduler_workspace.control.scheduler.active_depth_count == 3
    @test gaussian_cohort_scheduler_workspace.control.scheduler.remaining_steps == 2
    @test gaussian_cohort_scheduler_workspace.control.scheduler.continuation_active ==
          BitVector([true, true, true])
    expand_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test expand_ir isa UncertainTea.BatchedNUTSExpandIR
    @test expand_ir.active_depth == 1
    @test expand_ir.active_depth_count == 3
    @test expand_ir.remaining_steps == 2
    @test expand_ir.active_chains == BitVector([true, true, true])
    expand_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test expand_block isa UncertainTea.BatchedNUTSExpandControlBlock
    @test expand_block.active_chains == expand_ir.active_chains
    @test expand_block.step_direction == expand_ir.step_direction
    expand_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test expand_descriptor isa UncertainTea.BatchedNUTSExpandStepDescriptor
    @test expand_descriptor.copy_left === gaussian_cohort_scheduler_workspace.subtree_copy_left
    @test expand_descriptor.copy_right === gaussian_cohort_scheduler_workspace.subtree_copy_right
    @test expand_descriptor.select_proposal ===
          gaussian_cohort_scheduler_workspace.subtree_select_proposal
    @test expand_descriptor.turning === gaussian_cohort_scheduler_workspace.subtree_turning
    expand_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test expand_state isa UncertainTea.BatchedNUTSExpandStepState
    @test expand_state.descriptor.copy_left === gaussian_cohort_scheduler_workspace.subtree_copy_left
    @test expand_state.log_weight === gaussian_cohort_scheduler_workspace.subtree_log_weight
    @test expand_state.proposed_energy ===
          gaussian_cohort_scheduler_workspace.subtree_proposed_energy
    @test expand_state.delta_energy === gaussian_cohort_scheduler_workspace.subtree_delta_energy
    @test expand_state.proposal_energy ===
          gaussian_cohort_scheduler_workspace.subtree_proposal_energy
    @test expand_state.proposal_energy_error ===
          gaussian_cohort_scheduler_workspace.subtree_proposal_energy_error
    @test expand_state.accept_prob === gaussian_cohort_scheduler_workspace.subtree_accept_prob
    @test expand_state.candidate_log_weight ===
          gaussian_cohort_scheduler_workspace.subtree_candidate_log_weight
    @test expand_state.combined_log_weight ===
          gaussian_cohort_scheduler_workspace.subtree_combined_log_weight
    expand_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test expand_frame isa UncertainTea.BatchedNUTSExpandKernelFrame
    @test expand_frame.state.log_weight === gaussian_cohort_scheduler_workspace.subtree_log_weight
    @test expand_frame.current_position === gaussian_cohort_scheduler_workspace.tree_current_position
    @test expand_frame.next_position === gaussian_cohort_scheduler_workspace.tree_next_position
    @test expand_frame.proposed_logjoint === gaussian_cohort_scheduler_workspace.proposed_logjoint
    @test expand_frame.left_position === gaussian_cohort_scheduler_workspace.tree_left_position
    @test expand_frame.right_position === gaussian_cohort_scheduler_workspace.tree_right_position
    @test expand_frame.proposal_position === gaussian_cohort_scheduler_workspace.tree_proposal_position
    @test expand_frame.current_energy === gaussian_cohort_scheduler_workspace.current_energy
    @test UncertainTea._batched_nuts_kernel_frame(
        gaussian_cohort_scheduler_workspace,
        expand_state,
    ).state === expand_state
    expand_program = UncertainTea._batched_nuts_kernel_program(gaussian_cohort_scheduler_workspace)
    @test expand_program isa UncertainTea.BatchedNUTSExpandKernelProgram
    @test UncertainTea._batched_nuts_kernel_ops(expand_program) ==
          (
        UncertainTea.NUTSKernelReloadControl,
        UncertainTea.NUTSKernelLeapfrog,
        UncertainTea.NUTSKernelHamiltonian,
        UncertainTea.NUTSKernelAdvance,
        UncertainTea.NUTSKernelTransitionPhase,
    )
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(expand_program)) ==
          (
        UncertainTea.BatchedNUTSReloadControlStep,
        UncertainTea.BatchedNUTSLeapfrogStep,
        UncertainTea.BatchedNUTSHamiltonianStep,
        UncertainTea.BatchedNUTSAdvanceStep,
        UncertainTea.BatchedNUTSTransitionPhaseStep,
    )
    while gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerExpand
        @test UncertainTea._step_batched_nuts_subtree_scheduler!(
            gaussian_cohort_scheduler_workspace,
            gaussian_mean,
            [1.0],
            (),
            choicemap((:y, 0.4)),
            0.01,
            1000.0,
            cohort_rng,
        )
    end
    merge_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test merge_ir isa UncertainTea.BatchedNUTSMergeIR
    @test merge_ir.active_depth == 1
    @test merge_ir.active_depth_count == 3
    @test merge_ir.started_chains == BitVector([true, true, true])
    @test merge_ir.merge_active == BitVector([true, true, true])
    merge_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test merge_block isa UncertainTea.BatchedNUTSMergeControlBlock
    @test merge_block.started_chains == merge_ir.started_chains
    @test merge_block.merge_active == merge_ir.merge_active
    merge_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test merge_descriptor isa UncertainTea.BatchedNUTSMergeStepDescriptor
    @test merge_descriptor.select_proposal ===
          gaussian_cohort_scheduler_workspace.continuation_select_proposal
    @test merge_descriptor.merged_turning ===
          gaussian_cohort_scheduler_workspace.subtree_merged_turning
    merge_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test merge_state isa UncertainTea.BatchedNUTSMergeStepState
    @test merge_state.descriptor.select_proposal ===
          gaussian_cohort_scheduler_workspace.continuation_select_proposal
    @test merge_state.proposal_energy ===
          gaussian_cohort_scheduler_workspace.subtree_proposal_energy
    @test merge_state.proposal_energy_error ===
          gaussian_cohort_scheduler_workspace.subtree_proposal_energy_error
    @test merge_state.candidate_log_weight ===
          gaussian_cohort_scheduler_workspace.continuation_candidate_log_weight
    @test merge_state.combined_log_weight ===
          gaussian_cohort_scheduler_workspace.continuation_combined_log_weight
    merge_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test merge_frame isa UncertainTea.BatchedNUTSMergeKernelFrame
    @test merge_frame.state.candidate_log_weight ===
          gaussian_cohort_scheduler_workspace.continuation_candidate_log_weight
    @test merge_frame.left_position === gaussian_cohort_scheduler_workspace.left_position
    @test merge_frame.right_position === gaussian_cohort_scheduler_workspace.right_position
    @test merge_frame.tree_proposal_position ===
          gaussian_cohort_scheduler_workspace.tree_proposal_position
    @test merge_frame.proposal_position === gaussian_cohort_scheduler_workspace.proposal_position
    @test merge_frame.proposed_logjoint === gaussian_cohort_scheduler_workspace.proposed_logjoint
    @test merge_frame.continuation_proposal_logjoint ===
          gaussian_cohort_scheduler_workspace.continuation_proposal_logjoint
    @test merge_frame.continuation_log_weight ===
          gaussian_cohort_scheduler_workspace.continuation_log_weight
    @test UncertainTea._batched_nuts_kernel_frame(
        gaussian_cohort_scheduler_workspace,
        merge_state,
    ).state === merge_state
    merge_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_cohort_scheduler_workspace,
        merge_frame,
    )
    @test merge_access isa UncertainTea.BatchedNUTSMergeKernelAccess
    @test merge_access.block === merge_frame.state.descriptor.block
    @test merge_access.select_proposal ===
          gaussian_cohort_scheduler_workspace.continuation_select_proposal
    @test merge_access.left_position === gaussian_cohort_scheduler_workspace.left_position
    @test merge_access.right_position === gaussian_cohort_scheduler_workspace.right_position
    @test merge_access.proposal_position === gaussian_cohort_scheduler_workspace.proposal_position
    @test merge_access.continuation_log_weight ===
          gaussian_cohort_scheduler_workspace.continuation_log_weight
    merge_program = UncertainTea._batched_nuts_kernel_program(gaussian_cohort_scheduler_workspace)
    @test merge_program isa UncertainTea.BatchedNUTSMergeKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(merge_program) isa
          UncertainTea.BatchedNUTSMergeKernelAccess
    @test UncertainTea._batched_nuts_kernel_access(merge_program).left_position ===
          merge_access.left_position
    @test UncertainTea._batched_nuts_kernel_ops(merge_program) ==
          (
        UncertainTea.NUTSKernelReloadControl,
        UncertainTea.NUTSKernelActivateMerge,
        UncertainTea.NUTSKernelMerge,
        UncertainTea.NUTSKernelTransitionPhase,
    )
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(merge_program)) ==
          (
        UncertainTea.BatchedNUTSReloadControlStep,
        UncertainTea.BatchedNUTSActivateMergeStep,
        UncertainTea.BatchedNUTSMergeStep,
        UncertainTea.BatchedNUTSTransitionPhaseStep,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerMerge
    @test UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerDone
    @test gaussian_cohort_scheduler_workspace.control.scheduler.remaining_steps == 0
    @test gaussian_cohort_scheduler_workspace.control.scheduler.subtree_started ==
          BitVector([true, true, true])
    done_ir = UncertainTea._batched_nuts_control_ir(gaussian_cohort_scheduler_workspace)
    @test done_ir isa UncertainTea.BatchedNUTSDoneIR
    done_block = UncertainTea._batched_nuts_control_block(gaussian_cohort_scheduler_workspace)
    @test done_block isa UncertainTea.BatchedNUTSDoneControlBlock
    done_descriptor = UncertainTea._batched_nuts_step_descriptor(gaussian_cohort_scheduler_workspace)
    @test done_descriptor isa UncertainTea.BatchedNUTSDoneStepDescriptor
    done_state = UncertainTea._batched_nuts_step_state(gaussian_cohort_scheduler_workspace)
    @test done_state isa UncertainTea.BatchedNUTSDoneStepState
    done_frame = UncertainTea._batched_nuts_kernel_frame(gaussian_cohort_scheduler_workspace)
    @test done_frame isa UncertainTea.BatchedNUTSDoneKernelFrame
    done_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_cohort_scheduler_workspace,
        done_frame,
    )
    @test done_access isa UncertainTea.BatchedNUTSDoneKernelAccess
    @test done_access.block === done_frame.state.descriptor.block
    done_program = UncertainTea._batched_nuts_kernel_program(
        gaussian_cohort_scheduler_workspace,
        done_access,
    )
    @test done_program isa UncertainTea.BatchedNUTSDoneKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(done_program).block === done_access.block
    @test UncertainTea._batched_nuts_kernel_ops(done_program) ==
          (UncertainTea.NUTSKernelReloadControl,)
    @test typeof.(UncertainTea._batched_nuts_kernel_steps(done_program)) ==
          (UncertainTea.BatchedNUTSReloadControlStep,)
    @test !UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_cohort_scheduler_workspace,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        cohort_rng,
    )
    @test gaussian_cohort_scheduler_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerDone
    @test gaussian_cohort_scheduler_workspace.control.tree_depths == [2, 2, 2]
    @test gaussian_cohort_scheduler_workspace.subtree_active ==
          BitVector([true, true, true])
    @test all(isfinite, gaussian_cohort_scheduler_workspace.continuation_log_weight)
    @test all(isfinite, gaussian_cohort_scheduler_workspace.proposed_logjoint)
    gaussian_expand_ir_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_expand_ir_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(105),
    )
    expand_ir_rng = MersenneTwister(106)
    @test UncertainTea._begin_batched_nuts_subtree_scheduler!(
        gaussian_expand_ir_workspace,
        4,
        expand_ir_rng,
    )
    expand_direct_ir = UncertainTea._batched_nuts_control_ir(gaussian_expand_ir_workspace)
    @test expand_direct_ir isa UncertainTea.BatchedNUTSExpandIR
    expand_direct_block = UncertainTea._batched_nuts_control_block(expand_direct_ir)
    @test expand_direct_block isa UncertainTea.BatchedNUTSExpandControlBlock
    expand_direct_descriptor = UncertainTea._batched_nuts_step_descriptor(
        gaussian_expand_ir_workspace,
        expand_direct_block,
    )
    @test expand_direct_descriptor isa UncertainTea.BatchedNUTSExpandStepDescriptor
    expand_direct_state = UncertainTea._batched_nuts_step_state(
        gaussian_expand_ir_workspace,
        expand_direct_descriptor,
    )
    @test expand_direct_state isa UncertainTea.BatchedNUTSExpandStepState
    expand_direct_frame = UncertainTea._batched_nuts_kernel_frame(
        gaussian_expand_ir_workspace,
        expand_direct_state,
    )
    @test expand_direct_frame isa UncertainTea.BatchedNUTSExpandKernelFrame
    expand_direct_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_expand_ir_workspace,
        expand_direct_frame,
    )
    @test expand_direct_access isa UncertainTea.BatchedNUTSExpandKernelAccess
    @test expand_direct_access.block === expand_direct_frame.state.descriptor.block
    @test expand_direct_access.copy_left === gaussian_expand_ir_workspace.subtree_copy_left
    @test expand_direct_access.copy_right === gaussian_expand_ir_workspace.subtree_copy_right
    @test expand_direct_access.select_proposal ===
          gaussian_expand_ir_workspace.subtree_select_proposal
    @test expand_direct_access.current_position ===
          gaussian_expand_ir_workspace.tree_current_position
    @test expand_direct_access.next_position === gaussian_expand_ir_workspace.tree_next_position
    @test expand_direct_access.proposed_energy ===
          gaussian_expand_ir_workspace.subtree_proposed_energy
    expand_direct_program = UncertainTea._batched_nuts_kernel_program(
        gaussian_expand_ir_workspace,
        expand_direct_access,
    )
    @test expand_direct_program isa UncertainTea.BatchedNUTSExpandKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(expand_direct_program).next_position ===
          expand_direct_access.next_position
    @test UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_expand_ir_workspace,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        expand_ir_rng,
    )
    @test gaussian_expand_ir_workspace.control.step_direction == expand_direct_ir.step_direction
    @test gaussian_expand_ir_workspace.control.scheduler.remaining_steps ==
          expand_direct_ir.remaining_steps - 1
    @test sum(gaussian_expand_ir_workspace.subtree_integration_steps) > 0
    gaussian_merge_ir_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._initialize_batched_nuts_continuations!(
        gaussian_merge_ir_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        MersenneTwister(107),
    )
    merge_ir_rng = MersenneTwister(108)
    @test UncertainTea._begin_batched_nuts_subtree_scheduler!(
        gaussian_merge_ir_workspace,
        4,
        merge_ir_rng,
    )
    while gaussian_merge_ir_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerExpand
        @test UncertainTea._step_batched_nuts_subtree_scheduler!(
            gaussian_merge_ir_workspace,
            gaussian_mean,
            [1.0],
            (),
            choicemap((:y, 0.4)),
            0.01,
            1000.0,
            merge_ir_rng,
        )
    end
    merge_direct_ir = UncertainTea._batched_nuts_control_ir(gaussian_merge_ir_workspace)
    @test merge_direct_ir isa UncertainTea.BatchedNUTSMergeIR
    merge_direct_block = UncertainTea._batched_nuts_control_block(merge_direct_ir)
    @test merge_direct_block isa UncertainTea.BatchedNUTSMergeControlBlock
    merge_direct_descriptor = UncertainTea._batched_nuts_step_descriptor(
        gaussian_merge_ir_workspace,
        merge_direct_block,
    )
    @test merge_direct_descriptor isa UncertainTea.BatchedNUTSMergeStepDescriptor
    merge_direct_state = UncertainTea._batched_nuts_step_state(
        gaussian_merge_ir_workspace,
        merge_direct_descriptor,
    )
    @test merge_direct_state isa UncertainTea.BatchedNUTSMergeStepState
    merge_direct_frame = UncertainTea._batched_nuts_kernel_frame(
        gaussian_merge_ir_workspace,
        merge_direct_state,
    )
    @test merge_direct_frame isa UncertainTea.BatchedNUTSMergeKernelFrame
    merge_direct_access = UncertainTea._batched_nuts_kernel_access(
        gaussian_merge_ir_workspace,
        merge_direct_frame,
    )
    @test merge_direct_access isa UncertainTea.BatchedNUTSMergeKernelAccess
    @test merge_direct_access.block === merge_direct_frame.state.descriptor.block
    @test merge_direct_access.select_proposal ===
          gaussian_merge_ir_workspace.continuation_select_proposal
    @test merge_direct_access.left_position === gaussian_merge_ir_workspace.left_position
    @test merge_direct_access.tree_proposal_position ===
          gaussian_merge_ir_workspace.tree_proposal_position
    merge_direct_program = UncertainTea._batched_nuts_kernel_program(
        gaussian_merge_ir_workspace,
        merge_direct_access,
    )
    @test merge_direct_program isa UncertainTea.BatchedNUTSMergeKernelProgram
    @test UncertainTea._batched_nuts_kernel_access(merge_direct_program).proposal_position ===
          merge_direct_access.proposal_position
    merge_tree_depths = copy(gaussian_merge_ir_workspace.control.tree_depths)
    fill!(gaussian_merge_ir_workspace.control.scheduler.subtree_started, false)
    fill!(gaussian_merge_ir_workspace.subtree_active, false)
    @test UncertainTea._step_batched_nuts_subtree_scheduler!(
        gaussian_merge_ir_workspace,
        gaussian_mean,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.01,
        1000.0,
        merge_ir_rng,
    )
    @test gaussian_merge_ir_workspace.control.scheduler.subtree_started ==
          merge_direct_ir.started_chains
    @test gaussian_merge_ir_workspace.subtree_active == merge_direct_ir.merge_active
    @test gaussian_merge_ir_workspace.control.tree_depths ==
          merge_tree_depths .+ Int.(merge_direct_ir.started_chains)
    @test gaussian_merge_ir_workspace.control.scheduler.phase ==
          UncertainTea.NUTSSchedulerDone
    gaussian_finalized_nuts_workspace = UncertainTea.BatchedNUTSWorkspace(
        gaussian_mean,
        gaussian_batch_params,
        (),
        choicemap((:y, 0.4)),
    )
    UncertainTea._batched_nuts_proposals!(
        gaussian_finalized_nuts_workspace,
        gaussian_mean,
        gaussian_batch_params,
        gaussian_shared_batch_logjoint,
        gaussian_shared_batch_gradient,
        [1.0],
        (),
        choicemap((:y, 0.4)),
        0.05,
        3,
        1000.0,
        MersenneTwister(103),
    )
    @test gaussian_finalized_nuts_workspace.proposed_logjoint ≈
          gaussian_finalized_nuts_workspace.continuation_proposal_logjoint atol=1e-8
    @test gaussian_finalized_nuts_workspace.proposed_energy ≈
          gaussian_finalized_nuts_workspace.continuation_proposed_energy atol=1e-8
    @test gaussian_finalized_nuts_workspace.accept_prob ≈
          UncertainTea._mean_acceptance_stats!(
        similar(gaussian_finalized_nuts_workspace.accept_prob),
        gaussian_finalized_nuts_workspace.continuation_accept_stat_sum,
        gaussian_finalized_nuts_workspace.continuation_accept_stat_count,
    ) atol=1e-8
    @test gaussian_finalized_nuts_workspace.energy_error ≈
          gaussian_finalized_nuts_workspace.continuation_delta_energy atol=1e-8
    @test gaussian_backend_report.supported
    @test gaussian_backend_report.target == :gpu
    @test isempty(gaussian_backend_report.issues)
    @test gaussian_backend_plan.target == :gpu
    @test length(gaussian_backend_plan.steps) == 2
    @test gaussian_backend_plan.steps[1] isa UncertainTea.BackendChoicePlanStep
    @test gaussian_backend_plan.steps[1] isa UncertainTea.BackendNormalChoicePlanStep
    @test gaussian_backend_plan.numeric_slots == BitVector([true])
    @test gaussian_backend_plan.index_slots == BitVector([false])
    @test gaussian_backend_plan.generic_slots == BitVector([false])
    @test iid_backend_report.supported
    @test count(identity, iid_backend_plan.numeric_slots) == 1
    @test count(identity, iid_backend_plan.index_slots) == 2
    @test !any(iid_backend_plan.generic_slots)
    @test iid_backend_plan.index_slots[1]
    @test iid_backend_plan.numeric_slots[2]
    @test iid_backend_plan.index_slots[3]
    @test shifted_backend_report.supported
    @test shifted_backend_plan.numeric_slots == iid_backend_plan.numeric_slots
    @test shifted_backend_plan.index_slots == iid_backend_plan.index_slots
    @test offset_backend_report.supported
    @test indexed_scale_backend_report.supported
    @test indexed_scale_backend_plan.numeric_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:mu]]
    @test indexed_scale_backend_plan.index_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:n]]
    @test indexed_scale_backend_plan.index_slots[indexed_scale_plan.environment_layout.slot_by_symbol[:i]]
    @test !any(indexed_scale_backend_plan.generic_slots)
    @test !isnothing(UncertainTea._backend_loop_observed_choice(iid_backend_plan.steps[2]))
    @test !isnothing(UncertainTea._backend_loop_observed_choice(shifted_backend_plan.steps[2]))
    @test isnothing(UncertainTea._backend_loop_observed_choice(offset_backend_plan.steps[2]))
    @test isnothing(UncertainTea._backend_loop_observed_choice(backend_execution_plan(chain_model).steps[3]))
    @test coin_backend_report.supported
    @test coin_backend_plan.steps[1] isa UncertainTea.BackendBernoulliChoicePlanStep
    @test isempty(coin_backend_plan.numeric_slots)
    @test isempty(coin_backend_plan.index_slots)
    @test isempty(coin_backend_plan.generic_slots)
    @test coin_batch_logjoint ≈ [
        logjoint(observed_coin, Float64[], (), choicemap((:y, true))),
        logjoint(observed_coin, Float64[], (), choicemap((:y, false))),
        logjoint(observed_coin, Float64[], (), choicemap((:y, true))),
    ] atol=2e-8
    coin_workspace_values = UncertainTea._logjoint_with_batched_backend!(
        coin_workspace,
        zeros(0, 3),
        (),
        [
            choicemap((:y, true)),
            choicemap((:y, false)),
            choicemap((:y, true)),
        ],
    )
    coin_workspace_env = coin_workspace.batched_environment[]
    coin_workspace_observed = coin_workspace_env.observed_values
    @test coin_workspace_values ≈ coin_batch_logjoint atol=2e-8
    @test UncertainTea._logjoint_with_batched_backend!(
        coin_workspace,
        zeros(0, 3),
        (),
        choicemap((:y, true)),
    ) ≈ fill(logjoint(observed_coin, Float64[], (), choicemap((:y, true))), 3) atol=2e-8
    @test coin_workspace.batched_environment[] === coin_workspace_env
    @test coin_workspace_env.observed_values === coin_workspace_observed
    @test deterministic_backend_report.supported
    @test length(deterministic_backend_plan.steps) == 4
    @test deterministic_backend_plan.steps[3] isa UncertainTea.BackendDeterministicPlanStep
    @test all(deterministic_backend_plan.numeric_slots)
    @test !any(deterministic_backend_plan.index_slots)
    @test !any(deterministic_backend_plan.generic_slots)
    @test !unsupported_backend_report.supported
    @test any(occursin("cbrt", issue) for issue in unsupported_backend_report.issues)
    @test unsupported_backend_logjoint ≈ [
        logjoint(
            unsupported_backend_model,
            unsupported_backend_params[:, index],
            (),
            unsupported_backend_constraints[index],
        ) for index = 1:2
    ] atol=1e-8
    @test short_warmup_schedule.initial_buffer == 5
    @test short_warmup_schedule.slow_window_ends == [12]
    @test short_warmup_schedule.terminal_buffer == 0
    @test long_warmup_schedule.initial_buffer == 15
    @test long_warmup_schedule.slow_window_ends == [40, 90, 135]
    @test long_warmup_schedule.terminal_buffer == 15
    @test UncertainTea._warmup_window_length(short_warmup_schedule, 1) == 7
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 1) == 25
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 2) == 50
    @test UncertainTea._warmup_window_length(long_warmup_schedule, 3) == 45
    @test UncertainTea._mean_batched_adaptation_probability([0.8, 0.6, 0.4], falses(3)) ≈ 0.6 atol=1e-8
    @test UncertainTea._mean_batched_adaptation_probability([0.8, 0.6, 0.4], BitVector([false, true, false])) ≈
          0.4 atol=1e-8
    early_window_state = UncertainTea._running_variance_state(1, 24)
    late_window_state = UncertainTea._running_variance_state(1, 24)
    late_window_state.count = 24
    @test UncertainTea._mass_adaptation_weight(early_window_state, true, 0.1, false) == 1.0
    @test UncertainTea._mass_adaptation_weight(early_window_state, false, 0.25, false) == 1.0
    @test UncertainTea._mass_adaptation_weight(late_window_state, false, 0.25, false) == 0.25
    @test UncertainTea._mass_adaptation_weight(early_window_state, false, 0.25, true) == 0.0
    mass_adaptation_weights = zeros(3)
    UncertainTea._mass_adaptation_weights!(
        late_window_state,
        mass_adaptation_weights,
        BitVector([true, false, false]),
        [0.2, 0.3, 0.4],
        BitVector([false, false, true]),
    )
    @test mass_adaptation_weights ≈ [1.0, 0.3, 0.0] atol=1e-8
    @test UncertainTea._running_variance_clip_scale(early_window_state) == 8.0
    early_window_state.count = 4
    @test UncertainTea._running_variance_clip_scale(early_window_state) == 8.0
    early_window_state.count = 14
    @test UncertainTea._running_variance_clip_scale(early_window_state) ≈ 6.5 atol=1e-8
    @test UncertainTea._running_variance_window_progress(early_window_state) ≈ 0.5 atol=1e-8
    @test UncertainTea._running_variance_clip_scale(late_window_state) == 5.0
    @test UncertainTea._running_variance_window_progress(late_window_state) == 1.0
    masked_variance_state = UncertainTea._running_variance_state(2)
    UncertainTea._update_running_variance!(
        masked_variance_state,
        [1.0 2.0 5.0; 10.0 20.0 50.0],
        BitVector([true, false, true]),
    )
    @test masked_variance_state.count == 2
    @test masked_variance_state.mean ≈ [3.0, 30.0] atol=1e-8
    @test masked_variance_state.m2 ≈ [8.0, 800.0] atol=1e-8
    @test masked_variance_state.weight_sum == 2.0
    @test masked_variance_state.weight_square_sum == 2.0
    @test UncertainTea._running_variance_effective_count(masked_variance_state) == 2.0
    weighted_variance_state = UncertainTea._running_variance_state(1)
    UncertainTea._update_running_variance!(weighted_variance_state, [0.0], 1.0)
    UncertainTea._update_running_variance!(weighted_variance_state, [10.0], 0.25)
    @test weighted_variance_state.count == 2
    @test weighted_variance_state.weight_sum ≈ 1.25 atol=1e-8
    @test weighted_variance_state.weight_square_sum ≈ 1.0625 atol=1e-8
    @test UncertainTea._running_variance_effective_count(weighted_variance_state) ≈
          (1.25^2 / 1.0625) atol=1e-8
    robust_variance_state = UncertainTea._running_variance_state(1)
    for value in (0.0, 0.05, -0.05, 0.1, 100.0)
        UncertainTea._update_running_variance!(robust_variance_state, [value])
    end
    @test robust_variance_state.count == 5
    @test robust_variance_state.mean[1] < 1.0
    @test robust_variance_state.m2[1] < 1.0
    # `_inverse_mass_matrix` returns the regularized VARIANCE as M^{-1} (Stan
    # convention), not its reciprocal. The robustified variance is small and shrinks
    # toward 1, so the entry stays within the (regularization, 1) band.
    @test UncertainTea._inverse_mass_matrix(robust_variance_state, 1e-3)[1] ≈
          0.5282291666666666 atol = 1e-12
end
