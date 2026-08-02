using LinearAlgebra

@testset "nuts_uturn_turning" begin
    @tea static function dyadic_model()
        x ~ normal(0.0f0, 1.0f0)
        y ~ normal(0.0f0, 3.0f0)
        return x
    end

    @tea static function dyadic_model_1d()
        x ~ normal(0.0f0, 1.0f0)
        return x
    end

    function dyadic_new_state(num_params)
        return UncertainTea.NUTSState(zeros(num_params), zeros(num_params), 0.0, zeros(num_params))
    end

    function dyadic_leapfrog(target, state, imm, signed_step)
        dest = dyadic_new_state(length(state.position))
        ok = UncertainTea.leapfrog_step!(dest, target, state, imm, signed_step)
        return dest, ok
    end

    function dyadic_build_tree(target, state, imm, step_size, direction, depth)
        if depth == 0
            dest, _ = dyadic_leapfrog(target, state, imm, direction * step_size)
            return (leftmost=dest, rightmost=dest, turning=false, steps=1)
        end
        left = dyadic_build_tree(target, state, imm, step_size, direction, depth - 1)
        if left.turning
            return (leftmost=left.leftmost, rightmost=left.rightmost, turning=true, steps=left.steps)
        end
        right = dyadic_build_tree(target, left.rightmost, imm, step_size, direction, depth - 1)
        steps = left.steps + right.steps
        if right.turning
            return (leftmost=left.leftmost, rightmost=right.rightmost, turning=true, steps=steps)
        end
        if direction > 0
            turned = UncertainTea._is_turning(
                left.leftmost.position,
                right.rightmost.position,
                left.leftmost.momentum,
                right.rightmost.momentum,
            )
        else
            turned = UncertainTea._is_turning(
                right.rightmost.position,
                left.leftmost.position,
                right.rightmost.momentum,
                left.leftmost.momentum,
            )
        end
        return (leftmost=left.leftmost, rightmost=right.rightmost, turning=turned, steps=steps)
    end

    function dyadic_old_criterion(target, start_state, imm, step_size, direction, depth)
        state = start_state
        steps = 0
        turning = false
        for _ = 1:(1<<depth)
            dest, _ = dyadic_leapfrog(target, state, imm, direction * step_size)
            state = dest
            steps += 1
            if direction > 0
                turning = UncertainTea._is_turning(
                    start_state.position,
                    state.position,
                    start_state.momentum,
                    state.momentum,
                )
            else
                turning = UncertainTea._is_turning(
                    state.position,
                    start_state.position,
                    state.momentum,
                    start_state.momentum,
                )
            end
            turning && break
        end
        return (turning=turning, steps=steps)
    end

    dyadic_position = [1.6, -1.2]
    dyadic_momentum = [0.5, 0.9]
    dyadic_imm = [1.0, 1.0]
    dyadic_cache = UncertainTea._logjoint_gradient_cache(
        dyadic_model,
        dyadic_position,
        (),
        choicemap(),
    )
    dyadic_target = UncertainTea.ModelDensityTarget(
        dyadic_model,
        (),
        choicemap(),
        dyadic_cache,
    )
    dyadic_v0, dyadic_g0 = UncertainTea.target_logdensity_and_gradient!(dyadic_target, dyadic_position)
    dyadic_start_state = UncertainTea.NUTSState(
        copy(dyadic_position),
        copy(dyadic_momentum),
        dyadic_v0,
        copy(dyadic_g0),
    )
    dyadic_initial_hamiltonian = UncertainTea._hamiltonian(
        dyadic_start_state.logjoint,
        dyadic_start_state.momentum,
        dyadic_imm,
    )
    dyadic_max_delta_energy = 1.0e10
    dyadic_direction = 1

    dyadic_step_sizes = [0.3, 0.6, 0.9, 1.2, 1.5]
    dyadic_depths = [2, 3, 4]
    dyadic_turning_count, dyadic_diff_from_old = let
        turning_count = 0
        diff_from_old = 0
        for dyadic_step_size in dyadic_step_sizes
            for dyadic_depth in dyadic_depths
                dyadic_reference = dyadic_build_tree(
                    dyadic_target,
                    dyadic_start_state,
                    dyadic_imm,
                    dyadic_step_size,
                    dyadic_direction,
                    dyadic_depth,
                )
                dyadic_workspace = UncertainTea.NUTSSubtreeWorkspace(length(dyadic_position), 5)
                dyadic_summary = UncertainTea._build_nuts_subtree(
                    dyadic_workspace,
                    dyadic_target,
                    dyadic_start_state,
                    dyadic_imm,
                    dyadic_step_size,
                    dyadic_direction,
                    dyadic_depth,
                    dyadic_initial_hamiltonian,
                    dyadic_max_delta_energy,
                    MersenneTwister(hash((dyadic_step_size, dyadic_depth))),
                )
                @test dyadic_summary.divergent == false
                @test dyadic_summary.turning == dyadic_reference.turning
                @test dyadic_summary.integration_steps == dyadic_reference.steps
                dyadic_old = dyadic_old_criterion(
                    dyadic_target,
                    dyadic_start_state,
                    dyadic_imm,
                    dyadic_step_size,
                    dyadic_direction,
                    dyadic_depth,
                )
                if dyadic_reference.turning
                    turning_count += 1
                end
                if dyadic_summary.turning != dyadic_old.turning ||
                   dyadic_summary.integration_steps != dyadic_old.steps
                    diff_from_old += 1
                end
            end
        end
        (turning_count, diff_from_old)
    end
    @test dyadic_turning_count >= 4
    @test dyadic_diff_from_old >= 1

    dyadic_e2e_chains = nuts_chains(
        dyadic_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=200,
        num_warmup=200,
        step_size=0.4,
        max_tree_depth=8,
        rng=MersenneTwister(777),
    )
    dyadic_e2e_summary = summarize(dyadic_e2e_chains)
    for dyadic_row_index = 1:length(dyadic_e2e_summary)
        @test isfinite(dyadic_e2e_summary[dyadic_row_index].mean)
        @test isfinite(dyadic_e2e_summary[dyadic_row_index].sd)
        @test 1.0 <= dyadic_e2e_summary[dyadic_row_index].rhat <= 1.2
    end

    dyadic_depth_chain = first(nuts(
        dyadic_model_1d,
        (),
        choicemap();
        num_samples=300,
        num_warmup=300,
        step_size=0.4,
        rng=MersenneTwister(99),
    ))
    dyadic_depth_values = treedepths(dyadic_depth_chain)
    @test all(isfinite, dyadic_depth_chain.unconstrained_samples)
    @test (sum(dyadic_depth_values) / length(dyadic_depth_values)) <= 5

    # dyadicb_: batched (cohort) dyadic U-turn coverage. The scalar dyadic
    # promotion is validated by the grid test above (unchanged), which now
    # exercises the shared _store_tree_checkpoint! / _dyadic_turning helpers via
    # _build_nuts_subtree.

    # Direct unit test of the shared helpers against _is_turning. leaf_index 0
    # (even) stores checkpoint slot count_ones(0)+1 = 1; leaf_index 1 (odd)
    # tests block_start 0 (slot 1) versus the current endpoint.
    dyadicb_ckpt_pos = zeros(2, 3)
    dyadicb_ckpt_mom = zeros(2, 3)
    UncertainTea._store_tree_checkpoint!(
        dyadicb_ckpt_pos,
        dyadicb_ckpt_mom,
        0,
        [0.0, 0.0],
        [1.0, 0.0],
    )
    @test dyadicb_ckpt_pos[:, 1] == [0.0, 0.0]
    @test dyadicb_ckpt_mom[:, 1] == [1.0, 0.0]
    # Backward endpoint relative to the checkpoint: a U-turn under direction > 0.
    @test UncertainTea._dyadic_turning(
        dyadicb_ckpt_pos,
        dyadicb_ckpt_mom,
        1,
        [-1.0, 0.0],
        [1.0, 0.0],
        1,
        [1.0, 1.0],
    ) == UncertainTea._is_turning([0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
    @test UncertainTea._dyadic_turning(
        dyadicb_ckpt_pos,
        dyadicb_ckpt_mom,
        1,
        [-1.0, 0.0],
        [1.0, 0.0],
        1,
        [1.0, 1.0],
    )
    # Forward-progressing endpoint: no U-turn.
    @test !UncertainTea._dyadic_turning(
        dyadicb_ckpt_pos,
        dyadicb_ckpt_mom,
        1,
        [1.0, 0.0],
        [1.0, 0.0],
        1,
        [1.0, 1.0],
    )

    # metricturn_: metric-aware U-turn criterion (issue #69). The trajectory
    # velocity is M^{-1} p, so a non-identity inverse mass matrix must enter
    # every turning helper.
    metricturn_ql = [0.0, 0.0]
    metricturn_qr = [1.0, 1.0]
    metricturn_p = [1.0, -0.5]
    metricturn_imm = [0.1, 10.0]
    # Raw-momentum criterion says "not turning" (dot = 0.5) while the
    # metric-aware velocity dot is 0.1*1.0 + 10.0*(-0.5) = -4.9 <= 0: turning.
    @test !UncertainTea._is_turning(metricturn_ql, metricturn_qr, metricturn_p, metricturn_p)
    @test UncertainTea._is_turning(
        metricturn_ql,
        metricturn_qr,
        metricturn_p,
        metricturn_p,
        metricturn_imm,
    )
    # Mirrored counterexample: raw says turning (dot = -0.5), the metric-aware
    # velocity dot is 0.1*(-1.0) + 10.0*0.5 = 4.9 > 0: not turning.
    metricturn_p_mirror = [-1.0, 0.5]
    @test UncertainTea._is_turning(
        metricturn_ql,
        metricturn_qr,
        metricturn_p_mirror,
        metricturn_p_mirror,
    )
    @test !UncertainTea._is_turning(
        metricturn_ql,
        metricturn_qr,
        metricturn_p_mirror,
        metricturn_p_mirror,
        metricturn_imm,
    )
    # Identity metric reduces to the raw criterion.
    for metricturn_case_p in ([1.0, -0.5], [-1.0, 0.5], [1.0, 1.0], [-0.2, -0.1])
        @test UncertainTea._is_turning(
            metricturn_ql,
            metricturn_qr,
            metricturn_case_p,
            metricturn_case_p,
            [1.0, 1.0],
        ) == UncertainTea._is_turning(metricturn_ql, metricturn_qr, metricturn_case_p, metricturn_case_p)
    end
    # MassMetric wrappers agree with the raw diagonal vector, and a dense
    # metric with a diagonal Sigma matches the diagonal result.
    @test UncertainTea._is_turning(
        metricturn_ql,
        metricturn_qr,
        metricturn_p,
        metricturn_p,
        UncertainTea.DiagonalMetric(metricturn_imm),
    )
    @test UncertainTea._is_turning(
        metricturn_ql,
        metricturn_qr,
        metricturn_p,
        metricturn_p,
        UncertainTea.DenseMetric(Matrix(Diagonal(metricturn_imm))),
    )
    # Genuinely dense metric: check the criterion against the analytic
    # velocity dot (delta' Sigma p, Sigma = M^{-1}) on a random momentum grid.
    metricturn_dense_sigma = [1.0 -0.9; -0.9 1.0]
    metricturn_dense = UncertainTea.DenseMetric(metricturn_dense_sigma)
    metricturn_dense_rng = MersenneTwister(69)
    for _ = 1:50
        metricturn_case_pl = randn(metricturn_dense_rng, 2)
        metricturn_case_pr = randn(metricturn_dense_rng, 2)
        metricturn_delta = metricturn_qr .- metricturn_ql
        metricturn_expected =
            dot(metricturn_delta, metricturn_dense_sigma * metricturn_case_pl) <= 0 ||
            dot(metricturn_delta, metricturn_dense_sigma * metricturn_case_pr) <= 0
        @test UncertainTea._is_turning(
            metricturn_ql,
            metricturn_qr,
            metricturn_case_pl,
            metricturn_case_pr,
            metricturn_dense,
        ) == metricturn_expected
    end
    # _batched_is_turning! threads the metric per chain: shared Vector and
    # per-chain Matrix forms agree with the scalar criterion.
    metricturn_left_pos = zeros(2, 2)
    metricturn_right_pos = ones(2, 2)
    metricturn_left_mom = hcat(metricturn_p, metricturn_p_mirror)
    metricturn_right_mom = hcat(metricturn_p, metricturn_p_mirror)
    metricturn_batched = falses(2)
    UncertainTea._batched_is_turning!(
        metricturn_batched,
        metricturn_left_pos,
        metricturn_right_pos,
        metricturn_left_mom,
        metricturn_right_mom,
        metricturn_imm,
        trues(2),
    )
    @test metricturn_batched == BitVector([true, false])
    UncertainTea._batched_is_turning!(
        metricturn_batched,
        metricturn_left_pos,
        metricturn_right_pos,
        metricturn_left_mom,
        metricturn_right_mom,
        hcat(metricturn_imm, metricturn_imm),
        trues(2),
    )
    @test metricturn_batched == BitVector([true, false])
    # _dyadic_turning threads the metric into the checkpoint tests: the same
    # counterexample flips the verdict versus the identity metric.
    metricturn_ckpt_pos = zeros(2, 3)
    metricturn_ckpt_mom = zeros(2, 3)
    UncertainTea._store_tree_checkpoint!(
        metricturn_ckpt_pos,
        metricturn_ckpt_mom,
        0,
        metricturn_ql,
        metricturn_p,
    )
    @test !UncertainTea._dyadic_turning(
        metricturn_ckpt_pos,
        metricturn_ckpt_mom,
        1,
        metricturn_qr,
        metricturn_p,
        1,
        [1.0, 1.0],
    )
    @test UncertainTea._dyadic_turning(
        metricturn_ckpt_pos,
        metricturn_ckpt_mom,
        1,
        metricturn_qr,
        metricturn_p,
        1,
        metricturn_imm,
    )

    @tea static function dyadicb_model()
        x ~ normal(0.0f0, 1.0f0)
        y ~ normal(0.0f0, 2.0f0)
        return x
    end

    dyadicb_max_tree_depth = 6
    dyadicb_batched = batched_nuts(
        dyadicb_model,
        (),
        choicemap();
        num_chains=2,
        num_samples=120,
        num_warmup=120,
        step_size=0.35,
        max_tree_depth=dyadicb_max_tree_depth,
        rng=MersenneTwister(20240705),
    )
    for dyadicb_chain in dyadicb_batched
        @test all(1 <= depth <= dyadicb_chain.max_tree_depth for depth in treedepths(dyadicb_chain))
        @test all(isfinite, dyadicb_chain.unconstrained_samples)
    end
    dyadicb_rhat = rhat(dyadicb_batched)
    @test all(isfinite, dyadicb_rhat)
    @test all(0.9 <= value <= 1.3 for value in dyadicb_rhat)

    dyadicb_smc = batched_smc(
        dyadicb_model,
        (),
        choicemap();
        num_particles=24,
        proposal_loc=Float64[0.0, 0.0],
        proposal_log_scale=log.([1.0, 2.0]),
        target_ess_ratio=0.9,
        move_kernel=:nuts,
        move_steps=2,
        move_max_tree_depth=4,
        move_step_size=0.3,
        rng=MersenneTwister(20240706),
    )
    @test dyadicb_smc isa SMCResult
    @test isfinite(dyadicb_smc.importance.log_evidence_estimate)
    @test all(0.0 <= stage.move_acceptance_rate <= 1.0 for stage in dyadicb_smc.stages)
end
