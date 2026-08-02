@testset "proposal_diagnostics_overflow" begin
    proposal_favorable = UncertainTea._proposal_diagnostics(
        0.0,
        [0.0],
        ([0.0], [0.0], 500.0),
        [1.0],
        100.0,
    )
    @test proposal_favorable[2] ≈ -500.0 atol=1e-12
    @test !proposal_favorable[3]

    proposal_unfavorable = UncertainTea._proposal_diagnostics(
        0.0,
        [0.0],
        ([0.0], [0.0], -500.0),
        [1.0],
        100.0,
    )
    @test proposal_unfavorable[2] ≈ 500.0 atol=1e-12
    @test proposal_unfavorable[3]

    @tea static function overflow_scale_model()
        s ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(0.0f0, exp(s))
        return s
    end

    overflow_constraints = choicemap((:y, 0.5f0))
    overflow_proposal_loc = [709.7827]
    overflow_proposal_log_scale = log(0.01)

    overflow_is = @test_logs (:warn, r"non-finite") match_mode=:any batched_importance_sampling(
        overflow_scale_model,
        (),
        overflow_constraints;
        num_particles=64,
        proposal_loc=overflow_proposal_loc,
        proposal_log_scale=overflow_proposal_log_scale,
        rng=MersenneTwister(207),
    )
    @test isfinite(overflow_is.log_evidence_estimate)
    @test 0.0 < overflow_is.effective_sample_size < 64.0
    @test any(logweight -> logweight == -Inf, overflow_is.logweights)
    @test all(isfinite, overflow_is.normalized_weights)

    overflow_smc = @test_logs (:warn, r"non-finite") match_mode=:any batched_smc(
        overflow_scale_model,
        (),
        overflow_constraints;
        num_particles=64,
        proposal_loc=overflow_proposal_loc,
        proposal_log_scale=overflow_proposal_log_scale,
        rng=MersenneTwister(211),
    )
    @test overflow_smc isa SMCResult
    @test isfinite(overflow_smc.importance.log_evidence_estimate)

    overflow_advi = @test_logs (:warn, r"non-finite") match_mode=:any batched_advi(
        overflow_scale_model,
        (),
        overflow_constraints;
        num_particles=32,
        num_iterations=3,
        initial_params=overflow_proposal_loc,
        initial_log_scale=overflow_proposal_log_scale,
        rng=MersenneTwister(223),
    )
    @test all(isfinite, variational_mean(overflow_advi; space=:unconstrained))
    @test all(isfinite, overflow_advi.elbo_history)
end
