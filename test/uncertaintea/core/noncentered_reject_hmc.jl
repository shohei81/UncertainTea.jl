# Noncentered reparam + batched HMC/ChEES robustness (issue #202).
#
# `eight_schools_noncentered` is a DEPENDENT-transform model (noncentered theta
# depends on mu, tau), so its batched gradient rides the ForwardDiff plan-walk
# path (`_logjoint_unconstrained_with_workspace!` -> `_dependent_transform_walk!`),
# not the analytic backend. With a fixed/long leapfrog trajectory `batched_hmc`
# and trajectory-adapting `batched_chees` overshoot to a point where a finite
# unconstrained tau maps to a non-finite constrained tau (Inf); the noncentered
# change-of-variables finite-check then USED to throw unconditionally out of the
# gradient, killing the whole run (it reproduced on `main` with the seeds/configs
# below). `batched_nuts` avoided it in practice (smaller per-leaf steps) but that
# was luck, not robustness.
#
# The fix routes that finite-check through the same reject-invalid-parameters
# handling `_compiled_distribution` uses (issue #157): in reject mode (which every
# sampler-owned workspace enables) a non-finite loc/scale makes the step
# contribute -Inf to the unconstrained log-joint -- a divergent/rejected proposal
# -- instead of throwing. Outside reject mode the check still throws, so it keeps
# catching genuine model bugs.

@testset "noncentered reject HMC/ChEES (issue #202)" begin
    @tea static function es_nc_reject(sigma)
        mu ~ normal(0.0, 5.0)
        tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
        theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
        for i = 1:8
            {:y => i} ~ normal(theta[i], sigma[i])
        end
        return mu
    end

    y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    cm = choicemap([(:y => i, y[i]) for i = 1:8]...)

    # Canonical eight-schools posterior: mu ~= 4.4, tau ~= 3-4. We assert sane,
    # finite recovery (not tight moments -- the point is robustness, not a
    # calibration test): mu-mean in a wide plausible band and healthy rhat.
    @testset "batched_hmc completes and recovers (was: throws on main)" begin
        res = batched_hmc(
            es_nc_reject,
            (sigma,),
            cm;
            num_chains=4,
            num_samples=200,
            num_warmup=200,
            num_leapfrog_steps=10,
            target_accept=0.651,
            rng=MersenneTwister(100),
        )
        summary = summarize(res)
        mu_row = summary.parameters[findfirst(r -> r.binding == :mu, summary.parameters)]
        tau_row = summary.parameters[findfirst(r -> r.binding == :tau, summary.parameters)]
        @test isfinite(mu_row.mean)
        @test 0.0 <= mu_row.mean <= 10.0
        @test isfinite(tau_row.mean)
        @test tau_row.mean >= 0.0
        @test all(r -> isfinite(r.mean) && isfinite(r.sd), summary.parameters)
        @test mu_row.rhat < 1.2
    end

    @testset "batched_chees (adapt_trajectory_length) completes and recovers" begin
        res = batched_chees(
            es_nc_reject,
            (sigma,),
            cm;
            num_chains=4,
            num_samples=200,
            num_warmup=200,
            num_leapfrog_steps=10,
            target_accept=0.651,
            adapt_trajectory_length=true,
            rng=MersenneTwister(100),
        )
        summary = summarize(res)
        mu_row = summary.parameters[findfirst(r -> r.binding == :mu, summary.parameters)]
        tau_row = summary.parameters[findfirst(r -> r.binding == :tau, summary.parameters)]
        @test isfinite(mu_row.mean)
        @test 0.0 <= mu_row.mean <= 10.0
        @test isfinite(tau_row.mean)
        @test tau_row.mean >= 0.0
        @test all(r -> isfinite(r.mean) && isfinite(r.sd), summary.parameters)
        @test mu_row.rhat < 1.2
    end

    @testset "batched_nuts still samples the model (unchanged)" begin
        res = batched_nuts(
            es_nc_reject,
            (sigma,),
            cm;
            num_chains=4,
            num_samples=200,
            num_warmup=200,
            rng=MersenneTwister(100),
        )
        summary = summarize(res)
        mu_row = summary.parameters[findfirst(r -> r.binding == :mu, summary.parameters)]
        @test isfinite(mu_row.mean)
        @test 0.0 <= mu_row.mean <= 10.0
    end

    # A non-finite noncentered loc/scale that is NOT in reject mode must still
    # throw -- the finite-check catches genuine model bugs, and the fix is gated
    # strictly on `reject_invalid_parameters`. Force tau to overflow to Inf
    # (constrained tau = exp(1000)) so theta's noncentered scale is non-finite.
    @testset "non-reject finite-check still throws" begin
        # params order follows declaration: [mu, tau, z_1..z_8].
        params = [0.0, 1000.0, zeros(8)...]
        @test_throws ArgumentError logjoint_unconstrained(es_nc_reject, params, (sigma,), cm)
        # ... and in reject mode the same evaluation is a rejection (-Inf), no throw.
        lj = logjoint_unconstrained(es_nc_reject, params, (sigma,), cm; reject_invalid_parameters=true)
        @test !isnan(lj)
        @test lj == -Inf
    end
end
