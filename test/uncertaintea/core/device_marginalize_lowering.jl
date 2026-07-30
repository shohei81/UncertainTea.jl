# issue #67: device mirror of the marginalize=:enumerate step. A finite-support
# discrete latent is summed out on the device by folding the per-support-value
# suffix scores through a max-shifted log-sum-exp, so discrete-enumeration models
# (mixtures / indicators) run on the Metal (or future CUDA) device path. All tests
# here run on KernelAbstractions.CPU() Float64/Float32; the GPU smoke under
# test/gpu/ mirrors the posterior recovery on a Metal backend at Float32.
#
# The device path takes UNCONSTRAINED parameters and folds the transform log-abs-det
# in-kernel, so the authoritative CPU counterparts are `batched_logjoint_unconstrained`
# (values) and `batched_logjoint_gradient_unconstrained` (gradients).

using KernelAbstractions: CPU

# Local mean helper (Statistics is not imported by the test harness).
dmarg_mean(x) = sum(x) / length(x)

# --- marginalized models (mirroring the discrete_enum_cpu.jl acceptance set) ----

# the flagship indicator / two-Gaussian model: the suffix mean depends on z.
@tea static function dmarg_indicator_model()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
    return m1
end

# categorical support (1..K); the suffix mean scales with the category index.
@tea static function dmarg_categorical_model()
    mu ~ normal(0.0, 1.0)
    z ~ categorical([0.2, 0.3, 0.5]; marginalize=:enumerate)
    {:y} ~ normal(mu * z, 0.4)
    return mu
end

# two marginalized latents -> product enumeration through nested device steps.
@tea static function dmarg_two_latent_model()
    mu ~ normal(0.0, 1.0)
    a ~ bernoulli(0.4; marginalize=:enumerate)
    b ~ bernoulli(0.7; marginalize=:enumerate)
    {:y} ~ normal(mu + a + 2 * b, 0.5)
    return mu
end

# latent-flowing branch probabilities: d(logsumexp)/dp flows through the pmf.
@tea static function dmarg_dependent_p_model()
    x ~ beta(2.0, 2.0)
    z ~ bernoulli(x; marginalize=:enumerate)
    {:y} ~ normal(z * 2.0, 1.0)
    return x
end

# a zero-mass branch (bernoulli(1.0) puts no mass on z=false, whose suffix is
# unevaluable): the fold must exclude it by its FULL term, not throw.
@tea static function dmarg_zero_mass_model()
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(1.0; marginalize=:enumerate)
    {:y} ~ normal(mu, z * 1.0)
    return mu
end

# an impossible branch (finite prior mass, -Inf suffix: bernoulli(0) with y=true):
# excluding by the FULL term keeps the marginal gradient finite (issue #62 lesson).
@tea static function dmarg_impossible_branch_model()
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(0.5; marginalize=:enumerate)
    {:y} ~ bernoulli(1.0 * z)
    {:w} ~ normal(mu + z, 1.0)
    return mu
end

# a suffix that rebinds a PRE-EXISTING slot: the device rejects it honestly (the
# per-branch write would leak across enumeration branches with no restore).
@tea static function dmarg_rebind_model()
    mu ~ normal(0.0, 1.0)
    s = 1.0
    z ~ bernoulli(0.4; marginalize=:enumerate)
    s = s + 1.0
    {:y} ~ normal(mu * s + z, 0.5)
    return mu
end

# a suffix owning a loop: rejected (loop-id alignment + Metal compile budget).
@tea static function dmarg_loop_suffix_model(n)
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(0.5; marginalize=:enumerate)
    for i = 1:n
        {:y => i} ~ normal(mu + z, 1.0)
    end
    return mu
end

# unimodal contamination recovery: the z=1 branch shifts far and is rare, so the
# z=0 mode dominates and mu is identified unimodally (a clean R-hat target).
@tea static function dmarg_recover_model()
    mu ~ normal(0.0, 3.0)
    z ~ bernoulli(0.1; marginalize=:enumerate)
    {:y1} ~ normal(mu + z * 12.0, 0.8)
    {:y2} ~ normal(mu + z * 12.0, 0.8)
    {:y3} ~ normal(mu + z * 12.0, 0.8)
    {:y4} ~ normal(mu + z * 12.0, 0.8)
    return mu
end

@testset "device_marginalize_lowering" begin
    @testset "dmarg_report_supported" begin
        for model in (
            dmarg_indicator_model,
            dmarg_categorical_model,
            dmarg_two_latent_model,
            dmarg_dependent_p_model,
            dmarg_zero_mass_model,
            dmarg_impossible_branch_model,
        )
            supported, issues = device_lowering_report(model)
            @test supported
            @test isempty(issues)
        end

        # honest rejections for what the device deliberately does not cover.
        rebind_supported, rebind_issues = device_lowering_report(dmarg_rebind_model)
        @test !rebind_supported
        @test any(occursin("rebinds the pre-existing binding", issue) for issue in rebind_issues)

        loop_supported, loop_issues = device_lowering_report(dmarg_loop_suffix_model)
        @test !loop_supported
        @test any(occursin("suffix containing a loop", issue) for issue in loop_issues)
    end

    @testset "dmarg_logjoint_parity" begin
        # bernoulli indicator (per-column marginalize)
        cm = choicemap((:y, 0.8))
        params = [-1.5 -1.2 0.3; 1.7 2.1 -0.4]
        dev = device_batched_logjoint(dmarg_indicator_model, params, (), cm)
        ref = batched_logjoint_unconstrained(dmarg_indicator_model, params, (), cm)
        @test dev ≈ ref rtol = 1e-12
        dev32 = device_batched_logjoint(dmarg_indicator_model, Float32.(params), (), cm; precision=Float32)
        @test all(isapprox(Float64(d), r; rtol=1e-4, atol=1e-4 * max(1.0, abs(r))) for (d, r) in zip(dev32, ref))

        # the conditioned single-branch shortcut: conditioning selects one branch,
        # matching the plain joint at that value on the CPU path.
        for zv in (true, false)
            cmz = choicemap((:z, zv), (:y, 0.8))
            devz = device_batched_logjoint(dmarg_indicator_model, params, (), cmz)
            refz = batched_logjoint_unconstrained(dmarg_indicator_model, params, (), cmz)
            @test devz ≈ refz rtol = 1e-12
        end

        # categorical (support 1..K)
        cparams = reshape([0.6, -0.3], 1, 2)
        ccm = choicemap((:y, 1.1))
        @test device_batched_logjoint(dmarg_categorical_model, cparams, (), ccm) ≈
              batched_logjoint_unconstrained(dmarg_categorical_model, cparams, (), ccm) rtol = 1e-12

        # product enumeration (two nested marginalized latents)
        tparams = reshape([0.3, -0.2], 1, 2)
        tcm = choicemap((:y, 2.2))
        @test device_batched_logjoint(dmarg_two_latent_model, tparams, (), tcm) ≈
              batched_logjoint_unconstrained(dmarg_two_latent_model, tparams, (), tcm) rtol = 1e-12

        # latent-flowing branch probabilities
        dparams = reshape([0.2, -0.4], 1, 2)
        dcm = choicemap((:y, 1.5))
        @test device_batched_logjoint(dmarg_dependent_p_model, dparams, (), dcm) ≈
              batched_logjoint_unconstrained(dmarg_dependent_p_model, dparams, (), dcm) rtol = 1e-12

        # a zero-mass branch is excluded, matching the single live branch
        zparams = reshape([0.4, 0.1], 1, 2)
        zcm = choicemap((:y, 0.9))
        @test device_batched_logjoint(dmarg_zero_mass_model, zparams, (), zcm) ≈
              batched_logjoint_unconstrained(dmarg_zero_mass_model, zparams, (), zcm) rtol = 1e-12
    end

    @testset "dmarg_gradient_parity" begin
        # a hand-rolled central finite difference of the CPU logjoint: the
        # independent oracle for the responsibility-weighted marginal gradient.
        dmarg_fd = function (model, x, args, constraints)
            fdg = similar(x)
            for i in eachindex(x)
                h = cbrt(eps(Float64)) * max(1.0, abs(x[i]))
                xp = copy(x)
                xp[i] += h
                xm = copy(x)
                xm[i] -= h
                fdg[i] =
                    (
                        logjoint_unconstrained(model, xp, args, constraints) -
                        logjoint_unconstrained(model, xm, args, constraints)
                    ) / (2h)
            end
            return fdg
        end

        # indicator: device gradient == CPU analytic gradient == finite difference
        cm = choicemap((:y, 0.8))
        params = [-1.5 -1.2; 1.7 2.1]
        v, g = device_batched_logjoint_gradient(dmarg_indicator_model, params, (), cm)
        gref = batched_logjoint_gradient_unconstrained(dmarg_indicator_model, params, (), cm)
        vref = batched_logjoint_unconstrained(dmarg_indicator_model, params, (), cm)
        @test g ≈ gref rtol = 1e-10
        @test v ≈ vref rtol = 1e-12
        for index = 1:2
            @test g[:, index] ≈ dmarg_fd(dmarg_indicator_model, params[:, index], (), cm) atol = 5e-6
        end
        _, g32 = device_batched_logjoint_gradient(dmarg_indicator_model, Float32.(params), (), cm; precision=Float32)
        @test all(isapprox(Float64(a), b; rtol=2e-3, atol=2e-3) for (a, b) in zip(vec(g32), vec(gref)))

        # categorical + latent-flowing probabilities
        for (model, p, c) in (
            (dmarg_categorical_model, reshape([0.6, -0.3], 1, 2), choicemap((:y, 1.1))),
            (dmarg_dependent_p_model, reshape([0.2, -0.4], 1, 2), choicemap((:y, 1.5))),
            (dmarg_two_latent_model, reshape([0.3, -0.2], 1, 2), choicemap((:y, 2.2))),
        )
            gv, gg = device_batched_logjoint_gradient(model, p, (), c)
            ggref = batched_logjoint_gradient_unconstrained(model, p, (), c)
            @test gg ≈ ggref rtol = 1e-10
            for index = 1:size(p, 2)
                @test gg[:, index] ≈ dmarg_fd(model, p[:, index], (), c) atol = 5e-6
            end
        end

        # an impossible branch (finite mass, -Inf suffix) is excluded by its FULL
        # term, so its degenerate branch derivative cannot poison the marginal
        # gradient through 0 * NaN.
        icm = choicemap((:y, true), (:w, 1.2))
        iparams = reshape([0.4, -0.6], 1, 2)
        iv, ig = device_batched_logjoint_gradient(dmarg_impossible_branch_model, iparams, (), icm)
        igref = batched_logjoint_gradient_unconstrained(dmarg_impossible_branch_model, iparams, (), icm)
        @test all(isfinite, ig)
        @test ig ≈ igref rtol = 1e-8
        for index = 1:2
            @test ig[:, index] ≈ dmarg_fd(dmarg_impossible_branch_model, iparams[:, index], (), icm) atol = 5e-6
        end

        # zero-mass branch gradient stays finite and matches the CPU analytic tier
        zparams = reshape([0.4, 0.1], 1, 2)
        zcm = choicemap((:y, 0.9))
        _, zg = device_batched_logjoint_gradient(dmarg_zero_mass_model, zparams, (), zcm)
        zgref = batched_logjoint_gradient_unconstrained(dmarg_zero_mass_model, zparams, (), zcm)
        @test all(isfinite, zg)
        @test zg ≈ zgref rtol = 1e-10
    end

    @testset "dmarg_posterior_recovery" begin
        # batched_nuts on a marginalized model recovers the (unimodal) posterior on
        # CPU() Float64: R-hat < 1.05 and finite draws. The Metal Float32 mirror is
        # the test/gpu smoke.
        data = choicemap((:y1, 1.8), (:y2, 2.2), (:y3, 1.9), (:y4, 2.1))
        chains = batched_nuts(
            dmarg_recover_model, (), data;
            num_chains=4, num_samples=300, num_warmup=300,
            tree_strategy=:masked, backend=CPU(), rng=MersenneTwister(67),
        )
        draws = posterior_array(chains)
        @test all(isfinite, draws)
        @test maximum(rhat(chains)) < 1.05
        # the z=0 mode dominates, so mu concentrates near the data mean (~2.0)
        @test isapprox(dmarg_mean(draws), 2.0; atol=0.3)
        divrate =
            sum(sum(chain.divergent) for chain in chains.chains) /
            sum(length(chain.divergent) for chain in chains.chains)
        @test divrate < 0.05
    end
end
