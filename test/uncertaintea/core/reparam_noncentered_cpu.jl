using KernelAbstractions: CPU as ReparamDeviceCPU

# CPU semantics of reparam=:noncentered (PR-3 of docs/noncentered-reparam.md):
# the slot holds the standardized z, the trace/constrained space keeps theta,
# and the plan-walk transform materializes theta = loc + scale * z with the
# log|scale| change-of-variables term.

# weak observation (sigma = 10): the classic setting where the centered
# funnel defeats NUTS while the non-centered form mixes cleanly
@tea static function ncc_funnel_centered()
    v ~ normal(0.0, 3.0)
    x ~ normal(0.0, exp(v / 2))
    {:y} ~ normal(x, 10.0)
    return v
end

@tea static function ncc_funnel_flagged()
    v ~ normal(0.0, 3.0)
    x ~ normal(0.0, exp(v / 2); reparam=:noncentered)
    {:y} ~ normal(x, 10.0)
    return v
end

@tea static function ncc_funnel_manual()
    v ~ normal(0.0, 3.0)
    z ~ normal(0.0, 1.0)
    x = exp(v / 2) * z
    {:y} ~ normal(x, 10.0)
    return v
end

@tea static function ncc_arg_model(location_offset)
    tau ~ lognormal(0.0, 0.5)
    theta ~ normal(location_offset, tau; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

# a bound discrete latent has no parameter slot, so the walk cannot know its
# value: a location depending on it must be rejected
@tea static function ncc_slotless_dependent_model()
    k ~ bernoulli(0.5)
    theta ~ normal(k * 2.0, 1.0; reparam=:noncentered)
    {:y} ~ normal(theta, 1.0)
    return theta
end

# the codex-review case: the slotless dependency flows through a comparison
@tea static function ncc_slotless_branch_model()
    k ~ bernoulli(0.5)
    theta ~ normal(ifelse(k > 0, 2.0, 0.0), 1.0; reparam=:noncentered)
    {:y} ~ normal(theta, 1.0)
    return theta
end

@tea static function ncc_studentt_model()
    mu ~ normal(0.0, 1.0)
    theta ~ studentt(4.0, mu, 2.0; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

# well-identified hierarchical normal for SBC (the weak funnel can push v so
# far negative during step-size exploration that exp(v/2) underflows to zero,
# a pre-existing robustness limit of scale expressions)
@tea static function ncc_sbc_model()
    mu ~ normal(0.0, 1.0)
    tau ~ lognormal(0.0, 0.5)
    theta ~ normal(mu, tau; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

@tea static function ncc_laplace_model()
    mu ~ normal(0.0, 1.0)
    theta ~ laplace(mu, 1.5; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

# eight-schools shape (weak observations): the issue #19 acceptance model
@tea static function ncc_eight_schools_centered()
    mu ~ normal(0.0, 5.0)
    log_tau ~ normal(0.0, 1.5)
    tau = exp(log_tau)
    theta ~ iid(normal(mu, tau), 8)
    for i = 1:8
        {:y => i} ~ normal(theta[i], 10.0)
    end
    return mu
end

@tea static function ncc_eight_schools_flagged()
    mu ~ normal(0.0, 5.0)
    log_tau ~ normal(0.0, 1.5)
    tau = exp(log_tau)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], 10.0)
    end
    return mu
end

@tea static function ncc_lognormal_model()
    mu ~ normal(0.0, 1.0)
    theta ~ lognormal(mu, 0.7; reparam=:noncentered)
    {:y} ~ normal(theta, 0.5)
    return theta
end

@tea static function ncc_auto_model()
    mu ~ normal(0.0, 1.0)
    latent_loc ~ normal(mu, 2.0; reparam=:auto)
    literal_args ~ normal(0.0, 1.0; reparam=:auto)
    {:y} ~ normal(latent_loc + literal_args, 0.5)
    return latent_loc
end

# --- reachability of the noncentered transform walk (issue #100) -------------
#
# The transform walk must only evaluate deterministic steps that a noncentered
# loc/scale actually depends on. Here the enumerated `k` (a slotless choice
# bound to a poison value during the walk) flows through the deterministic
# binding `w` into the OBSERVATION mean only -- it never reaches the noncentered
# loc/scale (theta ~ normal(0, tau)). Eager evaluation of `w` used to poison the
# walk and throw a misleading error; the reachability fix leaves `w` unevaluated.
@tea static function ncc_reach_marginalize()
    tau ~ lognormal(0.0, 1.0)
    theta ~ normal(0.0, tau; reparam=:noncentered)
    k ~ bernoulli(0.5; marginalize=:enumerate)
    w = k * 2.0
    {:y} ~ normal(w + theta, 1.0)
end

# same model with `k` used directly in the observation's distribution
# arguments -- no deterministic binding in between. This spelling already
# worked before the fix and is the reference the walk must reproduce.
@tea static function ncc_reach_marginalize_direct()
    tau ~ lognormal(0.0, 1.0)
    theta ~ normal(0.0, tau; reparam=:noncentered)
    k ~ bernoulli(0.5; marginalize=:enumerate)
    {:y} ~ normal(k * 2.0 + theta, 1.0)
end

# centered twin of ncc_reach_marginalize (theta without reparam): the
# noncentered unconstrained logjoint equals this one at the matching theta plus
# the log|scale| Jacobian.
@tea static function ncc_reach_marginalize_centered()
    tau ~ lognormal(0.0, 1.0)
    theta ~ normal(0.0, tau)
    k ~ bernoulli(0.5; marginalize=:enumerate)
    w = k * 2.0
    {:y} ~ normal(w + theta, 1.0)
end

# a plain (non-marginalized) discrete latent conditioned through the
# constraints, flowing through a deterministic step into the observation only.
@tea static function ncc_reach_conditioned_discrete()
    tau ~ lognormal(0.0, 1.0)
    theta ~ normal(0.0, tau; reparam=:noncentered)
    k ~ bernoulli(0.5)
    w = k * 2.0
    {:y} ~ normal(w + theta, 1.0)
end

# NEGATIVE case: the enumerated `k` genuinely flows (through a deterministic
# step) into a noncentered SCALE, so the walk cannot proceed and must still
# raise the (now-accurate) error.
@tea static function ncc_reach_negative()
    k ~ bernoulli(0.5; marginalize=:enumerate)
    s = k * 1.0 + 1.0
    theta ~ normal(0.0, s; reparam=:noncentered)
    {:y} ~ normal(theta, 1.0)
end

@testset "reparam_noncentered_cpu" begin
    ncc_constraints = choicemap((:y, 0.5))

    @testset "ncc_transform_round_trip_and_density" begin
        z = [0.7, -1.2]
        constrained = transform_to_constrained(ncc_funnel_flagged, z)
        @test constrained[1] == z[1]
        @test constrained[2] ≈ exp(z[1] / 2) * z[2] atol = 1e-12
        @test transform_to_unconstrained(ncc_funnel_flagged, constrained) ≈ z atol = 1e-12

        # z-space density: N(v; 0, 3) + N(z; 0, 1) + likelihood at theta
        x_value = exp(z[1] / 2) * z[2]
        manual =
            UncertainTea.logpdf(normal(0.0, 3.0), z[1]) +
            UncertainTea.logpdf(normal(0.0, 1.0), z[2]) +
            UncertainTea.logpdf(normal(x_value, 10.0), 0.5)
        @test logjoint_unconstrained(ncc_funnel_flagged, z, (), ncc_constraints) ≈ manual atol = 1e-10

        gradient = logjoint_gradient_unconstrained(ncc_funnel_flagged, z, (), ncc_constraints)
        for i in eachindex(z)
            h = cbrt(eps(Float64))
            up = copy(z)
            up[i] += h
            down = copy(z)
            down[i] -= h
            fd =
                (
                    logjoint_unconstrained(ncc_funnel_flagged, up, (), ncc_constraints) -
                    logjoint_unconstrained(ncc_funnel_flagged, down, (), ncc_constraints)
                ) / (2h)
            @test gradient[i] ≈ fd atol = 5e-6
        end

        # trace round trip: traces keep theta on the constrained scale
        trace, _ = generate(ncc_funnel_flagged, (), ncc_constraints; rng=MersenneTwister(3))
        unconstrained = transform_to_unconstrained(trace)
        @test transform_to_constrained(ncc_funnel_flagged, unconstrained) ≈ parameter_vector(trace) atol = 1e-12
    end

    @testset "ncc_argument_dependent_location" begin
        z = [0.3, 0.9]
        constrained = transform_to_constrained(ncc_arg_model, z, (2.0,))
        @test constrained[2] ≈ 2.0 + exp(z[1]) * z[2] atol = 1e-12
        @test transform_to_unconstrained(ncc_arg_model, constrained, (2.0,)) ≈ z atol = 1e-12
        # the walk needs the model arguments
        @test_throws DimensionMismatch transform_to_constrained(ncc_arg_model, z)
        chain = nuts(
            ncc_arg_model,
            (2.0,),
            ncc_constraints;
            num_samples=30,
            num_warmup=30,
            rng=MersenneTwister(4),
        )
        @test all(isfinite, chain.constrained_samples)
    end

    @testset "ncc_slotless_dependent_location_rejected" begin
        @test_throws ArgumentError transform_to_constrained(ncc_slotless_dependent_model, [0.5])
        # comparisons and branching must not swallow the poison (NaN > 0 would
        # silently be false); the sentinel throws instead
        @test_throws ArgumentError transform_to_constrained(ncc_slotless_branch_model, [0.5])
    end

    @testset "ncc_studentt_laplace_density" begin
        z = [0.4, -0.8]
        observation = choicemap((:y, 0.4))
        # noncentered studentt: N(z2; 0, 1)-analog is the standardized studentt
        studentt_theta = z[1] + 2.0 * z[2]
        studentt_manual =
            UncertainTea.logpdf(normal(0.0, 1.0), z[1]) +
            UncertainTea.logpdf(studentt(4.0, 0.0, 1.0), z[2]) +
            UncertainTea.logpdf(normal(studentt_theta, 0.5), 0.4)
        @test logjoint_unconstrained(ncc_studentt_model, z, (), observation) ≈ studentt_manual atol = 1e-10

        laplace_theta = z[1] + 1.5 * z[2]
        laplace_manual =
            UncertainTea.logpdf(normal(0.0, 1.0), z[1]) +
            UncertainTea.logpdf(laplace(0.0, 1.0), z[2]) +
            UncertainTea.logpdf(normal(laplace_theta, 0.5), 0.4)
        @test logjoint_unconstrained(ncc_laplace_model, z, (), observation) ≈ laplace_manual atol = 1e-10
    end

    @testset "ncc_funnel_acceptance" begin
        total_centered = 0.0
        total_flagged = 0.0
        flagged_vs = Float64[]
        manual_vs = Float64[]
        for chain_seed = 1:6
            centered = nuts(
                ncc_funnel_centered,
                (),
                ncc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(600 + chain_seed),
            )
            flagged = nuts(
                ncc_funnel_flagged,
                (),
                ncc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(600 + chain_seed),
            )
            manual = nuts(
                ncc_funnel_manual,
                (),
                ncc_constraints;
                num_samples=300,
                num_warmup=200,
                rng=MersenneTwister(600 + chain_seed),
            )
            total_centered += divergencerate(centered)
            total_flagged += divergencerate(flagged)
            append!(flagged_vs, flagged.constrained_samples[1, :])
            append!(manual_vs, manual.constrained_samples[1, :])
        end
        # calibrated on both Julia versions: centered ~0.4 total divergence and
        # mean ~-2.2 (stuck in the neck); flagged ~0.02 and unbiased
        @test total_flagged < total_centered / 4
        @test total_flagged / 6 < 0.02

        # (the centered chains are also badly biased into the neck on most
        # streams -- mean ~ -2.2 vs ~ -0.2 -- but with 6 chains that signal is
        # stream-dependent, so only the divergence and agreement margins are
        # asserted)
        flagged_mean = sum(flagged_vs) / length(flagged_vs)
        manual_mean = sum(manual_vs) / length(manual_vs)
        flagged_sd = sqrt(sum(abs2, flagged_vs .- flagged_mean) / (length(flagged_vs) - 1))
        manual_sd = sqrt(sum(abs2, manual_vs .- manual_mean) / (length(manual_vs) - 1))
        @test abs(flagged_mean - manual_mean) < 0.5
        @test 0.8 < flagged_sd / manual_sd < 1.3
    end

    @testset "ncc_batched_backend" begin
        points = [0.6 -0.4; -1.1 0.8]
        batched_values = batched_logjoint_unconstrained(ncc_funnel_flagged, points, (), ncc_constraints)
        reference_values =
            [logjoint_unconstrained(ncc_funnel_flagged, points[:, i], (), ncc_constraints) for i = 1:2]
        @test batched_values ≈ reference_values atol = 1e-8

        cache = BatchedLogjointGradientCache(ncc_funnel_flagged, points, (), ncc_constraints)
        @test !isnothing(cache.backend_cache)
        batched_gradients = batched_logjoint_gradient_unconstrained(ncc_funnel_flagged, points, (), ncc_constraints)
        reference_gradients = hcat(
            [logjoint_gradient_unconstrained(ncc_funnel_flagged, points[:, i], (), ncc_constraints) for i = 1:2]...,
        )
        @test batched_gradients ≈ reference_gradients atol = 1e-8

        # the constrained entry must keep scoring theta space (per-column
        # fallback behind the z-space backend plan)
        theta_points = hcat([transform_to_constrained(ncc_funnel_flagged, points[:, i]) for i = 1:2]...)
        @test batched_logjoint(ncc_funnel_flagged, theta_points, (), ncc_constraints) ≈
              [logjoint(ncc_funnel_flagged, theta_points[:, i], (), ncc_constraints) for i = 1:2] atol = 1e-8

        batched = batched_nuts(
            ncc_funnel_flagged,
            (),
            ncc_constraints;
            num_chains=3,
            num_samples=25,
            num_warmup=25,
            rng=MersenneTwister(5),
        )
        @test length(batched.chains) == 3
        @test all(all(isfinite, chain.constrained_samples) for chain in batched.chains)
    end

    @testset "ncc_backend_guards" begin
        # a noncentered location fed by a slotless (discrete) latent is
        # rejected at lowering, matching the CPU transform
        report = backend_report(ncc_slotless_dependent_model)
        @test report.supported == false
        @test any(occursin("without a parameter slot", issue) for issue in report.issues)
    end

    @testset "ncc_device_parity" begin
        supported, _ = device_lowering_report(ncc_funnel_flagged)
        @test supported
        # model arguments stage into the device slots (issue #38 fixed), so
        # argument-dependent locations run with full parity
        arg_supported, _ = device_lowering_report(ncc_arg_model)
        @test arg_supported
        arg_points = [0.2 -0.3; 0.7 0.1]
        arg_constraints = choicemap((:y, 0.4))
        arg_values, arg_gradients = device_batched_logjoint_gradient(
            ncc_arg_model,
            arg_points,
            (2.5,),
            arg_constraints;
            backend=ReparamDeviceCPU(),
            precision=Float64,
        )
        @test collect(arg_values) ≈
              [logjoint_unconstrained(ncc_arg_model, arg_points[:, i], (2.5,), arg_constraints) for i = 1:2] atol =
            1e-10
        @test collect(arg_gradients) ≈ hcat(
            [
                logjoint_gradient_unconstrained(ncc_arg_model, arg_points[:, i], (2.5,), arg_constraints) for
                i = 1:2
            ]...,
        ) atol = 1e-10
        points = [0.3 -0.5; 0.9 0.2]
        device_values, device_gradients = device_batched_logjoint_gradient(
            ncc_funnel_flagged,
            points,
            (),
            ncc_constraints;
            backend=ReparamDeviceCPU(),
            precision=Float64,
        )
        reference_values =
            [logjoint_unconstrained(ncc_funnel_flagged, points[:, i], (), ncc_constraints) for i = 1:2]
        reference_gradients = hcat(
            [logjoint_gradient_unconstrained(ncc_funnel_flagged, points[:, i], (), ncc_constraints) for i = 1:2]...,
        )
        @test collect(device_values) ≈ reference_values atol = 1e-10
        @test collect(device_gradients) ≈ reference_gradients atol = 1e-10

        device_chains = batched_hmc(
            ncc_funnel_flagged,
            (),
            ncc_constraints;
            num_chains=2,
            num_samples=25,
            num_warmup=25,
            backend=ReparamDeviceCPU(),
            precision=Float64,
            rng=MersenneTwister(6),
        )
        @test all(all(isfinite, chain.constrained_samples) for chain in device_chains.chains)
        device_nuts = batched_nuts(
            ncc_funnel_flagged,
            (),
            ncc_constraints;
            num_chains=2,
            num_samples=20,
            num_warmup=20,
            tree_strategy=:masked,
            backend=ReparamDeviceCPU(),
            precision=Float64,
            rng=MersenneTwister(7),
        )
        @test all(all(isfinite, chain.constrained_samples) for chain in device_nuts.chains)
    end

    @testset "ncc_iid_eight_schools" begin
        eight_ys = [2.8, 0.8, -0.3, 0.7, -0.1, 0.1, 1.8, 1.2]
        eight_constraints = choicemap([(:y => i, eight_ys[i]) for i = 1:8]...)

        # z-space density identity (theta = mu .+ tau .* z)
        z = [0.4, -0.7, 0.3, 1.1, -0.2, 0.6, -0.9, 0.5, 0.1, -0.4]
        constrained = transform_to_constrained(ncc_eight_schools_flagged, z)
        tau = exp(z[2])
        expected_theta = z[1] .+ tau .* z[3:10]
        @test constrained[3:10] ≈ expected_theta atol = 1e-12
        @test transform_to_unconstrained(ncc_eight_schools_flagged, constrained) ≈ z atol = 1e-12
        manual =
            UncertainTea.logpdf(normal(0.0, 5.0), z[1]) +
            UncertainTea.logpdf(normal(0.0, 1.5), z[2]) +
            sum(UncertainTea.logpdf(normal(0.0, 1.0), z[2+i]) for i = 1:8) +
            sum(UncertainTea.logpdf(normal(expected_theta[i], 10.0), eight_ys[i]) for i = 1:8)
        @test logjoint_unconstrained(ncc_eight_schools_flagged, z, (), eight_constraints) ≈ manual atol = 1e-10

        gradient = logjoint_gradient_unconstrained(ncc_eight_schools_flagged, z, (), eight_constraints)
        step_size = cbrt(eps(Float64))
        for i in eachindex(z)
            up = copy(z)
            up[i] += step_size
            down = copy(z)
            down[i] -= step_size
            fd =
                (
                    logjoint_unconstrained(ncc_eight_schools_flagged, up, (), eight_constraints) -
                    logjoint_unconstrained(ncc_eight_schools_flagged, down, (), eight_constraints)
                ) / (2 * step_size)
            @test gradient[i] ≈ fd atol = 1e-5
        end

        # acceptance: near-zero divergences vs the centered form
        total_centered = 0.0
        total_flagged = 0.0
        for chain_seed = 1:4
            centered = nuts(
                ncc_eight_schools_centered,
                (),
                eight_constraints;
                num_samples=250,
                num_warmup=200,
                rng=MersenneTwister(800 + chain_seed),
            )
            flagged = nuts(
                ncc_eight_schools_flagged,
                (),
                eight_constraints;
                num_samples=250,
                num_warmup=200,
                rng=MersenneTwister(800 + chain_seed),
            )
            total_centered += divergencerate(centered)
            total_flagged += divergencerate(flagged)
        end
        @test total_flagged / 4 < 0.01
        @test total_flagged < total_centered

        # backend/device stay honest for the vector form...
        @test backend_report(ncc_eight_schools_flagged).supported == false
        # ...while the batched fallback routes through the dependent walk
        # (codex review on PR-6): value/gradient parity and batched_nuts work
        points = randn(MersenneTwister(31), 10, 2) .* 0.5
        batched_values = batched_logjoint_unconstrained(ncc_eight_schools_flagged, points, (), eight_constraints)
        reference_values =
            [logjoint_unconstrained(ncc_eight_schools_flagged, points[:, i], (), eight_constraints) for i = 1:2]
        @test batched_values ≈ reference_values atol = 1e-8
        batched_gradients =
            batched_logjoint_gradient_unconstrained(ncc_eight_schools_flagged, points, (), eight_constraints)
        reference_gradients = hcat(
            [
                logjoint_gradient_unconstrained(ncc_eight_schools_flagged, points[:, i], (), eight_constraints) for
                i = 1:2
            ]...,
        )
        @test batched_gradients ≈ reference_gradients atol = 1e-8
        batched_chains = batched_nuts(
            ncc_eight_schools_flagged,
            (),
            eight_constraints;
            num_chains=2,
            num_samples=15,
            num_warmup=15,
            rng=MersenneTwister(32),
        )
        @test all(all(isfinite, chain.constrained_samples) for chain in batched_chains.chains)
    end

    @testset "ncc_lognormal_logspace" begin
        z = [0.3, -0.6]
        observation = choicemap((:y, 0.8))
        constrained = transform_to_constrained(ncc_lognormal_model, z)
        @test constrained[2] ≈ exp(0.3 + 0.7 * -0.6) atol = 1e-12
        @test transform_to_unconstrained(ncc_lognormal_model, constrained) ≈ z atol = 1e-12
        manual =
            UncertainTea.logpdf(normal(0.0, 1.0), z[1]) +
            UncertainTea.logpdf(normal(0.0, 1.0), z[2]) +
            UncertainTea.logpdf(normal(constrained[2], 0.5), 0.8)
        @test logjoint_unconstrained(ncc_lognormal_model, z, (), observation) ≈ manual atol = 1e-10
        gradient = logjoint_gradient_unconstrained(ncc_lognormal_model, z, (), observation)
        step_size = cbrt(eps(Float64))
        for i = 1:2
            up = copy(z)
            up[i] += step_size
            down = copy(z)
            down[i] -= step_size
            fd =
                (
                    logjoint_unconstrained(ncc_lognormal_model, up, (), observation) -
                    logjoint_unconstrained(ncc_lognormal_model, down, (), observation)
                ) / (2 * step_size)
            @test gradient[i] ≈ fd atol = 1e-5
        end
    end

    @testset "ncc_auto_resolution" begin
        layout = parameterlayout(ncc_auto_model)
        @test layout.slots[2].transform isa UncertainTea.NoncenteredTransform
        @test layout.slots[3].transform isa IdentityTransform
        trace, _ = generate(ncc_auto_model, (), choicemap((:y, 0.4)); rng=MersenneTwister(8))
        unconstrained = transform_to_unconstrained(trace)
        @test transform_to_constrained(ncc_auto_model, unconstrained) ≈ parameter_vector(trace) atol = 1e-12
    end

    @testset "ncc_sbc" begin
        result = sbc(
            ncc_sbc_model;
            num_simulations=40,
            num_samples=16,
            num_warmup=60,
            rng=MersenneTwister(9),
        )
        @test all(result.pvalues .> 0.005)
        @test !has_warnings(result)
    end

    @testset "ncc_walk_reachability" begin
        reach_params = [0.2, 0.7]
        reach_constraints = choicemap((:y, 1.0))

        @testset "marginalize_through_deterministic_step" begin
            # the #100 repro now computes rather than throwing
            value = logjoint_unconstrained(ncc_reach_marginalize, reach_params, (), reach_constraints)
            @test isfinite(value)

            # oracle 1: identical to the spelling that feeds `k` directly into
            # the observation (no deterministic binding in between)
            direct =
                logjoint_unconstrained(ncc_reach_marginalize_direct, reach_params, (), reach_constraints)
            @test value ≈ direct atol = 1e-12

            # oracle 2: marginalizing k == logsumexp of conditioning on each of
            # its support values (docs/discrete-enumeration.md). Conditioning
            # changes the signature and takes the observed-value path.
            branch_false = logjoint_unconstrained(
                ncc_reach_marginalize,
                reach_params,
                (),
                choicemap((:y, 1.0), (:k, false)),
            )
            branch_true = logjoint_unconstrained(
                ncc_reach_marginalize,
                reach_params,
                (),
                choicemap((:y, 1.0), (:k, true)),
            )
            m = max(branch_false, branch_true)
            marginal = m + log(exp(branch_false - m) + exp(branch_true - m))
            @test value ≈ marginal atol = 1e-10

            # cross-check against the centered-equivalent model: the noncentered
            # unconstrained logjoint equals the centered twin at the matching
            # theta = tau * z, plus exactly the log|scale| Jacobian (log(tau)).
            tau = exp(reach_params[1])
            theta = tau * reach_params[2]
            centered =
                logjoint_unconstrained(ncc_reach_marginalize_centered, [reach_params[1], theta], (), reach_constraints)
            @test value ≈ centered + log(tau) atol = 1e-10

            # gradient matches ForwardDiff through the marginalized walk
            gradient =
                logjoint_gradient_unconstrained(ncc_reach_marginalize, reach_params, (), reach_constraints)
            fd = UncertainTea.ForwardDiff.gradient(
                p -> logjoint_unconstrained(ncc_reach_marginalize, p, (), reach_constraints),
                reach_params,
            )
            @test gradient ≈ fd atol = 1e-10
        end

        @testset "conditioned_discrete_through_deterministic_step" begin
            for k_value in (false, true)
                cond = choicemap((:y, 1.0), (:k, k_value))
                value =
                    logjoint_unconstrained(ncc_reach_conditioned_discrete, reach_params, (), cond)
                @test isfinite(value)
                # oracle: hand-built z-space density with the observed k folded
                # into the observation mean (theta = tau * z, Jacobian log(tau))
                tau = exp(reach_params[1])
                theta = tau * reach_params[2]
                w = (k_value ? 1.0 : 0.0) * 2.0
                manual =
                    UncertainTea.logpdf(lognormal(0.0, 1.0), tau) +
                    log(tau) +
                    UncertainTea.logpdf(normal(0.0, 1.0), reach_params[2]) +
                    UncertainTea.logpdf(bernoulli(0.5), k_value) +
                    UncertainTea.logpdf(normal(w + theta, 1.0), 1.0)
                @test value ≈ manual atol = 1e-10

                gradient =
                    logjoint_gradient_unconstrained(ncc_reach_conditioned_discrete, reach_params, (), cond)
                fd = UncertainTea.ForwardDiff.gradient(
                    p -> logjoint_unconstrained(ncc_reach_conditioned_discrete, p, (), cond),
                    reach_params,
                )
                @test gradient ≈ fd atol = 1e-10
            end
        end

        @testset "slotless_into_noncentered_scale_still_rejected" begin
            err = try
                logjoint_unconstrained(ncc_reach_negative, [0.3], (), choicemap((:y, 0.5)))
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            message = sprint(showerror, err)
            @test occursin("noncentered", message)
            @test occursin("marginalize=:enumerate", message)
        end
    end
end
