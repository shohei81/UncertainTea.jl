# PR: vectorized (broadcast) observations and iid latents.
#
# `{:y} ~ normal.(mu, sigma)` scores one dense vector value against broadcast
# elementwise arguments (a single address instead of N loop-addressed choices), and
# lowers to a backend-native `BackendBroadcastNormalChoicePlanStep` that fuses the
# per-element scoring for the batched logjoint and its manual gradient. `eps ~
# iid(dist_call, n)` sugars a length-n latent vector under a single address with a
# per-element transform.

const bc_FD = UncertainTea.ForwardDiff
const bc_LA = UncertainTea.LinearAlgebra

# --- shared models -----------------------------------------------------------
@tea static function bc_broadcast_model(xs)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(slope .* xs, sigma)
end

@tea static function bc_loop_model(xs, n)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(slope * xs[i], sigma)
    end
end

@tea static function bc_iid_model()
    eps ~ iid(normal(0.0, 1.0), 5)
    scales ~ iid(lognormal(0.0, 1.0), 3)
    return eps
end

@testset "vectorized observations and iid latents" begin
    bc_xs = collect(0.0:0.5:2.0)
    bc_n = length(bc_xs)
    bc_y = [0.1, 0.7, 1.4, 2.0, 2.9]
    bc_theta = [1.3, log(0.8)]

    @testset "bc_broadcast_matches_loop_logjoint" begin
        cm_bc = choicemap(:y => bc_y)
        cm_loop = choicemap([(:y => i) => bc_y[i] for i = 1:bc_n]...)
        lj_bc = logjoint_unconstrained(bc_broadcast_model, bc_theta, (bc_xs,), cm_bc)
        lj_loop = logjoint_unconstrained(bc_loop_model, bc_theta, (bc_xs, bc_n), cm_loop)
        @test lj_bc ≈ lj_loop atol = 1e-8

        # constrained-space parity and generate/logjoint agreement
        constrained = [1.3, 0.8]
        @test logjoint(bc_broadcast_model, constrained, (bc_xs,), cm_bc) ≈
              logjoint(bc_loop_model, constrained, (bc_xs, bc_n), cm_loop) atol = 1e-8
        # a fully-constrained generate scores the whole joint, matching logjoint
        cm_full = choicemap((:slope, 1.3), (:sigma, 0.8), (:y, bc_y))
        gen_trace, gen_weight = generate(bc_broadcast_model, (bc_xs,), cm_full; rng=MersenneTwister(7))
        gen_pv = parameter_vector(gen_trace)
        @test gen_weight ≈ logjoint(bc_broadcast_model, gen_pv, (bc_xs,), cm_bc) atol = 1e-8
    end

    @testset "bc_backend_report_and_plan" begin
        report = backend_report(bc_broadcast_model)
        @test report.supported == true
        plan = backend_execution_plan(bc_broadcast_model)
        @test any(s -> s isa UncertainTea.BackendBroadcastNormalChoicePlanStep, plan.steps)
        broadcast_index = findfirst(s -> s isa UncertainTea.BackendBroadcastNormalChoicePlanStep, plan.steps)
        @test plan.steps[broadcast_index] isa UncertainTea.BackendBroadcastNormalChoicePlanStep
    end

    @testset "bc_batched_gradient_parity_and_backend_tier" begin
        cm_bc = choicemap(:y => bc_y)
        batch_size = 4
        params = randn(MersenneTwister(11), 2, batch_size)

        batched_lj = batched_logjoint_unconstrained(bc_broadcast_model, params, (bc_xs,), cm_bc)
        for column = 1:batch_size
            single = logjoint_unconstrained(bc_broadcast_model, params[:, column], (bc_xs,), cm_bc)
            @test batched_lj[column] ≈ single atol = 1e-8
        end

        cache = BatchedLogjointGradientCache(bc_broadcast_model, params, (bc_xs,), cm_bc)
        @test cache.backend_cache !== nothing

        batched_grad = batched_logjoint_gradient_unconstrained(bc_broadcast_model, params, (bc_xs,), cm_bc)
        for column = 1:batch_size
            single = logjoint_gradient_unconstrained(bc_broadcast_model, params[:, column], (bc_xs,), cm_bc)
            @test batched_grad[:, column] ≈ single atol = 1e-8
        end
    end

    @testset "bc_generate_semantics" begin
        cm_bc = choicemap(:y => bc_y)
        # constrained observation works
        trace, weight = generate(bc_broadcast_model, (bc_xs,), cm_bc; rng=MersenneTwister(3))
        @test isfinite(weight)
        @test length(trace[(:y,)]) == bc_n

        # unconstrained with a vector argument infers the sample length
        trace2, _ = generate(bc_broadcast_model, (bc_xs,); rng=MersenneTwister(4))
        @test length(trace2[(:y,)]) == bc_n

        # unconstrained with all-scalar broadcast arguments cannot infer a length
        @tea static function bc_scalar_model()
            m ~ normal(0.0, 1.0)
            {:y} ~ normal.(m, 1.0)
        end
        @test_throws ArgumentError generate(bc_scalar_model, (); rng=MersenneTwister(5))
    end

    @testset "bc_broadcast_unsupported_family_throws" begin
        @test_throws ArgumentError macroexpand(
            @__MODULE__,
            :(@tea static function bc_bad_family()
                a ~ gamma(2.0, 1.0)
                {:y} ~ gamma.(a, 1.0)
            end),
        )
    end

    @testset "iid_layout_and_agreement" begin
        layout = parameterlayout(bc_iid_model)
        @test length(layout.slots) == 2
        eps_slot = layout.slots[1]
        scales_slot = layout.slots[2]
        @test eps_slot.binding == :eps
        @test eps_slot.value_length == 5
        @test eps_slot.dimension == 5
        @test eps_slot.transform isa VectorIdentityTransform
        @test scales_slot.value_length == 3
        @test scales_slot.transform isa VectorLogTransform
        @test parametervaluecount(layout) == 8
        @test parametercount(layout) == 8

        trace, _ = generate(bc_iid_model, (); rng=MersenneTwister(1))
        @test length(trace[(:eps,)]) == 5
        @test length(trace[(:scales,)]) == 3
        @test all(>(0), trace[(:scales,)])
        pv = parameter_vector(trace)
        cm = choicemap((:eps, trace[(:eps,)]), (:scales, trace[(:scales,)]))
        # a fully-constrained re-generate scores the whole joint, matching logjoint
        _, regen_weight = generate(bc_iid_model, (), cm; rng=MersenneTwister(2))
        @test regen_weight ≈ logjoint(bc_iid_model, pv, (), choicemap()) atol = 1e-8
    end

    @testset "iid_vector_log_transform_roundtrip" begin
        transform = VectorLogTransform(3)
        unconstrained = [0.3, -0.7, 1.2]
        constrained = UncertainTea.to_constrained(transform, unconstrained)
        @test all(>(0), constrained)
        recovered = UncertainTea.to_unconstrained(transform, constrained)
        @test recovered ≈ unconstrained atol = 1e-10

        analytic = UncertainTea.logabsdetjac(transform, unconstrained)
        jacobian = bc_FD.jacobian(u -> UncertainTea.to_constrained(transform, u), unconstrained)
        @test analytic ≈ bc_LA.logabsdet(jacobian)[1] atol = 1e-8

        # VectorLogitTransform mirrors the scalar logit element math.
        logit_transform = VectorLogitTransform(4)
        logit_u = [0.5, -1.1, 0.2, 2.0]
        logit_c = UncertainTea.to_constrained(logit_transform, logit_u)
        @test all(v -> 0 < v < 1, logit_c)
        @test UncertainTea.to_unconstrained(logit_transform, logit_c) ≈ logit_u atol = 1e-10
        logit_jac = bc_FD.jacobian(u -> UncertainTea.to_constrained(logit_transform, u), logit_u)
        @test UncertainTea.logabsdetjac(logit_transform, logit_u) ≈ bc_LA.logabsdet(logit_jac)[1] atol = 1e-8
    end

    @testset "iid_nuts_runs_finite" begin
        chain = first(nuts(
            bc_iid_model,
            (),
            choicemap();
            num_samples=60,
            num_warmup=60,
            rng=MersenneTwister(99),
        ))
        @test size(chain.constrained_samples, 2) == 60
        @test all(isfinite, chain.constrained_samples)
        # scales occupy value indices 6:8 (after the 5 eps components); constrained
        # positive after the vector-log transform.
        @test all(>(0), chain.constrained_samples[6:8, :])
    end

    @testset "iid_non_literal_n_throws" begin
        @test_throws ArgumentError macroexpand(
            @__MODULE__,
            :(@tea static function bc_iid_bad(k)
                eps ~ iid(normal(0.0, 1.0), k)
            end),
        )
    end

    @testset "bc_nuts_recovers_slope" begin
        recover_xs = collect(0.0:0.1:2.0)
        rng = MersenneTwister(2718)
        true_slope = 1.5
        recover_y = true_slope .* recover_xs .+ 0.05 .* randn(rng, length(recover_xs))
        obs = choicemap(:y => recover_y)
        chain = first(nuts(
            bc_broadcast_model,
            (recover_xs,),
            obs;
            num_samples=200,
            num_warmup=200,
            rng=MersenneTwister(2024),
        ))
        @test all(isfinite, chain.constrained_samples)
        slope_mean = sum(chain.constrained_samples[1, :]) / size(chain.constrained_samples, 2)
        @test isapprox(slope_mean, true_slope; atol=0.3)
    end
end

# Issue #87: the broadcast normal sampling path validates the scale like the
# scoring path already does, including every element of a vector sigma.
@testset "bc_broadcast_sampling_sigma_validation" begin
    @tea static function bc_neg_sigma_model()
        {:y} ~ normal.([0.0, 0.0], -1.0)
    end
    @test_throws ArgumentError generate(bc_neg_sigma_model; rng=MersenneTwister(1))

    @tea static function bc_neg_sigma_vec_model()
        {:y} ~ normal.([0.0, 0.0], [1.0, -1.0])
    end
    @test_throws ArgumentError generate(bc_neg_sigma_vec_model; rng=MersenneTwister(1))

    @test_throws ArgumentError rand(
        MersenneTwister(1),
        UncertainTea.BroadcastNormalDist([0.0, 0.0], -1.0),
    )
    @test_throws ArgumentError rand(
        MersenneTwister(1),
        UncertainTea.BroadcastNormalDist([0.0, 0.0], [1.0, 0.0]),
    )
    bc_valid_draws = rand(MersenneTwister(1), UncertainTea.BroadcastNormalDist([0.0, 0.0], 1.0))
    @test length(bc_valid_draws) == 2
    @test all(isfinite, bc_valid_draws)
end
