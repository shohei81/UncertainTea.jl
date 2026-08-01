# Static whole-vector observation staging (issue #288): a single all-literal
# address observing a VECTOR ({:y} ~ gaussianprocess(...), {:y} ~ poisson.(...),
# {:y} ~ hmm(...)) now stages as one dense site (sentinel iterator slot -1) and
# takes the type-stable generated-scorer path — the same objective the Enzyme
# reverse-mode entry points differentiate (#268). Previously these models fell
# back to the boxed interpreter and reverse_mode_gradient(model, ...) rejected
# them outright.

using LinearAlgebra

@tea static function svo_gp(X)
    logl ~ normal(0.0, 1.0)
    logv ~ normal(0.0, 1.0)
    logn ~ normal(-1.0, 1.0)
    {:y} ~ gaussianprocess(X, exp(logl), exp(logv), exp(logn))
    return logl
end

@tea static function svo_glm(x, n)
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ poisson.(exp.(a .+ b .* x))
    return a
end

@tea static function svo_hmm(init, trans)
    m1 ~ normal(-1.0, 2.0)
    log_gap ~ normal(0.0, 1.0)
    logs ~ normal(-0.5, 0.5)
    {:y} ~ hmm(init, trans, [m1, m1 + exp(log_gap)], exp(logs))
    return m1
end

@testset "static_vector_obs_staging" begin
    svo_rng = MersenneTwister(288)
    svo_n = 16
    svo_X = reshape(sort(rand(svo_rng, svo_n) .* 5), 1, svo_n)
    svo_y = randn(svo_rng, svo_n)
    svo_gp_cm = choicemap((:y, svo_y))
    svo_x = collect(range(-1.0, 1.0; length=8))
    svo_glm_cm = choicemap((:y, Float64[2, 1, 3, 2, 4, 5, 3, 6]))
    svo_init = [0.6, 0.4]
    svo_trans = [0.8 0.2; 0.3 0.7]
    svo_hmm_cm = choicemap((:y, randn(MersenneTwister(3), 30)))

    @testset "GP / broadcast GLM / HMM take the generated-scorer path" begin
        cases = (
            (svo_gp, [0.1, 0.2, -0.8], (svo_X,), svo_gp_cm),
            (svo_glm, [0.3, 0.5], (svo_x, 8), svo_glm_cm),
            (svo_hmm, [0.1, 0.2, -0.3], (svo_init, svo_trans), svo_hmm_cm),
        )
        for (model, theta, args, cm) in cases
            gradient = logjoint_gradient_unconstrained(model, theta, args, cm)
            @test UncertainTea._GEN_SCORER_LAST_USED[]
            @test all(isfinite, gradient)

            # the generated objective scores the SAME logjoint as the interpreter
            objective = UncertainTea._generated_gradient_objective_or_nothing(model, theta, args, cm)
            @test objective !== nothing
            @test Base.invokelatest(objective, theta) ==
                  logjoint_unconstrained(model, theta, args, cm)

            # gradient parity against the interpreter-backed batched column path
            batched = batched_logjoint_gradient_unconstrained(
                model, reshape(theta, length(theta), 1), args, cm,
            )
            @test isapprox(batched[:, 1], gradient; rtol=1e-9, atol=1e-12)
        end
    end

    @testset "the stage holds the whole observed vector" begin
        resolved = UncertainTea._resolve_signature_plan(svo_gp, svo_gp_cm)
        @test resolved.compiled.stage_count == 1
        stage = UncertainTea._stage_observations(svo_gp, resolved, [0.1, 0.2, -0.8], (svo_X,), svo_gp_cm)
        @test stage isa UncertainTea.ObservationStage
        @test stage.sites[1].values == svo_y
        @test all(stage.sites[1].filled)
    end

    @testset "non-Float64 vector observations fall back to the interpreter" begin
        svo_f32_cm = choicemap((:y, Float32.(svo_y)))
        gradient = logjoint_gradient_unconstrained(svo_gp, [0.1, 0.2, -0.8], (svo_X,), svo_f32_cm)
        @test !UncertainTea._GEN_SCORER_LAST_USED[]
        @test all(isfinite, gradient)
    end

    @testset "reverse_mode_gradient(model, ...) accepts these models (no Enzyme -> MethodError)" begin
        # the objective is built (no ArgumentError rejection anymore); the inner
        # call raises MethodError because Enzyme is not loaded in this suite
        @test_throws MethodError reverse_mode_gradient(svo_gp, [0.1, 0.2, -0.8], (svo_X,), svo_gp_cm)
    end
end
