@testset "dist_discrete_gaps" begin
    loggamma = UncertainTea.loggamma
    digamma = UncertainTea.digamma
    logbeta = (a, b) -> loggamma(a) + loggamma(b) - loggamma(a + b)

    # --- logpdf correctness vs closed form -----------------------------------
    let n = 10, a = 2.0, b = 3.0, k = 4
        ref =
            (loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)) +
            logbeta(k + a, n - k + b) - logbeta(a, b)
        @test UncertainTea.logpdf(betabinomial(n, a, b), k) ≈ ref atol = 1e-12
        # a proper pmf sums to 1 over its finite support 0..n
        @test sum(exp(UncertainTea.logpdf(betabinomial(n, a, b), j)) for j = 0:n) ≈ 1.0 atol = 1e-10
        # out-of-support and non-integer values score -Inf
        @test UncertainTea.logpdf(betabinomial(n, a, b), n + 1) == -Inf
        @test UncertainTea.logpdf(betabinomial(n, a, b), 1.5) == -Inf
    end

    let N = 12, p = [0.2, 0.5, 0.3], x = [3, 4, 5]
        ref = loggamma(N + 1) - sum(loggamma(xi + 1) for xi in x) + sum(x[i] * log(p[i]) for i = 1:3)
        @test UncertainTea.logpdf(multinomial(N, p), x) ≈ ref atol = 1e-12
        # counts must sum to N and match the simplex length
        @test UncertainTea.logpdf(multinomial(N, p), [3, 4, 4]) == -Inf
        @test UncertainTea.logpdf(multinomial(N, p), [6, 6]) == -Inf
    end

    let du = discreteuniform(1, 6)
        @test UncertainTea.logpdf(du, 4) ≈ -log(6.0) atol = 1e-12
        @test UncertainTea.logpdf(du, 7) == -Inf
        @test UncertainTea.logpdf(du, 0) == -Inf
        # negative-inclusive support
        @test UncertainTea.logpdf(discreteuniform(-3, 3), -2) ≈ -log(7.0) atol = 1e-12
    end

    # --- rand stays in support -----------------------------------------------
    let rng = MersenneTwister(230)
        for _ = 1:64
            @test 0 <= rand(rng, betabinomial(20, 2.0, 5.0)) <= 20
            draw = rand(rng, multinomial(10, [0.2, 0.5, 0.3]))
            @test length(draw) == 3 && sum(draw) == 10 && all(>=(0), draw)
            @test -3 <= rand(rng, discreteuniform(-3, 3)) <= 3
        end
    end

    # --- betabinomial: backend scoring + analytic (digamma) gradient ---------
    @tea static function betabinomial_model()
        log_alpha ~ normal(0.0, 0.5)
        log_beta ~ normal(0.0, 0.5)
        alpha = exp(log_alpha)
        beta = exp(log_beta)
        {:y} ~ betabinomial(20, alpha, beta)
        return alpha + beta
    end

    betabinomial_constraints = choicemap((:y, 8))
    betabinomial_trace, _ =
        generate(betabinomial_model, (), betabinomial_constraints; rng=MersenneTwister(231))
    betabinomial_plan = backend_execution_plan(betabinomial_model)
    betabinomial_params = parameter_vector(betabinomial_trace)
    betabinomial_batch = hcat(
        betabinomial_params .+ Float64[-0.2, 0.1],
        betabinomial_params,
        betabinomial_params .+ Float64[0.3, -0.15],
    )
    betabinomial_batch_constraints = [choicemap((:y, 3)), choicemap((:y, 8)), choicemap((:y, 15))]
    betabinomial_batch_gradient = batched_logjoint_gradient_unconstrained(
        betabinomial_model,
        betabinomial_batch,
        (),
        betabinomial_batch_constraints,
    )
    betabinomial_cache = BatchedLogjointGradientCache(
        betabinomial_model,
        betabinomial_batch,
        (),
        betabinomial_batch_constraints,
    )

    @test backend_report(betabinomial_model).supported
    @test betabinomial_plan.steps[end] isa UncertainTea.BackendBetaBinomialChoicePlanStep
    # analytic batched gradient matches the per-column ForwardDiff reference tightly
    @test betabinomial_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                betabinomial_model,
                betabinomial_batch[:, index],
                (),
                betabinomial_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol = 1e-8
    # the analytic backend tier is the one exercised (no flat ForwardDiff fallback)
    @test !isnothing(betabinomial_cache.backend_cache)
    @test isnothing(betabinomial_cache.flat_cache)

    # direct digamma-derivative check against a finite difference of the log-pmf
    let n = 20, k = 8, a = 1.7, b = 2.9, h = 1e-6
        dalpha = digamma(k + a) - digamma(n + a + b) - digamma(a) + digamma(a + b)
        fd =
            (
                UncertainTea.logpdf(betabinomial(n, a + h, b), k) -
                UncertainTea.logpdf(betabinomial(n, a - h, b), k)
            ) / (2h)
        @test dalpha ≈ fd atol = 1e-6
    end

    # --- discreteuniform: backend scoring + (zero) gradient ------------------
    @tea static function discreteuniform_model()
        mu ~ normal(0.0, 1.0)
        {:y} ~ discreteuniform(1, 10)
        return mu
    end

    discreteuniform_constraints = choicemap((:y, 4))
    discreteuniform_trace, _ =
        generate(discreteuniform_model, (), discreteuniform_constraints; rng=MersenneTwister(232))
    discreteuniform_plan = backend_execution_plan(discreteuniform_model)
    discreteuniform_params = parameter_vector(discreteuniform_trace)
    discreteuniform_batch =
        reshape(discreteuniform_params[1] .+ Float64[-0.1, 0.0, 0.2], 1, 3)
    discreteuniform_batch_constraints = [choicemap((:y, 1)), choicemap((:y, 4)), choicemap((:y, 10))]
    discreteuniform_batch_gradient = batched_logjoint_gradient_unconstrained(
        discreteuniform_model,
        discreteuniform_batch,
        (),
        discreteuniform_batch_constraints,
    )

    @test backend_report(discreteuniform_model).supported
    @test discreteuniform_plan.steps[end] isa UncertainTea.BackendDiscreteUniformChoicePlanStep
    @test discreteuniform_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                discreteuniform_model,
                discreteuniform_batch[:, index],
                (),
                discreteuniform_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol = 1e-8
    # the observed discreteuniform term is a constant -log(10): the gradient of
    # the (only) latent equals that of the bare normal prior alone
    @test discreteuniform_batch_gradient ≈ -discreteuniform_batch atol = 1e-10
    # out-of-support conditioning scores -Inf through the backend logjoint
    @test logjoint(discreteuniform_model, discreteuniform_params, (), choicemap((:y, 11))) == -Inf

    # --- observed-loop fast path (betabinomial) ------------------------------
    @tea static function betabinomial_loop_model()
        log_alpha ~ normal(0.0, 0.5)
        for i = 1:4
            {(:y, i)} ~ betabinomial(10, exp(log_alpha), 2.0)
        end
        return log_alpha
    end

    betabinomial_loop_observations = [2, 5, 7, 3]
    betabinomial_loop_constraints =
        choicemap([((:y, i), betabinomial_loop_observations[i]) for i = 1:4]...)
    betabinomial_loop_trace, _ = generate(
        betabinomial_loop_model,
        (),
        betabinomial_loop_constraints;
        rng=MersenneTwister(233),
    )
    betabinomial_loop_params = parameter_vector(betabinomial_loop_trace)
    betabinomial_loop_batch =
        reshape(betabinomial_loop_params[1] .+ Float64[-0.1, 0.0, 0.2], 1, 3)
    betabinomial_loop_batch_constraints =
        [betabinomial_loop_constraints, betabinomial_loop_constraints, betabinomial_loop_constraints]
    betabinomial_loop_scores = batched_logjoint_unconstrained(
        betabinomial_loop_model,
        betabinomial_loop_batch,
        (),
        betabinomial_loop_batch_constraints,
    )
    betabinomial_loop_gradient = batched_logjoint_gradient_unconstrained(
        betabinomial_loop_model,
        betabinomial_loop_batch,
        (),
        betabinomial_loop_batch_constraints,
    )

    @test betabinomial_loop_scores ≈ [
        logjoint_unconstrained(
            betabinomial_loop_model,
            betabinomial_loop_batch[:, index],
            (),
            betabinomial_loop_batch_constraints[index],
        ) for index = 1:3
    ] atol = 1e-8
    @test betabinomial_loop_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                betabinomial_loop_model,
                betabinomial_loop_batch[:, index],
                (),
                betabinomial_loop_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol = 1e-8

    # --- multinomial: dirichlet-multinomial through the compiled + AD tier ---
    @tea static function dirichlet_multinomial_model()
        theta ~ dirichlet(2.0, 3.0, 4.0)
        {:y} ~ multinomial(12, theta)
        return theta
    end

    dirichlet_multinomial_observations = [3, 4, 5]
    dirichlet_multinomial_constraints = choicemap((:y, dirichlet_multinomial_observations))
    dirichlet_multinomial_trace, _ = generate(
        dirichlet_multinomial_model,
        (),
        dirichlet_multinomial_constraints;
        rng=MersenneTwister(234),
    )
    dirichlet_multinomial_theta = dirichlet_multinomial_trace[:theta]

    # multinomial is a vector observation: the backend path is deferred, so it
    # is honestly reported unsupported and scores through the compiled plan.
    @test !backend_report(dirichlet_multinomial_model).supported
    @test assess(
        dirichlet_multinomial_model,
        (),
        choicemap((:theta, dirichlet_multinomial_theta), (:y, dirichlet_multinomial_observations)),
    ) ≈ (
        UncertainTea.logpdf(dirichlet(2.0, 3.0, 4.0), dirichlet_multinomial_theta) +
        UncertainTea.logpdf(multinomial(12, dirichlet_multinomial_theta), dirichlet_multinomial_observations)
    ) atol = 1e-10
    # the continuous latent still differentiates through the ForwardDiff tier
    dirichlet_multinomial_gradient = logjoint_gradient_unconstrained(
        dirichlet_multinomial_model,
        transform_to_unconstrained(dirichlet_multinomial_trace),
        (),
        dirichlet_multinomial_constraints,
    )
    @test all(isfinite, dirichlet_multinomial_gradient)

    # --- device: all three honestly reject (no #218 binom-coeff IR) ----------
    for (model, model_constraints) in (
        (betabinomial_model, betabinomial_constraints),
        (discreteuniform_model, discreteuniform_constraints),
        (dirichlet_multinomial_model, dirichlet_multinomial_constraints),
    )
        supported, issues = UncertainTea.device_lowering_report(model; constraints=model_constraints)
        @test !supported
        @test !isempty(issues)
    end
end
