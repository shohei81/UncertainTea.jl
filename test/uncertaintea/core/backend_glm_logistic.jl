# Issue #150: the GLM observation site (`bernoullilogit` + the fused linear
# predictor `alpha + sum(beta .* X[:, i])`) backend-lowers, so logistic
# regression rides the HOST batched analytic path instead of the per-column
# ForwardDiff fallback. The observation step is a
# `BackendBernoulliLogitChoicePlanStep` whose eta is a `BackendLinearPredictorExpr`;
# the coefficient latent VECTOR contributes `d_eta/d_coef[d] = X[d, index]`
# seeded onto its unconstrained rows, and the intercept flows through its own
# sub-expr gradient. Correctness is validated same-process against the compiled
# `logjoint`/`logjoint_gradient_unconstrained` interpreter reference (no golden
# snapshots — a reassociated reduction is only tolerance-equal, and CI runs a
# different OS/Julia).

@testset "backend_glm_logistic" begin
    # d = 3 covariates, n = 6 observations; X is d x n (column i = obs i).
    X = [
        0.5 -1.2 0.3 0.9 -0.4 1.1
        -0.7 0.4 1.5 -0.2 0.8 -1.0
        1.3 0.1 -0.6 0.5 -1.1 0.2
    ]
    n = size(X, 2)
    yobs = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]

    @tea static function glm_logit(X, n)
        alpha ~ normal(0.0, 2.5)
        beta ~ mvnormal((0.0, 0.0, 0.0), (1.5, 1.5, 1.5))
        for i = 1:n
            {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
        end
        return alpha
    end

    cons = choicemap(((:y => i, yobs[i]) for i = 1:n)...)

    # lowering now supported, with the fused linear-predictor node
    @test backend_report(glm_logit).supported == true
    plan = backend_execution_plan(glm_logit)
    loopstep = plan.steps[findfirst(s -> s isa UncertainTea.BackendLoopPlanStep, plan.steps)]
    obs = loopstep.body[1]
    @test obs isa UncertainTea.BackendBernoulliLogitChoicePlanStep
    @test obs.eta isa UncertainTea.BackendLinearPredictorExpr

    pc = parametercount(parameterlayout(glm_logit))  # 1 + 3
    @test pc == 4

    # interpreter-vs-analytic parity across several parameter vectors, incl. a
    # saturating one (large eta where a sigmoid parameterization would NaN)
    params = Float64[
        0.3 -1.0 2.0 -8.0
        0.5 1.2 -0.7 6.0
        -0.4 0.3 1.1 -5.0
        0.8 -0.6 0.2 7.0
    ]
    K = size(params, 2)
    batched_lj = batched_logjoint_unconstrained(glm_logit, params, (X, n), cons)
    batched_grad = batched_logjoint_gradient_unconstrained(glm_logit, params, (X, n), cons)
    for k = 1:K
        ref_lj = logjoint_unconstrained(glm_logit, params[:, k], (X, n), cons)
        ref_grad = logjoint_gradient_unconstrained(glm_logit, params[:, k], (X, n), cons)
        @test batched_lj[k] ≈ ref_lj rtol = 1e-9
        @test batched_grad[:, k] ≈ ref_grad rtol = 1e-9
        @test all(isfinite, batched_grad[:, k])   # stability at the saturating vector
    end

    # cache provenance: the analytic backend tier is actually taken (no flat
    # fallback, no per-column caches)
    glm_cache = BatchedLogjointGradientCache(glm_logit, params, (X, n), cons)
    @test !isnothing(glm_cache.backend_cache)
    @test isnothing(glm_cache.flat_cache)
    @test isempty(glm_cache.column_caches)

    # ties #149 and #150 together: bernoullilogit vs bernoulli(sigmoid) agree at
    # moderate eta (both finite), same joint density
    @tea static function glm_sigmoid(X, n)
        alpha ~ normal(0.0, 2.5)
        beta ~ mvnormal((0.0, 0.0, 0.0), (1.5, 1.5, 1.5))
        for i = 1:n
            {:y => i} ~ bernoulli(1.0 / (1.0 + exp(-(alpha + sum(beta .* X[:, i])))))
        end
        return alpha
    end
    moderate = params[:, 1]
    @test logjoint_unconstrained(glm_logit, moderate, (X, n), cons) ≈
          logjoint_unconstrained(glm_sigmoid, moderate, (X, n), cons) rtol = 1e-9

    # an intercept-free linear predictor `sum(beta .* X[:, i])` also lowers
    @tea static function glm_no_intercept(X, n)
        beta ~ mvnormal((0.0, 0.0, 0.0), (1.5, 1.5, 1.5))
        for i = 1:n
            {:y => i} ~ bernoullilogit(sum(beta .* X[:, i]))
        end
        return beta[1]
    end
    @test backend_report(glm_no_intercept).supported == true
    ni_obs = backend_execution_plan(glm_no_intercept).steps[end].body[1]
    @test ni_obs.eta isa UncertainTea.BackendLinearPredictorExpr
    @test isnothing(ni_obs.eta.intercept)
    ni_params = Float64[0.4 -1.0; -0.3 0.7; 0.9 0.2]
    ni_batched = batched_logjoint_gradient_unconstrained(glm_no_intercept, ni_params, (X, n), cons)
    for k = 1:2
        @test ni_batched[:, k] ≈
              logjoint_gradient_unconstrained(glm_no_intercept, ni_params[:, k], (X, n), cons) rtol = 1e-9
    end

    # bernoullilogit(scalar_expr) still lowers via the generic numeric eta path
    @tea static function scalar_logit()
        s ~ normal(0.0, 1.0)
        {:y} ~ bernoullilogit(2.0 * s - 0.5)
        return s
    end
    @test backend_report(scalar_logit).supported == true
    sl_obs = backend_execution_plan(scalar_logit).steps[2]
    @test sl_obs isa UncertainTea.BackendBernoulliLogitChoicePlanStep
    @test !(sl_obs.eta isa UncertainTea.BackendLinearPredictorExpr)
    sl_params = reshape(Float64[-0.6, 0.2, 1.4], 1, 3)
    sl_cons = [choicemap((:y, 1.0)), choicemap((:y, 0.0)), choicemap((:y, 1.0))]
    @test batched_logjoint_gradient_unconstrained(scalar_logit, sl_params, (), sl_cons) ≈ hcat(
        [logjoint_gradient_unconstrained(scalar_logit, sl_params[:, i], (), sl_cons[i]) for i = 1:3]...,
    ) rtol = 1e-9

    # Float32 batched path matches the Float64 interpreter within f32 tolerance
    X32 = Float32.(X)
    params32 = Float32.(params[:, 1:2])
    g32 = batched_logjoint_gradient_unconstrained(glm_logit, params32, (X32, n), cons)
    for k = 1:2
        ref = logjoint_gradient_unconstrained(glm_logit, Float64.(params32[:, k]), (X, n), cons)
        @test Float64.(g32[:, k]) ≈ ref rtol = 1e-3
    end
end
