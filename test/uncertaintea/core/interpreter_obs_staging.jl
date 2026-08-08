# Issue #145: single-chain interpreter quick wins.
#
# 1. Dense observed-value staging: `LogjointGradientCache` (and the public
#    gradient entry point) pre-resolve loop-choice observations addressed as
#    `(literal..., loop-index)` into dense per-site Float64 vectors so the
#    scoring loop reads `values[i]` instead of assembling an address tuple and
#    hashing into the ChoiceMap. The staged path must be bit-identical to the
#    ChoiceMap path, and must fall back to live lookups whenever the staged
#    snapshot is stale (the constraints object was mutated, as gibbs does) or
#    a site is dynamic (non-Int index, non-Float64 value, unvisited index).
# 2. Expression compilation: `X[:, i]` compiles to a `view` and `sum(a .* b)`
#    to an allocation-free broadcast reduction; both read the same element
#    values in the same order as the naive forms.
# 3. The public `logjoint_gradient_unconstrained` memoizes its
#    `ForwardDiff.GradientConfig` (and observation stage) per resolved
#    signature instead of rebuilding them per call.

const stg_FD = UncertainTea.ForwardDiff

@tea static function stg_gauss(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

@tea static function stg_logistic(X, n)
    alpha ~ normal(0.0, 2.5)
    beta ~ iid(normal(0.0, 2.5), 3)
    for i = 1:n
        {:y => i} ~ bernoulli(1.0 / (1.0 + exp(-(alpha + sum(beta .* X[:, i])))))
    end
    return alpha
end

@tea static function stg_eight_schools(sigma)
    mu ~ normal(0.0, 5.0)
    tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

@tea static function stg_broadcast(xs)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(slope .* xs, sigma)
end

# non-Int loop items: the address matches the `literal => loop-index` shape
# but the index cannot key a dense vector
@tea static function stg_float_loop()
    mu ~ normal(0.0, 1.0)
    for x in (0.5, 1.5)
        {:y => x} ~ normal(mu, 1.0)
    end
end

# a non-contiguous loop range fills only the visited indices
@tea static function stg_strided(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:2:n
        {:y => i} ~ normal(mu, 1.0)
    end
end

# an address whose tail is not the bare loop index is never staged
@tea static function stg_indirect(idx, n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => idx[i]} ~ normal(mu, 1.0)
    end
end

# reference gradient through the UNstaged public density (stage=nothing path)
stg_reference_gradient(model, theta, args, cons) =
    stg_FD.gradient(t -> logjoint_unconstrained(model, t, args, cons), theta)

@testset "interpreter observation staging and expression fast paths" begin
    stg_rng = MersenneTwister(145)
    stg_n = 200
    stg_y = randn(stg_rng, stg_n) .* 1.3 .+ 0.4
    stg_gauss_cons = choicemap(((:y => i, stg_y[i]) for i = 1:stg_n)...)

    stg_ln = 60
    stg_X = randn(stg_rng, 3, stg_ln)
    stg_ly = Float64.(rand(stg_rng, stg_ln) .< 0.5)
    stg_logi_cons = choicemap(((:y => i, stg_ly[i]) for i = 1:stg_ln)...)

    stg_sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    stg_es_y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    stg_es_cons = choicemap(((:y => i, stg_es_y[i]) for i = 1:8)...)

    stg_xs = collect(0.0:0.1:2.0)
    stg_bc_y = 1.3 .* stg_xs .+ 0.2 .* randn(stg_rng, length(stg_xs))
    stg_bc_cons = choicemap(:y => stg_bc_y)

    stg_norm_lpdf(x, m, s) = -0.5 * log(2pi) - log(s) - 0.5 * ((x - m) / s)^2

    @testset "stg_staged_gradients_bitwise_match_choicemap_path" begin
        for (model, args, cons, dim) in (
            (stg_gauss, (stg_n,), stg_gauss_cons, 2),
            (stg_logistic, (stg_X, stg_ln), stg_logi_cons, 4),
            (stg_eight_schools, (stg_sigma,), stg_es_cons, 10),
            (stg_broadcast, (stg_xs,), stg_bc_cons, 2),
        )
            for seed = 1:3
                theta = randn(MersenneTwister(100 * seed + dim), dim) .* 0.4
                reference = stg_reference_gradient(model, theta, args, cons)
                # public entry point (memoized stage + config)
                @test logjoint_gradient_unconstrained(model, theta, args, cons) == reference
                # cached entry point (per-cache stage), as NUTS uses it
                cache = UncertainTea._logjoint_gradient_cache(model, theta, args, cons)
                @test copy(UncertainTea._logjoint_gradient!(cache, theta)) == reference
            end
        end
    end

    @testset "stg_stage_marking_and_snapshot" begin
        resolved = UncertainTea._resolve_signature_plan(stg_gauss, stg_gauss_cons, ())
        @test resolved.compiled.stage_count == 1
        theta = [0.1, -0.2]
        stage = UncertainTea._stage_observations(stg_gauss, resolved, theta, (stg_n,), stg_gauss_cons)
        @test stage isa UncertainTea.ObservationStage
        @test sum(stage.sites[1].filled) == stg_n
        @test stage.sites[1].values == stg_y
        @test UncertainTea._stage_is_current(stage, stg_gauss_cons)
        # a broadcast (single static address) model now stages the WHOLE observed
        # vector as one static site (iterator slot -1, issue #288), which puts it
        # on the generated-scorer / reverse-mode path
        bc_resolved = UncertainTea._resolve_signature_plan(stg_broadcast, stg_bc_cons, ())
        @test bc_resolved.compiled.stage_count == 1
        bc_stage = UncertainTea._stage_observations(stg_broadcast, bc_resolved, [0.1, -0.2], (stg_xs,), stg_bc_cons)
        @test bc_stage isa UncertainTea.ObservationStage
        @test all(bc_stage.sites[1].filled)
    end

    @testset "stg_mutated_constraints_fall_back_to_live_lookups" begin
        # gibbs mutates its merged constraints in place between gradient
        # evaluations; a stale stage must not serve the old value
        cons = choicemap(((:y => i, stg_y[i]) for i = 1:stg_n)...)
        theta = [0.3, 0.1]
        cache = UncertainTea._logjoint_gradient_cache(stg_gauss, theta, (stg_n,), cons)
        before = copy(UncertainTea._logjoint_gradient!(cache, theta))
        @test before == stg_reference_gradient(stg_gauss, theta, (stg_n,), cons)
        UncertainTea._pushchoice!(cons, :y => 3, 99.0)
        after = copy(UncertainTea._logjoint_gradient!(cache, theta))
        @test after == stg_reference_gradient(stg_gauss, theta, (stg_n,), cons)
        @test after != before

        # public path: the memoized stage is invalidated by the mutation too
        pub_cons = choicemap(((:y => i, stg_y[i]) for i = 1:stg_n)...)
        pub_before = logjoint_gradient_unconstrained(stg_gauss, theta, (stg_n,), pub_cons)
        UncertainTea._pushchoice!(pub_cons, :y => 7, -42.0)
        pub_after = logjoint_gradient_unconstrained(stg_gauss, theta, (stg_n,), pub_cons)
        @test pub_after == stg_reference_gradient(stg_gauss, theta, (stg_n,), pub_cons)
        @test pub_after != pub_before
    end

    @testset "stg_dynamic_sites_fall_back" begin
        float_cons = choicemap((:y => 0.5, 0.2), (:y => 1.5, -0.4))
        theta1 = [0.15]
        @test logjoint_gradient_unconstrained(stg_float_loop, theta1, (), float_cons) ==
              stg_reference_gradient(stg_float_loop, theta1, (), float_cons)

        # non-Float64 observed values stay on the ChoiceMap path
        f32_cons = choicemap(((:y => i, Float32(stg_y[i])) for i = 1:stg_n)...)
        @test logjoint_gradient_unconstrained(stg_gauss, [0.2, 0.0], (stg_n,), f32_cons) ==
              stg_reference_gradient(stg_gauss, [0.2, 0.0], (stg_n,), f32_cons)

        strided_cons = choicemap(((:y => i, stg_y[i]) for i = 1:2:9)...)
        @test logjoint_gradient_unconstrained(stg_strided, [0.4], (9,), strided_cons) ==
              stg_reference_gradient(stg_strided, [0.4], (9,), strided_cons)

        indirect_cons = choicemap((:y => 11, 0.3), (:y => 12, -0.6))
        indirect_resolved = UncertainTea._resolve_signature_plan(stg_indirect, indirect_cons, ())
        @test indirect_resolved.compiled.stage_count == 0
        @test logjoint_gradient_unconstrained(stg_indirect, [0.1], ([11, 12], 2), indirect_cons) ==
              stg_reference_gradient(stg_indirect, [0.1], ([11, 12], 2), indirect_cons)
    end

    @testset "stg_view_and_dot_rewrites" begin
        # semantic oracle computed with plain Julia formulas, independent of
        # the compiled expression path
        theta = randn(MersenneTwister(7), 4) .* 0.3
        alpha, beta = theta[1], theta[2:4]
        expected = stg_norm_lpdf(alpha, 0.0, 2.5)
        for j = 1:3
            expected += stg_norm_lpdf(beta[j], 0.0, 2.5)
        end
        for i = 1:stg_ln
            p = 1.0 / (1.0 + exp(-(alpha + sum(beta .* stg_X[:, i]))))
            expected += stg_ly[i] == 1.0 ? log(p) : log1p(-p)
        end
        @test logjoint(stg_logistic, theta, (stg_X, stg_ln), stg_logi_cons) ≈ expected rtol = 1e-12

        # the compiled tree really uses view/_sum_broadcast_multiply
        resolved = UncertainTea._resolve_signature_plan(stg_logistic, stg_logi_cons, ())
        loop_step = nothing
        for step in resolved.compiled.steps
            step isa UncertainTea.CompiledLoopPlanStep && (loop_step = step)
        end
        @test loop_step !== nothing
        stg_uses_view = Ref(false)
        stg_uses_dot = Ref(false)
        function stg_scan(expr)
            if expr isa UncertainTea.CompiledCallExpr
                if expr.callee isa UncertainTea.CompiledLiteralExpr
                    expr.callee.value === view && (stg_uses_view[] = true)
                    expr.callee.value === UncertainTea._sum_broadcast_multiply && (stg_uses_dot[] = true)
                end
                stg_scan(expr.callee)
                foreach(stg_scan, expr.arguments)
            end
            return nothing
        end
        stg_scan(loop_step.body[1].arguments[1])
        @test stg_uses_view[]
        @test stg_uses_dot[]

        # bitwise agreement of the fused reduction with the naive form
        for len in (1, 2, 3, 7, 8, 33, 1000)
            a = randn(MersenneTwister(len), len)
            b = randn(MersenneTwister(len + 1), len)
            @test UncertainTea._sum_broadcast_multiply(a, b) === sum(a .* b)
        end
    end

    @testset "stg_public_gradient_config_memoized" begin
        theta = [0.25, -0.1]
        logjoint_gradient_unconstrained(stg_gauss, theta, (stg_n,), stg_gauss_cons)
        resolved = UncertainTea._resolve_signature_plan(stg_gauss, stg_gauss_cons, ())
        store = resolved.gradient_config_cache[]
        @test store isa Dict
        entries = length(store)
        @test entries >= 1
        # a second call reuses the memoized config and stage
        logjoint_gradient_unconstrained(stg_gauss, theta, (stg_n,), stg_gauss_cons)
        @test length(resolved.gradient_config_cache[]) == entries
        @test resolved.observation_stage_cache[] isa UncertainTea.ObservationStage
    end

    @testset "stg_constrained_logjoint_oracle" begin
        # staging leaves the plain density entry points untouched; check the
        # gauss joint against a formula oracle (gamma(2, 1): logpdf = log(x) - x)
        cparams = [0.6, 1.4]
        expected = stg_norm_lpdf(cparams[1], 0.0, 1.0)
        expected += log(cparams[2]) - cparams[2]
        for i = 1:stg_n
            expected += stg_norm_lpdf(stg_y[i], cparams[1], cparams[2])
        end
        @test logjoint(stg_gauss, cparams, (stg_n,), stg_gauss_cons) ≈ expected rtol = 1e-12
    end
end
