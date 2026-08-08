# Runtime-dimension vector latents (issue #289, PR-2 through PR-4).
#
# `theta ~ mvnormal(zeros(n), ones(n))` with a model argument `n` resolves its
# parameter slot at signature-resolution time: `_resolve_runtime_dims` walks
# the dependency cone of the size-bearing argument against the model arguments
# (choice bindings poisoned), and the signature pass late-constructs the
# transform from the resolved Int (src/evaluator/runtime_dims.jl, ir.jl) --
# `VectorIdentityTransform(n)` for the mv normal/Student-t families,
# `SimplexTransform(n)` (parameter dimension n-1) for dirichlet. The crux test
# is TWIN EQUALITY: the runtime-dim model must score bitwise identically to a
# literal-length twin on every CPU path. Batched/device lowering and the
# args-independent layout APIs are honestly unsupported, each with a tested
# failure mode. wishart/inversewishart/lkjcholesky are literal-dimension by
# design (macro-time errors, pinned below) and never become candidates.

const rdl_UT = UncertainTea

@tea static function rdl_runtime(n)
    theta ~ mvnormal(zeros(n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
    return theta
end

@tea static function rdl_literal3()
    theta ~ mvnormal([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    {:obs} ~ normal(sum(theta), 1.0)
    return theta
end

@tea static function rdl_literal7()
    theta ~ mvnormal([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    {:obs} ~ normal(sum(theta), 1.0)
    return theta
end

@tea static function rdl_dense_runtime(n, L)
    theta ~ mvnormaldense(zeros(n), L)
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_dense_literal(L)
    theta ~ mvnormaldense([0.0, 0.0, 0.0], L)
    {:obs} ~ normal(sum(theta), 1.0)
end

# The latent-mean case: the mean VALUES may depend on an earlier latent as
# long as the LENGTH comes from the arguments alone (`fill(z, n)` has length
# `n` whatever `z` is).
@tea static function rdl_latent_mean(n)
    z ~ normal(0.0, 1.0)
    theta ~ mvnormal(fill(z, n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_latent_mean_lit()
    z ~ normal(0.0, 1.0)
    theta ~ mvnormal(fill(z, 3), ones(3))
    {:obs} ~ normal(sum(theta), 1.0)
end

# identical bodies for the fresh-vs-cached comparison
@tea static function rdl_cache_session(n)
    theta ~ mvnormal(zeros(n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_cache_fresh3(n)
    theta ~ mvnormal(zeros(n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_cache_fresh5(n)
    theta ~ mvnormal(zeros(n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
end

# latent-dependent LENGTH: rejected (dims must be argument-computable)
@tea static function rdl_lengthdep_discrete()
    k ~ poisson(3.0)
    theta ~ mvnormal(zeros(k), ones(k))
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_lengthdep_det(n)
    z ~ normal(0.0, 1.0)
    m = n + round(Int, z)
    theta ~ mvnormal(zeros(m), ones(m))
    {:obs} ~ normal(sum(theta), 1.0)
end

# PR-3 families: runtime-length dirichlet / mvstudentt / mvstudenttdense
@tea static function rdl_dirichlet_runtime(alpha)
    w ~ dirichlet(alpha)
    {:obs} ~ normal(w[1], 1.0)
end

@tea static function rdl_dirichlet_lit3()
    w ~ dirichlet([2.0, 3.0, 4.0])
    {:obs} ~ normal(w[1], 1.0)
end

@tea static function rdl_dirichlet_lit4()
    w ~ dirichlet([2.0, 3.0, 4.0, 5.0])
    {:obs} ~ normal(w[1], 1.0)
end

@tea static function rdl_dirichlet_fresh3(alpha)
    w ~ dirichlet(alpha)
    {:obs} ~ normal(w[1], 1.0)
end

@tea static function rdl_dirichlet_fresh4(alpha)
    w ~ dirichlet(alpha)
    {:obs} ~ normal(w[1], 1.0)
end

# latent-dependent concentration LENGTH: still rejected (values may depend on
# latents, the length may not)
@tea static function rdl_dirichlet_lengthdep()
    k ~ poisson(3.0)
    w ~ dirichlet(2.0 .* ones(k))
    {:obs} ~ normal(w[1], 1.0)
end

@tea static function rdl_mvt_runtime(n)
    theta ~ mvstudentt(4.0, zeros(n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_mvt_lit3()
    theta ~ mvstudentt(4.0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_mvt_lit7()
    theta ~ mvstudentt(4.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_mvt_fresh3(n)
    theta ~ mvstudentt(4.0, zeros(n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_mvt_fresh7(n)
    theta ~ mvstudentt(4.0, zeros(n), ones(n))
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_mvtd_runtime(n, L)
    theta ~ mvstudenttdense(4.0, zeros(n), L)
    {:obs} ~ normal(sum(theta), 1.0)
end

@tea static function rdl_mvtd_literal(L)
    theta ~ mvstudenttdense(4.0, [0.0, 0.0, 0.0], L)
    {:obs} ~ normal(sum(theta), 1.0)
end

# GP teaser: a latent function value with a runtime-length zero mean and an
# in-model gp_cholesky scale
@tea static function rdl_gp_runtime(xs, n)
    f ~ mvnormaldense(zeros(n), gp_cholesky(xs, 1.0, 0.5, 1e-6))
    {:obs} ~ normal(sum(f), 1.0)
end

@tea static function rdl_gp_literal(xs)
    f ~ mvnormaldense([0.0, 0.0, 0.0], gp_cholesky(xs, 1.0, 0.5, 1e-6))
    {:obs} ~ normal(sum(f), 1.0)
end

# Interpreter-vs-generated comparison at one parameter vector (mirrors
# generated_scorer_identity.jl): returns both results plus whether the
# generated path served the ON run.
function rdl_gen_compare(model, args, cons, params)
    prev = rdl_UT._USE_GENERATED_SCORER[]
    try
        rdl_UT._USE_GENERATED_SCORER[] = false
        lu_i = logjoint_unconstrained(model, params, args, cons)
        gr_i = copy(logjoint_gradient_unconstrained(model, params, args, cons))
        rdl_UT._USE_GENERATED_SCORER[] = true
        lu_g = logjoint_unconstrained(model, params, args, cons)
        used = rdl_UT._GEN_SCORER_LAST_USED[]
        gr_g = copy(logjoint_gradient_unconstrained(model, params, args, cons))
        return (; lu_i, lu_g, gr_i, gr_g, used)
    finally
        rdl_UT._USE_GENERATED_SCORER[] = prev
    end
end

@testset "runtime_dim_latents" begin
    U = rdl_UT
    rdl_cons = choicemap((:obs, 1.2))

    @testset "twin equality: runtime-dim mvnormal == literal twin (n=3, n=7)" begin
        cases = (
            (rdl_runtime, (3,), rdl_literal3, (), [0.3, -0.4, 0.25]),
            (rdl_runtime, (7,), rdl_literal7, (), [0.3, -0.4, 0.25, 0.1, -0.05, 0.7, -1.1]),
        )
        for (rt_model, rt_args, lit_model, lit_args, params) in cases
            @test logjoint(rt_model, params, rt_args, rdl_cons) ==
                  logjoint(lit_model, params, lit_args, rdl_cons)
            @test logjoint_unconstrained(rt_model, params, rt_args, rdl_cons) ==
                  logjoint_unconstrained(lit_model, params, lit_args, rdl_cons)
            @test logjoint_gradient_unconstrained(rt_model, params, rt_args, rdl_cons) ==
                  logjoint_gradient_unconstrained(lit_model, params, lit_args, rdl_cons)
        end
        # the resolved dims and slot layout match the literal twin's
        @test U._resolve_runtime_dims(rdl_runtime, Set{U.Address}([(:obs,)]), (3,)) === (3,)
        layout = U._conditioned_parameter_layout(rdl_runtime, rdl_cons, (3,))
        @test U.parametercount(layout) == 3
        slot = only(layout.slots)
        @test slot.transform == VectorIdentityTransform(3)
    end

    @testset "twin equality: runtime-dim mvnormaldense with a matrix argument" begin
        L = [1.0 0.0 0.0; 0.3 0.9 0.0; -0.2 0.1 0.8]
        params = [0.3, -0.4, 0.25]
        @test logjoint(rdl_dense_runtime, params, (3, L), rdl_cons) ==
              logjoint(rdl_dense_literal, params, (L,), rdl_cons)
        @test logjoint_unconstrained(rdl_dense_runtime, params, (3, L), rdl_cons) ==
              logjoint_unconstrained(rdl_dense_literal, params, (L,), rdl_cons)
        @test logjoint_gradient_unconstrained(rdl_dense_runtime, params, (3, L), rdl_cons) ==
              logjoint_gradient_unconstrained(rdl_dense_literal, params, (L,), rdl_cons)
    end

    @testset "latent-VALUED mean with argument-computable length scores" begin
        # length comes from `n`, values from the latent `z`: legal, and equal
        # to the literal-length twin
        params = [0.4, 0.2, -0.3, 0.9]
        @test logjoint_unconstrained(rdl_latent_mean, params, (3,), rdl_cons) ==
              logjoint_unconstrained(rdl_latent_mean_lit, params, (), rdl_cons)
        @test logjoint_gradient_unconstrained(rdl_latent_mean, params, (3,), rdl_cons) ==
              logjoint_gradient_unconstrained(rdl_latent_mean_lit, params, (), rdl_cons)
    end

    @testset "fresh-vs-cached across n: n=3, n=5, n=3 in one session" begin
        p3 = [0.2, -0.1, 0.45]
        p5 = [0.2, -0.1, 0.45, 0.8, -0.6]
        first_3 = logjoint(rdl_cache_session, p3, (3,), rdl_cons)
        first_5 = logjoint(rdl_cache_session, p5, (5,), rdl_cons)
        again_3 = logjoint(rdl_cache_session, p3, (3,), rdl_cons)
        # cached results equal fresh evaluations on identically-shaped models
        @test again_3 == first_3
        @test first_3 == logjoint(rdl_cache_fresh3, p3, (3,), rdl_cons)
        @test first_5 == logjoint(rdl_cache_fresh5, p5, (5,), rdl_cons)
        # exactly one cache entry per (signature, dims): two entries, not three
        cache = rdl_cache_session.signature_cache[]
        @test cache isa Dict{U._SignatureCacheKey,U.ResolvedSignaturePlan}
        @test length(cache) == 2
        @test Set(key[2] for key in keys(cache)) == Set([(3,), (5,)])
        # re-resolving n=3 returns the identical memoized plan object
        @test U._resolve_signature_plan(rdl_cache_session, rdl_cons, (3,)) ===
              U._resolve_signature_plan(rdl_cache_session, rdl_cons, (3,))
    end

    @testset "generated-scorer identity per n" begin
        for (args, params) in (
            ((3,), [0.3, -0.4, 0.25]),
            ((7,), [0.3, -0.4, 0.25, 0.1, -0.05, 0.7, -1.1]),
        )
            r = rdl_gen_compare(rdl_runtime, args, rdl_cons, params)
            @test r.lu_g == r.lu_i
            @test r.gr_g == r.gr_i
            # mvnormal latents stay on the interpreter today (also true for
            # LITERAL-length mvnormal latents); flip this pin if the generated
            # scorer learns vector latents
            @test r.used == false
        end
    end

    @testset "sampler smoke: nuts at two n in one session, sane vs twin" begin
        res3 = nuts(
            rdl_runtime, (3,), rdl_cons;
            num_samples=300, num_warmup=300, rng=MersenneTwister(289),
        )
        res7 = nuts(
            rdl_runtime, (7,), rdl_cons;
            num_samples=300, num_warmup=300, rng=MersenneTwister(289),
        )
        d3, names3 = constrained_draws(res3)
        d7, names7 = constrained_draws(res7)
        @test size(d3, 1) == 3 && names3 == ["theta[1]", "theta[2]", "theta[3]"]
        @test size(d7, 1) == 7 && length(names7) == 7
        # scoring is bitwise identical to the literal twin, so the same-seed
        # NUTS run produces the identical chain
        lit3 = nuts(
            rdl_literal3, (), rdl_cons;
            num_samples=300, num_warmup=300, rng=MersenneTwister(289),
        )
        dlit, _ = constrained_draws(lit3)
        @test d3 == dlit
        # posterior sanity: sum(theta) ~ N(0, n) prior, obs ~ N(sum, 1) at 1.2
        # -> per-coordinate posterior mean y/(n+1)
        @test isapprox(sum(d3) / length(d3), 1.2 / 4; atol=0.25)
        @test isapprox(sum(d7) / length(d7), 1.2 / 8; atol=0.25)
    end

    @testset "generate/assess predate the slot machinery and still work" begin
        trace, _ = generate(rdl_runtime, (4,), choicemap((:obs, 1.0)); rng=MersenneTwister(11))
        @test length(trace[(:theta,)]) == 4
        theta = [0.1, 0.2, -0.1]
        score = assess(rdl_runtime, (3,), choicemap((:obs, 1.0), (:theta, theta)))
        expected =
            U.logpdf(mvnormal(zeros(3), ones(3)), theta) + U.logpdf(normal(sum(theta), 1.0), 1.0)
        @test isapprox(score, expected; atol=1e-12)
    end

    @testset "latent-dependent LENGTH is rejected, naming the address" begin
        for (model, args) in ((rdl_lengthdep_discrete, ()), (rdl_lengthdep_det, (3,)))
            err = try
                logjoint(model, zeros(4), args, rdl_cons)
                nothing
            catch caught
                caught
            end
            @test err isa ArgumentError
            @test occursin("model arguments alone", err.msg)
            @test occursin("(:theta,)", err.msg)
            @test occursin("issue #289", err.msg)
        end
    end

    @testset "batched mixed-n args are rejected with both dims named" begin
        err = try
            U._batched_signature_layout(rdl_runtime, rdl_cons, [(3,), (5,)])
            nothing
        catch caught
            caught
        end
        @test err isa DimensionMismatch
        @test occursin("(3,)", err.msg)
        @test occursin("(5,)", err.msg)
        @test occursin("issue #289", err.msg)
        # the public batched sampler rejects too
        @test_throws DimensionMismatch batched_nuts(
            rdl_runtime, [(3,), (5,)], rdl_cons;
            num_chains=2, num_samples=5, num_warmup=5,
        )
        # per-column args that AGREE resolve fine
        layout = U._batched_signature_layout(rdl_runtime, rdl_cons, [(3,), (3,)])
        @test U.parametercount(layout) == 3
    end

    @testset "twin equality: runtime-dim dirichlet == literal twin (n=3, n=4)" begin
        # logjoint takes the CONSTRAINED simplex value (length n); the
        # unconstrained forms take the (n-1)-wide SimplexTransform coordinates
        cases = (
            (([2.0, 3.0, 4.0],), rdl_dirichlet_lit3, [0.3, -0.2], [0.2, 0.3, 0.5]),
            (([2.0, 3.0, 4.0, 5.0],), rdl_dirichlet_lit4, [0.3, -0.2, 0.45], [0.1, 0.2, 0.3, 0.4]),
        )
        for (rt_args, lit_model, params, value) in cases
            @test logjoint(rdl_dirichlet_runtime, value, rt_args, rdl_cons) ==
                  logjoint(lit_model, value, (), rdl_cons)
            @test logjoint_unconstrained(rdl_dirichlet_runtime, params, rt_args, rdl_cons) ==
                  logjoint_unconstrained(lit_model, params, (), rdl_cons)
            @test logjoint_gradient_unconstrained(rdl_dirichlet_runtime, params, rt_args, rdl_cons) ==
                  logjoint_gradient_unconstrained(lit_model, params, (), rdl_cons)
        end
        # the dims entry is the VALUE length n; the slot is n-1 wide behind a
        # SimplexTransform(n), exactly the static dirichlet slot sizing
        n3_alpha = ([2.0, 3.0, 4.0],)
        @test U._resolve_runtime_dims(rdl_dirichlet_runtime, Set{U.Address}([(:obs,)]), n3_alpha) === (3,)
        layout = U._conditioned_parameter_layout(rdl_dirichlet_runtime, rdl_cons, n3_alpha)
        @test U.parametercount(layout) == 2
        @test U.parametervaluecount(layout) == 3
        @test only(layout.slots).transform == SimplexTransform(3)
    end

    @testset "simplex round trip through the signature-aware transforms" begin
        for (rt_args, params) in (
            (([2.0, 3.0, 4.0],), [0.3, -0.2]),
            (([2.0, 3.0, 4.0, 5.0],), [0.3, -0.2, 0.45]),
        )
            constrained = U.transform_to_constrained(rdl_dirichlet_runtime, params, rt_args, rdl_cons)
            @test length(constrained) == length(params) + 1
            @test all(constrained .> 0)
            @test isapprox(sum(constrained), 1.0; atol=1e-12)
            back = U.transform_to_unconstrained(rdl_dirichlet_runtime, constrained, rt_args, rdl_cons)
            @test isapprox(back, params; atol=1e-12)
        end
    end

    @testset "twin equality: runtime-dim mvstudentt == literal twin (n=3, n=7)" begin
        cases = (
            ((3,), rdl_mvt_lit3, [0.3, -0.4, 0.25]),
            ((7,), rdl_mvt_lit7, [0.3, -0.4, 0.25, 0.1, -0.05, 0.7, -1.1]),
        )
        for (rt_args, lit_model, params) in cases
            @test logjoint(rdl_mvt_runtime, params, rt_args, rdl_cons) ==
                  logjoint(lit_model, params, (), rdl_cons)
            @test logjoint_unconstrained(rdl_mvt_runtime, params, rt_args, rdl_cons) ==
                  logjoint_unconstrained(lit_model, params, (), rdl_cons)
            @test logjoint_gradient_unconstrained(rdl_mvt_runtime, params, rt_args, rdl_cons) ==
                  logjoint_gradient_unconstrained(lit_model, params, (), rdl_cons)
        end
        layout = U._conditioned_parameter_layout(rdl_mvt_runtime, rdl_cons, (3,))
        @test U.parametercount(layout) == 3
        @test only(layout.slots).transform == VectorIdentityTransform(3)
    end

    @testset "twin equality: runtime-dim mvstudenttdense with a matrix argument" begin
        L = [1.0 0.0 0.0; 0.3 0.9 0.0; -0.2 0.1 0.8]
        params = [0.3, -0.4, 0.25]
        @test logjoint(rdl_mvtd_runtime, params, (3, L), rdl_cons) ==
              logjoint(rdl_mvtd_literal, params, (L,), rdl_cons)
        @test logjoint_unconstrained(rdl_mvtd_runtime, params, (3, L), rdl_cons) ==
              logjoint_unconstrained(rdl_mvtd_literal, params, (L,), rdl_cons)
        @test logjoint_gradient_unconstrained(rdl_mvtd_runtime, params, (3, L), rdl_cons) ==
              logjoint_gradient_unconstrained(rdl_mvtd_literal, params, (L,), rdl_cons)
    end

    @testset "fresh-vs-cached across n: dirichlet and mvstudentt" begin
        # dirichlet: n=3, n=4, n=3 in one session (constrained simplex values)
        a3 = ([2.0, 3.0, 4.0],)
        a4 = ([2.0, 3.0, 4.0, 5.0],)
        p3 = [0.2, 0.3, 0.5]
        p4 = [0.1, 0.2, 0.3, 0.4]
        first_3 = logjoint(rdl_dirichlet_runtime, p3, a3, rdl_cons)
        first_4 = logjoint(rdl_dirichlet_runtime, p4, a4, rdl_cons)
        @test logjoint(rdl_dirichlet_runtime, p3, a3, rdl_cons) == first_3
        @test first_3 == logjoint(rdl_dirichlet_fresh3, p3, a3, rdl_cons)
        @test first_4 == logjoint(rdl_dirichlet_fresh4, p4, a4, rdl_cons)
        cache = rdl_dirichlet_runtime.signature_cache[]
        @test Set(key[2] for key in keys(cache)) == Set([(3,), (4,)])
        # mvstudentt: n=3, n=7, n=3 in one session
        q3 = [0.2, -0.1, 0.45]
        q7 = [0.2, -0.1, 0.45, 0.8, -0.6, 0.05, 1.3]
        mvt_3 = logjoint(rdl_mvt_runtime, q3, (3,), rdl_cons)
        mvt_7 = logjoint(rdl_mvt_runtime, q7, (7,), rdl_cons)
        @test logjoint(rdl_mvt_runtime, q3, (3,), rdl_cons) == mvt_3
        @test mvt_3 == logjoint(rdl_mvt_fresh3, q3, (3,), rdl_cons)
        @test mvt_7 == logjoint(rdl_mvt_fresh7, q7, (7,), rdl_cons)
    end

    @testset "sampler smoke: dirichlet nuts at two n in one session" begin
        res3 = nuts(
            rdl_dirichlet_runtime, ([2.0, 3.0, 4.0],), rdl_cons;
            num_samples=300, num_warmup=300, rng=MersenneTwister(289),
        )
        res4 = nuts(
            rdl_dirichlet_runtime, ([2.0, 3.0, 4.0, 5.0],), rdl_cons;
            num_samples=300, num_warmup=300, rng=MersenneTwister(289),
        )
        d3, names3 = constrained_draws(res3)
        d4, names4 = constrained_draws(res4)
        @test size(d3, 1) == 3 && names3 == ["w[1]", "w[2]", "w[3]"]
        @test size(d4, 1) == 4 && length(names4) == 4
        # the SimplexTransform is exercised end to end: every draw is a simplex
        @test all(d3 .> 0) && all(d4 .> 0)
        @test all(abs.(vec(sum(d3; dims=1)) .- 1.0) .< 1e-10)
        @test all(abs.(vec(sum(d4; dims=1)) .- 1.0) .< 1e-10)
        # scoring is bitwise identical to the literal twin, so the same-seed
        # NUTS run produces the identical chain
        lit3 = nuts(
            rdl_dirichlet_lit3, (), rdl_cons;
            num_samples=300, num_warmup=300, rng=MersenneTwister(289),
        )
        dlit, _ = constrained_draws(lit3)
        @test d3 == dlit
    end

    @testset "dirichlet latent-dependent concentration LENGTH is rejected" begin
        err = try
            logjoint(rdl_dirichlet_lengthdep, zeros(3), (), rdl_cons)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("model arguments alone", err.msg)
        @test occursin("(:w,)", err.msg)
        @test occursin("issue #289", err.msg)
    end

    @testset "no candidate family remains on the pending error" begin
        # every family `_runtime_dim_size_expr` can emit as a candidate is in
        # the supported list, so the defensive resolution error is unreachable
        # for today's models
        for family in (:mvnormal, :mvnormaldense, :mvstudentt, :mvstudenttdense, :dirichlet)
            @test family in U._RUNTIME_DIM_SUPPORTED_FAMILIES
        end
    end

    @testset "wishart/inversewishart/lkjcholesky stay literal (macro-time pin)" begin
        # NOT runtime-dim candidates by design: a non-literal size hard-errors
        # at macro expansion (@eval wraps macro errors in LoadError)
        rdl_unwrap(err) = err isa LoadError ? rdl_unwrap(err.error) : err
        for (src, needle) in (
            (
                :(@tea static function rdl_wishart_bad(S)
                    Sigma ~ wishart(5.0, S)
                    {:obs} ~ normal(Sigma[1], 1.0)
                end),
                "wishart latents require a static square scale-matrix literal",
            ),
            (
                :(@tea static function rdl_invwishart_bad(S)
                    Sigma ~ inversewishart(5.0, S)
                    {:obs} ~ normal(Sigma[1], 1.0)
                end),
                "inversewishart latents require a static square scale-matrix literal",
            ),
            (
                :(@tea static function rdl_lkj_bad(d)
                    C ~ lkjcholesky(d, 2.0)
                    {:obs} ~ normal(C[1], 1.0)
                end),
                "lkjcholesky requires a literal integer dimension",
            ),
        )
            err = try
                @eval $src
                nothing
            catch caught
                rdl_unwrap(caught)
            end
            @test err isa ArgumentError
            @test occursin(needle, err.msg)
        end
    end

    @testset "backend lowering falls back to per-column ForwardDiff" begin
        resolved = U._resolve_signature_plan(rdl_runtime, rdl_cons, (3,))
        report = U._signature_backend_lowering(rdl_runtime, resolved).report
        @test !report.supported
        @test any(occursin("mvnormal", issue) for issue in report.issues)
        # PR-4 posture: the reason names the runtime-length choice and the
        # fallback instead of the generic tuple-lowering errors, on the
        # args-independent report and the signature-specific one alike
        for issues in (backend_report(rdl_runtime).issues, report.issues)
            reason = only(filter(issue -> occursin("issue #289", issue), issues))
            @test occursin("mvnormal", reason)
            @test occursin("(:theta,)", reason)
            @test occursin("runtime-length vector choice", reason)
            @test occursin("per-column ForwardDiff", reason)
            # the misleading per-argument errors are gone
            @test !any(occursin("unsupported call `zeros`", issue) for issue in issues)
            @test !any(occursin("tuple-like backend arguments", issue) for issue in issues)
        end
        @test !backend_report(rdl_runtime).supported
        # ... and the fallback gradient equals the single-path gradient
        params = hcat([0.3, -0.4, 0.25], [0.1, 0.0, -0.2])
        batched = batched_logjoint_gradient_unconstrained(rdl_runtime, params, (3,), rdl_cons)
        @test batched[:, 1] == logjoint_gradient_unconstrained(rdl_runtime, params[:, 1], (3,), rdl_cons)
        @test batched[:, 2] == logjoint_gradient_unconstrained(rdl_runtime, params[:, 2], (3,), rdl_cons)
        values = batched_logjoint_unconstrained(rdl_runtime, params, (3,), rdl_cons)
        @test values[1] == logjoint_unconstrained(rdl_runtime, params[:, 1], (3,), rdl_cons)
    end

    @testset "reverse tier: the probe declines and the column tier serves" begin
        params = hcat([0.3, -0.4, 0.25], [0.1, 0.0, -0.2])
        cache = U.BatchedLogjointGradientCache(rdl_runtime, params, (3,), rdl_cons)
        # no analytic backend plan, no flat cache: per-column ForwardDiff tier
        @test isnothing(cache.backend_cache)
        @test isnothing(cache.flat_cache)
        @test length(cache.column_caches) == 2
        # the reverse probe declines (the model is not on the generated-scorer
        # path; without Enzyme it declines even earlier) -- either way the
        # forward tiers stay in charge
        @test isnothing(cache.reverse_cache)
    end

    @testset "device lowering reports unsupported with a clear message" begin
        # device lowering consumes the backend plan, so the runtime-length
        # reason (and the honest device verdict inside it) surfaces here too;
        # the DeviceDiagNormalChoiceStepDyn generalization stays out of scope
        # until a whole-vector backend step exists (issue #289, PR-4)
        ok, issues = device_lowering_report(rdl_runtime; args=(3,), constraints=rdl_cons)
        @test !ok
        @test !isempty(issues)
        reason = only(filter(issue -> occursin("issue #289", issue), issues))
        @test occursin("mvnormal", reason)
        @test occursin("(:theta,)", reason)
        @test occursin("runtime-length vector choice", reason)
        @test occursin("device path is unsupported", reason)
    end

    @testset "args-independent layout APIs throw the informative error" begin
        for thrown in (
            (() -> U.parameterlayout(rdl_runtime)),
            (() -> U.transform_to_constrained(rdl_runtime, zeros(3), (3,))),
            (() -> U.transform_to_unconstrained(rdl_runtime, zeros(3), (3,))),
            (() -> U.parameterchoicemap(rdl_runtime, zeros(3))),
        )
            err = try
                thrown()
                nothing
            catch caught
                caught
            end
            @test err isa ArgumentError
            @test occursin("issue #289", err.msg)
            @test occursin("signature-aware", err.msg)
        end
        # the signature-aware 4-argument transforms keep working
        constrained = U.transform_to_constrained(rdl_runtime, [0.3, -0.4, 0.25], (3,), rdl_cons)
        @test constrained == [0.3, -0.4, 0.25]
    end

    @testset "GP teaser: runtime-length latent function values" begin
        xs = [0.0, 0.5, 1.0]
        params = [0.3, -0.4, 0.25]
        @test logjoint(rdl_gp_runtime, params, (xs, 3), rdl_cons) ==
              logjoint(rdl_gp_literal, params, (xs,), rdl_cons)
        @test logjoint_gradient_unconstrained(rdl_gp_runtime, params, (xs, 3), rdl_cons) ==
              logjoint_gradient_unconstrained(rdl_gp_literal, params, (xs,), rdl_cons)
    end
end
