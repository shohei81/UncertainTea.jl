# Issue #138: sufficient-statistics fusion for the single-chain generated scorer.
#
# The #144 generated scorer already replaces the Any-boxed interpreter for the
# single-chain scoring entry points. This extends it (src/generated_scorer.jl)
# so a fusable observation loop -- a single staged observed choice of an
# exponential-family family (normal / exponential / poisson) with LOOP-INVARIANT
# parameter expressions -- is scored in O(1) from statistics of the observation
# data (computed once at obs-build time) instead of the O(observations)
# per-iteration logpdf loop, mirroring the batched #146 fusion.
#
# These are in-process before/after comparisons. `_GEN_SCORER_SUFFSTATS[]`
# toggles the fused closed form off (the #144 per-observation loop) and on (the
# fused O(1) form) WITHOUT re-emitting the scorer; `_USE_GENERATED_SCORER[]`
# toggles the whole generated path off (the interpreter). No hardcoded goldens
# -- the interpreter (and the unfused generated loop) are the references. The
# critical numerics test is the |ybar| >> sigma cancellation regime, which the
# CENTERED statistics form must hold to rtol 1e-10 (a naive power-sum form would
# not).

const gss_UT = UncertainTea

@tea static function gss_gauss(n)
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea static function gss_expo(n)
    lr ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ exponential(exp(lr))
    end
    return lr
end

@tea static function gss_pois(n)
    ll ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ poisson(exp(ll))
    end
    return ll
end

# A loop whose parameter reads the iterator (mu = theta * i): NOT fusable, stays
# on the per-observation loop even with statistics turned on.
@tea static function gss_varying(n)
    slope ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(slope * i, sigma)
    end
    return slope
end

# Run each scoring entry point in three modes -- interpreter, generated-unfused
# (per-observation loop), generated-fused (O(1) statistics) -- for the same
# model/params. `logjoint` scores the constrained image of `params` so the
# distribution arguments stay valid.
function gss_modes(model, args, cons, params; reject::Bool=false)
    prev_gen = gss_UT._USE_GENERATED_SCORER[]
    prev_ss = gss_UT._GEN_SCORER_SUFFSTATS[]
    try
        constrained = transform_to_constrained(model, params, args, cons)

        gss_UT._USE_GENERATED_SCORER[] = false
        lj_i = logjoint(model, constrained, args, cons)
        lu_i = logjoint_unconstrained(model, params, args, cons; reject_invalid_parameters=reject)
        gc_i = gss_UT._logjoint_gradient_cache(model, params, args, cons; reject_invalid_parameters=reject)
        gr_i = copy(gss_UT._logjoint_gradient!(gc_i, params))
        pg_i = reject ? gr_i : logjoint_gradient_unconstrained(model, params, args, cons)

        gss_UT._USE_GENERATED_SCORER[] = true
        gss_UT._GEN_SCORER_SUFFSTATS[] = false
        lj_u = logjoint(model, constrained, args, cons)
        used = gss_UT._GEN_SCORER_LAST_USED[]
        lu_u = logjoint_unconstrained(model, params, args, cons; reject_invalid_parameters=reject)
        gc_u = gss_UT._logjoint_gradient_cache(model, params, args, cons; reject_invalid_parameters=reject)
        gr_u = copy(gss_UT._logjoint_gradient!(gc_u, params))
        pg_u = reject ? gr_u : logjoint_gradient_unconstrained(model, params, args, cons)

        gss_UT._GEN_SCORER_SUFFSTATS[] = true
        lj_f = logjoint(model, constrained, args, cons)
        lu_f = logjoint_unconstrained(model, params, args, cons; reject_invalid_parameters=reject)
        gc_f = gss_UT._logjoint_gradient_cache(model, params, args, cons; reject_invalid_parameters=reject)
        gr_f = copy(gss_UT._logjoint_gradient!(gc_f, params))
        pg_f = reject ? gr_f : logjoint_gradient_unconstrained(model, params, args, cons)

        return (;
            lj_i, lj_u, lj_f, lu_i, lu_u, lu_f, gr_i, gr_u, gr_f, pg_i, pg_u, pg_f, used,
        )
    finally
        gss_UT._USE_GENERATED_SCORER[] = prev_gen
        gss_UT._GEN_SCORER_SUFFSTATS[] = prev_ss
    end
end

@testset "generated scorer sufficient-statistics fusion (issue #138)" begin
    gss_rng = MersenneTwister(138)

    @testset "fusable loops are classified per family" begin
        # the fusion map records which stage gets the O(1) closed form
        gauss_cons = choicemap(((:y => i, randn(gss_rng)) for i = 1:8)...)
        expo_cons = choicemap(((:y => i, abs(randn(gss_rng))) for i = 1:8)...)
        pois_cons = choicemap(((:y => i, float(rand(gss_rng, 0:6))) for i = 1:8)...)
        vary_cons = choicemap(((:y => i, randn(gss_rng)) for i = 1:8)...)
        @test gss_UT._gen_stage_fusion(gss_UT._resolve_signature_plan(gss_gauss, gauss_cons).compiled) == [:normal]
        @test gss_UT._gen_stage_fusion(gss_UT._resolve_signature_plan(gss_expo, expo_cons).compiled) ==
              [:exponential]
        @test gss_UT._gen_stage_fusion(gss_UT._resolve_signature_plan(gss_pois, pois_cons).compiled) == [:poisson]
        # loop-varying mu (reads the iterator): not fusable
        @test gss_UT._gen_stage_fusion(gss_UT._resolve_signature_plan(gss_varying, vary_cons).compiled) == [:none]
    end

    @testset "fused == unfused == interpreter to rtol 1e-10" begin
        n = 1000
        gy = randn(gss_rng, n) .* 1.3 .+ 0.4
        gauss_cons = choicemap(((:y => i, gy[i]) for i = 1:n)...)
        ey = abs.(randn(gss_rng, n)) .+ 0.05
        expo_cons = choicemap(((:y => i, ey[i]) for i = 1:n)...)
        py = float.(rand(gss_rng, 0:8, n))
        pois_cons = choicemap(((:y => i, py[i]) for i = 1:n)...)

        cases = (
            ("gauss", gss_gauss, (n,), gauss_cons, [0.13, 0.22]),
            ("exponential", gss_expo, (n,), expo_cons, [0.35]),
            ("poisson", gss_pois, (n,), pois_cons, [0.6]),
        )
        for (name, model, args, cons, params) in cases, reject in (false, true)
            r = gss_modes(model, args, cons, params; reject=reject)
            @test r.used
            # fused agrees with the interpreter (the source of truth)
            @test isapprox(r.lj_f, r.lj_i; rtol=1e-10)
            @test isapprox(r.lu_f, r.lu_i; rtol=1e-10)
            @test isapprox(r.gr_f, r.gr_i; rtol=1e-10)
            @test isapprox(r.pg_f, r.pg_i; rtol=1e-10)
            # fused agrees with the unfused generated loop (same emitted scorer,
            # closed form vs per-observation reduction)
            @test isapprox(r.lj_f, r.lj_u; rtol=1e-10)
            @test isapprox(r.lu_f, r.lu_u; rtol=1e-10)
            @test isapprox(r.gr_f, r.gr_u; rtol=1e-10)
            @test isapprox(r.pg_f, r.pg_u; rtol=1e-10)
        end
    end

    @testset "centered form holds the |ybar| >> sigma cancellation regime" begin
        # ybar ~ 1e6, sigma = 1: the naive sum(y^2) - 2*mu*sum(y) + n*mu^2 form
        # cancels catastrophically; the centered sum((y - ybar)^2) form does not.
        n = 1000
        cy = randn(MersenneTwister(861), n) .* 1.0 .+ 1.0e6
        cons = choicemap(((:y => i, cy[i]) for i = 1:n)...)
        params = [1.0e6, 0.0]  # mu = 1e6, log sigma = 0 -> sigma = 1
        r = gss_modes(gss_gauss, (n,), cons, params)
        @test r.used
        @test isapprox(r.lj_f, r.lj_i; rtol=1e-10)
        @test isapprox(r.lu_f, r.lu_i; rtol=1e-10)
        @test isapprox(r.gr_f, r.gr_i; rtol=1e-10)
        # sanity: the log density really is dominated by the cancellation scale
        @test r.lj_i < -1.0e5
    end

    @testset "data the closed form cannot represent falls back per-observation" begin
        # a non-finite normal value, a negative exponential value, and a
        # non-count poisson value each disqualify the fused form (empty stats);
        # the emitted per-observation loop still runs and matches the interpreter
        n = 40
        gy = randn(MersenneTwister(4), n)
        gy[7] = Inf                     # normal: non-finite
        gauss_cons = choicemap(((:y => i, gy[i]) for i = 1:n)...)
        @test isempty(gss_UT._gen_normal_stats(gy))
        r = gss_modes(gss_gauss, (n,), gauss_cons, [0.1, 0.2]; reject=true)
        @test r.used
        @test r.lu_f == r.lu_i        # both -Inf
        @test isapprox(r.gr_f, r.gr_i; rtol=1e-10, nans=true)

        ey = abs.(randn(MersenneTwister(5), n)) .+ 0.1
        ey[3] = -0.5                    # exponential: negative
        expo_cons = choicemap(((:y => i, ey[i]) for i = 1:n)...)
        @test isempty(gss_UT._gen_exponential_stats(ey))
        re = gss_modes(gss_expo, (n,), expo_cons, [0.2]; reject=true)
        @test re.lu_f == re.lu_i

        py = float.(rand(MersenneTwister(6), 0:5, n))
        py[9] = 2.5                     # poisson: non-count
        pois_cons = choicemap(((:y => i, py[i]) for i = 1:n)...)
        @test isempty(gss_UT._gen_poisson_stats(py))
        rp = gss_modes(gss_pois, (n,), pois_cons, [0.4]; reject=true)
        @test rp.lu_f == rp.lu_i
    end

    @testset "non-fusable loop-varying parameters stay correct" begin
        n = 300
        vy = randn(MersenneTwister(7), n)
        cons = choicemap(((:y => i, vy[i]) for i = 1:n)...)
        r = gss_modes(gss_varying, (n,), cons, [0.5, 0.1])
        @test r.used
        @test isapprox(r.lj_f, r.lj_i; rtol=1e-10)
        @test isapprox(r.gr_f, r.gr_i; rtol=1e-10)
    end

    @testset "in-place constraint mutation refreshes statistics (gibbs path)" begin
        # the reused gradient cache re-stages its dense obs AND recomputes the
        # sufficient statistics when the constraints object is mutated in place
        n = 200
        y = randn(MersenneTwister(8), n) .* 1.1 .+ 0.3
        cons = choicemap(((:y => i, y[i]) for i = 1:n)...)
        params = [0.2, 0.15]
        prev = gss_UT._GEN_SCORER_SUFFSTATS[]
        try
            gss_UT._GEN_SCORER_SUFFSTATS[] = true
            cache = gss_UT._logjoint_gradient_cache(gss_gauss, params, (n,), cons)
            g0 = copy(gss_UT._logjoint_gradient!(cache, params))
            @test cache.objective isa gss_UT._GenGradientObjective
            # mutate several observations in place (bumps the mutation counter)
            for i in (5, 50, 123)
                gss_UT._pushchoice!(cons, :y => i, y[i] + 3.0)
            end
            g1 = copy(gss_UT._logjoint_gradient!(cache, params))
            # ground truth: a fresh interpreter gradient on the mutated data
            gss_UT._USE_GENERATED_SCORER[] = false
            ref = logjoint_gradient_unconstrained(gss_gauss, params, (n,), cons)
            gss_UT._USE_GENERATED_SCORER[] = true
            @test !isapprox(g1, g0; rtol=1e-6)          # the mutation changed the gradient
            @test isapprox(g1, ref; rtol=1e-10)          # and the fused cache tracked it
        finally
            gss_UT._GEN_SCORER_SUFFSTATS[] = prev
            gss_UT._USE_GENERATED_SCORER[] = true
        end
    end
end
