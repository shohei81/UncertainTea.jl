# Tests for the Enzyme reverse-mode extension `reverse_mode_gradient` (issue #268).
#
# NOT part of the package test target (Enzyme is heavy and version-sensitive),
# but covered by the dedicated .github/workflows/enzyme.yml workflow: weekly
# schedule + workflow_dispatch + pull_request when the fragile contract files
# (ext/UncertainTeaEnzymeExt.jl, test/enzyme/**, src/generated_scorer.jl)
# change (issue #333). See test/enzyme/Project.toml for local setup.
#
# Loading Enzyme activates UncertainTeaEnzymeExt, which supplies the
# `reverse_mode_gradient` method. Every case checks reverse-mode against the
# authoritative ForwardDiff gradient.

using Random, Test, LinearAlgebra
using UncertainTea
using UncertainTea.Inference
using UncertainTea.Diagnostics
const UT = UncertainTea
using ForwardDiff
using Enzyme   # activates UncertainTeaEnzymeExt

@testset "UncertainTeaEnzymeExt reverse_mode_gradient" begin
    @testset "GP hyperparameter gradient matches ForwardDiff" begin
        rng = MersenneTwister(262)
        n = 30
        X = reshape(sort(rand(rng, n) .* 5), 1, n)
        Ktrue = exp.(-0.5 .* [(X[1, i] - X[1, j])^2 for i = 1:n, j = 1:n]) + 0.04 .* Matrix(I, n, n)
        y = cholesky(Symmetric(Ktrue)).L * randn(rng, n)
        gp_nlml(h) = UT.logpdf(gaussianprocess(X, exp(h[1]), exp(h[2]), exp(h[3])), y)

        h0 = [0.0, 0.0, -1.0]
        fd = ForwardDiff.gradient(gp_nlml, h0)
        rev = reverse_mode_gradient(gp_nlml, h0)
        @test rev ≈ fd rtol = 1e-8
        @test length(rev) == 3
    end

    @testset "ARD-GP hyperparameter gradient (D+2 params) matches ForwardDiff" begin
        # the reverse-mode payoff for GP is the many-hyperparameter ARD kernel:
        # D per-dimension lengthscales + variance + noise (issue #273).
        rng = MersenneTwister(273)
        D, N = 16, 40
        X = randn(rng, D, N)
        y = randn(rng, N)
        # hyper = [log-lengthscales (D); log-variance; log-noise]
        ard_nlml(h) = UT.logpdf(gaussianprocess(X, exp.(h[1:D]), exp(h[D+1]), exp(h[D+2])), y)
        h0 = vcat(zeros(D), 0.0, -1.0)
        fd = ForwardDiff.gradient(ard_nlml, h0)
        rev = reverse_mode_gradient(ard_nlml, h0)
        @test rev ≈ fd rtol = 1e-6
        @test length(rev) == D + 2
    end

    @testset "pure high-P coupled logjoint matches ForwardDiff" begin
        # normal(0,1) prior + per-element nonlinear neighbour coupling; nothing
        # sufficient-statistics-fuses, the class where reverse-mode scales O(P)
        # against ForwardDiff's O(P^2) (see bench/reverse_mode/).
        function coupled_logjoint(x, y)
            T = eltype(x)
            c = T(0.9189385332046727)
            lp = zero(T)
            @inbounds for i in eachindex(x)
                lp += -x[i]^2 / 2 - c
            end
            @inbounds for i = 1:(length(x)-1)
                z = (y[i] - (tanh(x[i]) + T(0.5) * x[i+1])) / T(0.3)
                lp += -z^2 / 2 - log(T(0.3)) - c
            end
            return lp
        end

        rng = MersenneTwister(268)
        P = 100
        x = randn(rng, P)
        y = [tanh(x[i]) + 0.5 * x[i+1] + 0.3 * randn(rng) for i = 1:(P-1)]
        obj(u) = coupled_logjoint(u, y)
        fd = ForwardDiff.gradient(obj, x)
        rev = reverse_mode_gradient(obj, x)
        @test rev ≈ fd rtol = 1e-8
        @test length(rev) == P
    end

    @testset "matches a closed-form gradient" begin
        # f(x) = -||x - a||^2 / 2  =>  grad = a - x
        a = [1.0, -2.0, 0.5, 3.0]
        f(x) = -sum((x .- a) .^ 2) / 2
        x0 = [0.2, 0.1, -0.3, 1.5]
        @test reverse_mode_gradient(f, x0) ≈ (a .- x0) rtol = 1e-10
    end

    @testset "model-level reverse_mode_gradient matches forward-mode" begin
        # A non-analytic coupled model on the type-stable generated-scorer path
        # (issue #268, part A): the model-level reverse-mode gradient must equal
        # the forward-mode logjoint_gradient_unconstrained exactly.
        @tea static function coupled_rev()
            x ~ iid(normal(0.0, 1.0), 12)
            for i = 1:11
                {:y => i} ~ normal(tanh(x[i]) + 0.5 * x[i+1], 0.3)
            end
            return x
        end
        rng = MersenneTwister(268)
        xt = randn(rng, 12)
        cm = UT.choicemap([(:y => i, tanh(xt[i]) + 0.5 * xt[i+1] + 0.3 * randn(rng)) for i = 1:11])
        theta = randn(rng, 12)

        fwd = logjoint_gradient_unconstrained(coupled_rev, theta, (), cm)
        rev = reverse_mode_gradient(coupled_rev, theta, (), cm)
        @test rev ≈ fwd rtol = 1e-8
        @test length(rev) == 12

        # a different position still agrees
        theta2 = randn(MersenneTwister(7), 12)
        @test reverse_mode_gradient(coupled_rev, theta2, (), cm) ≈
              logjoint_gradient_unconstrained(coupled_rev, theta2, (), cm) rtol = 1e-8
    end

    @testset "static-vector-obs models: reverse matches forward (issue #288)" begin
        # GP / broadcast-GLM / HMM hyperparameter models -- the reverse-mode
        # killer cases -- now take the generated-scorer path via static
        # whole-vector observation staging, so the model-level reverse gradient
        # works where it previously threw ArgumentError.
        @tea static function svo_gp_rev(X)
            logl ~ normal(0.0, 1.0)
            logv ~ normal(0.0, 1.0)
            logn ~ normal(-1.0, 1.0)
            {:y} ~ gaussianprocess(X, exp(logl), exp(logv), exp(logn))
            return logl
        end
        rng288 = MersenneTwister(288)
        Xg = reshape(sort(rand(rng288, 16) .* 5), 1, 16)
        cm_g = UT.choicemap((:y, randn(rng288, 16)))
        fg = logjoint_gradient_unconstrained(svo_gp_rev, [0.1, 0.2, -0.8], (Xg,), cm_g)
        rg = reverse_mode_gradient(svo_gp_rev, [0.1, 0.2, -0.8], (Xg,), cm_g)
        @test rg ≈ fg rtol = 1e-8

        @tea static function svo_glm_rev(x, n)
            a ~ normal(0.0, 1.0)
            b ~ normal(0.0, 1.0)
            {:y} ~ poisson.(exp.(a .+ b .* x))
            return a
        end
        xg = collect(range(-1.0, 1.0; length=8))
        cm_p = UT.choicemap((:y, Float64[2, 1, 3, 2, 4, 5, 3, 6]))
        fp = logjoint_gradient_unconstrained(svo_glm_rev, [0.3, 0.5], (xg, 8), cm_p)
        rp = reverse_mode_gradient(svo_glm_rev, [0.3, 0.5], (xg, 8), cm_p)
        @test rp ≈ fp rtol = 1e-8

        @tea static function svo_hmm_rev(init, trans)
            m1 ~ normal(-1.0, 2.0)
            log_gap ~ normal(0.0, 1.0)
            logs ~ normal(-0.5, 0.5)
            {:y} ~ hmm(init, trans, [m1, m1 + exp(log_gap)], exp(logs))
            return m1
        end
        cm_h = UT.choicemap((:y, randn(MersenneTwister(3), 30)))
        fh = logjoint_gradient_unconstrained(svo_hmm_rev, [0.1, 0.2, -0.3], ([0.6, 0.4], [0.8 0.2; 0.3 0.7]), cm_h)
        rh = reverse_mode_gradient(svo_hmm_rev, [0.1, 0.2, -0.3], ([0.6, 0.4], [0.8 0.2; 0.3 0.7]), cm_h)
        @test rh ≈ fh rtol = 1e-8
    end

    @testset "interpreter-fallback model is rejected with a clear message" begin
        # a scalar (non-loop) observation falls off the generated-scorer path;
        # reverse-mode must reject it rather than silently take a slow route.
        @tea static function scalar_obs()
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu, 1.0)
            return mu
        end
        cm = UT.choicemap((:y, 0.7))
        @test_throws ArgumentError reverse_mode_gradient(scalar_obs, [0.3], (), cm)
    end

    # --- batched per-column reverse-mode tier (issue #268, part A2) -----------

    @tea static function batched_big()
        x ~ iid(normal(0.0, 1.0), 40)
        for i = 1:39
            {:y => i} ~ normal(tanh(x[i]) + 0.5 * x[i+1], 0.3)
        end
        return x
    end

    function batched_big_constraints(seed)
        rng = MersenneTwister(seed)
        xt = randn(rng, 40)
        return UT.choicemap([(:y => i, tanh(xt[i]) + 0.5 * xt[i+1] + 0.3 * randn(rng)) for i = 1:39])
    end

    @testset "batched reverse cache: adtype selection + threshold" begin
        cm = batched_big_constraints(1)
        params = randn(MersenneTwister(2), 40, 5)

        # :forward never engages reverse
        cf = UT.BatchedLogjointGradientCache(batched_big, params, (), cm; adtype=:forward)
        @test cf.reverse_cache === nothing
        # :auto engages reverse (40 >= threshold, Enzyme loaded, shared constraints)
        ca = UT.BatchedLogjointGradientCache(batched_big, params, (), cm; adtype=:auto)
        @test ca.reverse_cache !== nothing
        # :reverse forces it
        cr = UT.BatchedLogjointGradientCache(batched_big, params, (), cm; adtype=:reverse)
        @test cr.reverse_cache !== nothing

        # a below-threshold model stays forward under :auto but is forced under :reverse
        @tea static function batched_small()
            x ~ iid(normal(0.0, 1.0), 8)
            for i = 1:7
                {:y => i} ~ normal(tanh(x[i]) + 0.5 * x[i+1], 0.3)
            end
            return x
        end
        cm_s = UT.choicemap([(:y => i, 0.1 * i) for i = 1:7])
        p_s = randn(MersenneTwister(3), 8, 3)
        @test UT.BatchedLogjointGradientCache(batched_small, p_s, (), cm_s; adtype=:auto).reverse_cache === nothing
        @test UT.BatchedLogjointGradientCache(batched_small, p_s, (), cm_s; adtype=:reverse).reverse_cache !== nothing

        # threshold boundary (issue #277): :auto engages reverse at exactly 24
        # parameters and stays forward just below it.
        @tea static function coupled_at_threshold()
            x ~ iid(normal(0.0, 1.0), 24)
            for i = 1:23
                {:y => i} ~ normal(tanh(x[i]) + 0.5 * x[i+1], 0.3)
            end
            return x
        end
        @tea static function coupled_below_threshold()
            x ~ iid(normal(0.0, 1.0), 20)
            for i = 1:19
                {:y => i} ~ normal(tanh(x[i]) + 0.5 * x[i+1], 0.3)
            end
            return x
        end
        cm_24 = UT.choicemap([(:y => i, 0.05 * i) for i = 1:23])
        cm_20 = UT.choicemap([(:y => i, 0.05 * i) for i = 1:19])
        @test UT.BatchedLogjointGradientCache(coupled_at_threshold, randn(MersenneTwister(4), 24, 3), (), cm_24; adtype=:auto).reverse_cache !==
              nothing
        @test UT.BatchedLogjointGradientCache(
            coupled_below_threshold,
            randn(MersenneTwister(5), 20, 3),
            (),
            cm_20;
            adtype=:auto,
        ).reverse_cache === nothing

        @test_throws ArgumentError UT.BatchedLogjointGradientCache(batched_big, params, (), cm; adtype=:bogus)
    end

    @testset "batched reverse gradient + value match forward exactly" begin
        cm = batched_big_constraints(1)
        params = randn(MersenneTwister(2), 40, 5)

        fref = copy(
            UT.batched_logjoint_gradient_unconstrained!(
                UT.BatchedLogjointGradientCache(batched_big, params, (), cm; adtype=:forward), params,
            ),
        )
        vref = UT.batched_logjoint_unconstrained(batched_big, params, (), cm)

        cr = UT.BatchedLogjointGradientCache(batched_big, params, (), cm; adtype=:reverse)
        grev = copy(UT.batched_logjoint_gradient_unconstrained!(cr, params))
        @test grev ≈ fref rtol = 1e-8

        dest = zeros(5)
        d, g = UT._batched_logjoint_and_gradient_unconstrained!(dest, cr, params)
        @test g ≈ fref rtol = 1e-8
        @test d ≈ vref rtol = 1e-8
    end

    @testset "batched_nuts adtype=:reverse matches :forward distributionally" begin
        cm = batched_big_constraints(1)
        chf = batched_nuts(
            batched_big, (), cm; num_chains=6, num_samples=400, num_warmup=400,
            adtype=:forward, rng=MersenneTwister(42),
        )
        chr = batched_nuts(
            batched_big, (), cm; num_chains=6, num_samples=400, num_warmup=400,
            adtype=:reverse, rng=MersenneTwister(99),
        )
        af = UT.posterior_array(chf)
        ar = UT.posterior_array(chr)
        mf = vec(sum(af; dims=(1, 2))) ./ (size(af, 1) * size(af, 2))
        mr = vec(sum(ar; dims=(1, 2))) ./ (size(ar, 1) * size(ar, 2))
        # different seeds -> pure MC error; the invariant distribution is identical
        # (chaotic leapfrog divergence rules out a bitwise sample match).
        @test maximum(abs.(mf .- mr)) < 0.15
    end

    @testset "adtype on advi/svgd/smc engages reverse (issue #275)" begin
        cm = batched_big_constraints(1)

        # ADVI/SVGD gradient descent is deterministic and the reverse gradient
        # matches forward to ~1e-15, so with the same seed the results agree to
        # very high precision -- a strong check that reverse actually engaged.
        # (Not bitwise: the ~1e-15 gradient difference accumulates over the Adam
        # steps, so compare with a tight tolerance rather than `==`.)
        af = batched_advi(batched_big, (), cm; num_particles=16, num_iterations=150, adtype=:forward, rng=MersenneTwister(5))
        ar = batched_advi(batched_big, (), cm; num_particles=16, num_iterations=150, adtype=:reverse, rng=MersenneTwister(5))
        @test af.location ≈ ar.location rtol = 1e-8

        sf = batched_svgd(batched_big, (), cm; num_particles=16, num_iterations=80, adtype=:forward, rng=MersenneTwister(6))
        sr = batched_svgd(batched_big, (), cm; num_particles=16, num_iterations=80, adtype=:reverse, rng=MersenneTwister(6))
        @test sf.constrained_particles ≈ sr.constrained_particles rtol = 1e-8

        # SMC with a NUTS move kernel: reverse runs and produces the same tempering
        # schedule length as forward (a linear-gaussian model that converges).
        @tea static function smc_med()
            x ~ iid(normal(0.0, 1.0), 34)
            for i = 1:33
                {:y => i} ~ normal(0.7 * x[i] + 0.2 * x[i+1], 0.5)
            end
            return x
        end
        srng = MersenneTwister(1)
        sxt = randn(srng, 34)
        scm = UT.choicemap([(:y => i, 0.7 * sxt[i] + 0.2 * sxt[i+1] + 0.5 * randn(srng)) for i = 1:33])
        smf = batched_smc(
            smc_med,
            (),
            scm;
            num_particles=32,
            max_stages=200,
            move_kernel=:nuts,
            move_steps=3,
            adtype=:forward,
            rng=MersenneTwister(7),
        )
        smr = batched_smc(
            smc_med,
            (),
            scm;
            num_particles=32,
            max_stages=200,
            move_kernel=:nuts,
            move_steps=3,
            adtype=:reverse,
            rng=MersenneTwister(7),
        )
        @test length(smf.stages) == length(smr.stages)
        @test all(isfinite, smr.importance.constrained_particles)
    end
end
