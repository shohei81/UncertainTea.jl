# Family-generic vectorized observations (issue #287): `{:y} ~ family.(args...)`
# for poisson / bernoulli / bernoullilogit / exponential / studentt — the
# bread-and-butter GLM likelihoods — now lower to a backend-native dense
# per-element step (BackendBroadcastScalarChoicePlanStep) with an analytic
# batched gradient built from the issue-#285 single-source kernels/partials,
# instead of being rejected at macro time. Dotted FUNCTION calls in broadcast
# arguments (`exp.(a .+ b .* x)`) and in compiled deterministic expressions are
# supported alongside the dotted operators.

using KernelAbstractions: CPU

bsf_x = collect(range(-1.0, 1.0; length=8))

@tea static function bsf_pois_glm(x, n)
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    {:y} ~ poisson.(exp.(a .+ b .* x))
    return a
end

@tea static function bsf_pois_loop(x, n)
    a ~ normal(0.0, 1.0)
    b ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ poisson(exp(a + b * x[i]))
    end
    return a
end

@tea static function bsf_blogit(x, n)
    b ~ normal(0.0, 1.0)
    {:y} ~ bernoullilogit.(b .* x)
    return b
end

@tea static function bsf_bern(x, n)
    b ~ normal(0.0, 1.0)
    {:y} ~ bernoulli.(clamp.(0.5 .+ 0.2 .* b .* x, 0.01, 0.99))
    return b
end

@tea static function bsf_expo(x, n)
    logr ~ normal(0.0, 0.5)
    {:y} ~ exponential.(exp.(logr .+ 0.3 .* x))
    return logr
end

@tea static function bsf_tdist(x, n)
    m ~ normal(0.0, 1.0)
    {:y} ~ studentt.(4.0, m .+ x, 0.8)
    return m
end

@testset "broadcast_scalar_families" begin
    bsf_ys = Float64[2, 1, 3, 2, 4, 5, 3, 6]
    bsf_cm = choicemap((:y, bsf_ys))

    @testset "broadcast GLMs are backend-native" begin
        for model in (bsf_pois_glm, bsf_blogit, bsf_bern, bsf_expo, bsf_tdist)
            @test UncertainTea.backend_report(model).supported
        end
    end

    @testset "broadcast logjoint matches the loop-addressed equivalent" begin
        bsf_cm_loop = choicemap([(:y => i, bsf_ys[i]) for i = 1:8])
        lb = logjoint(bsf_pois_glm, [0.3, 0.5], (bsf_x, 8), bsf_cm)
        ll = logjoint(bsf_pois_loop, [0.3, 0.5], (bsf_x, 8), bsf_cm_loop)
        @test lb ≈ ll rtol = 1e-12
        # off-support element: a non-integer count scores -Inf on both spellings
        bad = copy(bsf_ys)
        bad[3] = 2.5
        @test logjoint(bsf_pois_glm, [0.3, 0.5], (bsf_x, 8), choicemap((:y, bad))) == -Inf
    end

    @testset "analytic batched gradients match ForwardDiff" begin
        cases = (
            (bsf_pois_glm, [0.3, 0.5], bsf_cm),
            (bsf_pois_glm, [-0.2, 0.8], bsf_cm),
            (bsf_blogit, [0.4], choicemap((:y, Float64[0, 1, 0, 1, 1, 0, 1, 1]))),
            (bsf_bern, [0.6], choicemap((:y, Float64[0, 1, 0, 1, 1, 0, 1, 1]))),
            (bsf_expo, [0.2], choicemap((:y, Float64[0.5, 1.2, 0.3, 2.0, 0.8, 1.5, 0.9, 0.4]))),
            (bsf_tdist, [0.3], choicemap((:y, Float64[-0.5, 0.2, 0.8, -1.2, 1.5, 0.1, -0.3, 2.0]))),
        )
        for (model, theta, cm) in cases
            gb = batched_logjoint_gradient_unconstrained(
                model, reshape(theta, length(theta), 1), (bsf_x, 8), cm,
            )
            gref = logjoint_gradient_unconstrained(model, theta, (bsf_x, 8), cm)
            @test isapprox(gb[:, 1], gref; rtol=1e-9, atol=1e-12)
        end
    end

    @testset "device lowering reports honestly unsupported" begin
        report_ok, _ = device_lowering_report(bsf_pois_glm)
        @test !report_ok
    end

    @testset "unsupported broadcast family is rejected at macro time" begin
        @test_throws LoadError @eval @tea static function bsf_bad(x)
            b ~ normal(0.0, 1.0)
            {:y} ~ gamma.(b .* x, 1.0)
            return b
        end
    end

    @testset "BroadcastScalarDist rand + logpdf direct" begin
        dist = UncertainTea.BroadcastScalarDist{:poisson}(exp.(0.3 .+ 0.5 .* bsf_x))
        sample = rand(MersenneTwister(1), dist)
        @test length(sample) == 8
        @test all(s -> s >= 0 && isinteger(float(s)), sample)
        @test isfinite(UncertainTea.logpdf(dist, bsf_ys))
        # dimension mismatch rejected
        @test_throws DimensionMismatch UncertainTea.logpdf(dist, bsf_ys[1:5])
    end

    @testset "NUTS runs on the broadcast Poisson GLM" begin
        chain = batched_nuts(
            bsf_pois_glm, (bsf_x, 8), bsf_cm;
            num_chains=2, num_samples=100, num_warmup=100, rng=MersenneTwister(1),
        )
        @test all(isfinite, posterior_array(chain))
    end
end
