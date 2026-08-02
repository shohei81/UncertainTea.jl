# Ordered logistic (cumulative-logit) distribution for ordinal outcomes
# (issue #291). CPU-reference only. Statistics is unavailable in the harness,
# so use local helpers.

using ForwardDiff: ForwardDiff

ordl_mean(x) = sum(x) / length(x)

@tea static function ordl_model(cutpoints, n)
    beta ~ normal(0.0, 2.0)
    for i = 1:n
        {:y => i} ~ orderedlogistic(beta * (i - 5.5) / 3, cutpoints)
    end
    return beta
end

@testset "dist_ordered_logistic" begin
    ordl_cut = [-1.0, 0.5, 2.0]

    @testset "probabilities are a normalized cumulative-logit partition" begin
        for eta in (-2.0, 0.0, 1.5)
            d = orderedlogistic(eta, ordl_cut)
            ps = [exp(UncertainTea.logpdf(d, k)) for k = 1:4]
            @test all(p -> 0 < p < 1, ps)
            @test sum(ps) ≈ 1.0 rtol = 1e-12
            # matches the direct sigmoid differences
            sig(z) = 1 / (1 + exp(-z))
            direct = [
                sig(ordl_cut[1] - eta),
                sig(ordl_cut[2] - eta) - sig(ordl_cut[1] - eta),
                sig(ordl_cut[3] - eta) - sig(ordl_cut[2] - eta),
                1 - sig(ordl_cut[3] - eta),
            ]
            @test isapprox(ps, direct; rtol=1e-10)
        end
        # a larger eta shifts mass to higher categories
        low = exp(UncertainTea.logpdf(orderedlogistic(-3.0, ordl_cut), 1))
        high = exp(UncertainTea.logpdf(orderedlogistic(3.0, ordl_cut), 1))
        @test low > high
    end

    @testset "tail accuracy vs BigFloat (issue #344)" begin
        # eta far below the cutpoints used to cancel catastrophically in the
        # middle categories (-Inf at eta <= -37, Inf gradient); compare every
        # regime against a 256-bit BigFloat reference.
        tail_cut = [0.0, 1.0]
        ref = setprecision(BigFloat, 256) do
            bsig(z) = inv(1 + exp(-z))
            function blogp(eta, k)
                e = BigFloat(eta)
                p = if k == 1
                    bsig(tail_cut[1] - e)
                elseif k == 3
                    1 - bsig(tail_cut[2] - e)
                else
                    bsig(tail_cut[2] - e) - bsig(tail_cut[1] - e)
                end
                return Float64(log(p))
            end
            Dict(
                (eta, k) => blogp(eta, k) for
                eta in (-50.0, -37.0, -36.0, -30.0, 0.0, 30.0, 37.0, 50.0), k in (1, 2, 3)
            )
        end
        for ((eta, k), truth) in ref
            got = UncertainTea.logpdf(orderedlogistic(eta, tail_cut), k)
            @test isfinite(got)
            @test isapprox(got, truth; rtol=1e-10, atol=1e-8)
        end

        # Gradient at the old blow-up point: finite and matching the analytic
        # derivative of log P(y = 2) computed in BigFloat.
        grad = ForwardDiff.derivative(
            e -> UncertainTea.logpdf(orderedlogistic(e, tail_cut), 2), -37.0,
        )
        grad_ref = setprecision(BigFloat, 256) do
            bsig(z) = inv(1 + exp(-z))
            bpdf(z) = bsig(z) * (1 - bsig(z))
            e = BigFloat(-37)
            lo, hi = tail_cut
            Float64((bpdf(lo - e) - bpdf(hi - e)) / (bsig(hi - e) - bsig(lo - e)))
        end
        @test isfinite(grad)
        @test isapprox(grad, grad_ref; rtol=1e-10, atol=1e-8)
    end

    @testset "support and validation" begin
        d = orderedlogistic(0.5, ordl_cut)
        @test UncertainTea.logpdf(d, 0) == -Inf
        @test UncertainTea.logpdf(d, 5) == -Inf
        @test UncertainTea.logpdf(d, 2.5) == -Inf
        @test isfinite(UncertainTea.logpdf(d, 4))
        @test_throws ArgumentError orderedlogistic(0.0, [1.0, 0.5])
        @test_throws ArgumentError orderedlogistic(0.0, Float64[])
    end

    @testset "rand draws valid categories with the right skew" begin
        rng = MersenneTwister(291)
        lo = [rand(rng, orderedlogistic(-3.0, ordl_cut)) for _ = 1:200]
        hi = [rand(rng, orderedlogistic(3.0, ordl_cut)) for _ = 1:200]
        @test all(k -> k in 1:4, lo) && all(k -> k in 1:4, hi)
        @test ordl_mean(lo) < ordl_mean(hi)
    end

    @testset "backend/device honestly unsupported (CPU-reference only)" begin
        @test !UncertainTea.backend_report(ordl_model).supported
        @test !device_lowering_report(ordl_model)[1]
    end

    @testset "NUTS recovers the ordinal regression slope" begin
        rng = MersenneTwister(7)
        n = 60
        beta_true = 1.4
        ys = Float64[rand(rng, orderedlogistic(beta_true * (i - 5.5) / 3, ordl_cut)) for i = 1:n]
        cm = choicemap([(:y => i, ys[i]) for i = 1:n])
        chain = nuts(ordl_model, (ordl_cut, n), cm; num_samples=400, num_warmup=400, rng=MersenneTwister(11))
        draws = chain.constrained_samples
        @test all(isfinite, draws)
        @test abs(ordl_mean(draws[1, :]) - beta_true) < 0.8
    end
end
