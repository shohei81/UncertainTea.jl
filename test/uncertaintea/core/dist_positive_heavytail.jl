# Positive-support / heavy-tail families (issue #230): pareto, frechet,
# rayleigh, inversegaussian. frechet/rayleigh/inversegaussian are elementary
# positive densities with CPU-reference, backend-native analytic gradient, and
# device (KernelAbstractions) support. pareto is CPU + backend obs-native: its
# latent unconstrains through a LowerBoundedTransform that is not a batched /
# device transform, so a latent pareto honestly falls back to the ForwardDiff
# column path (and the device reports it unsupported), mirroring uniform.
#
# No `using Statistics` (CI hygiene): local mean/std helpers below.
_pht_mean(v) = sum(v) / length(v)
function _pht_std(v)
    m = _pht_mean(v)
    return sqrt(sum(x -> (x - m)^2, v) / (length(v) - 1))
end

@testset "dist_positive_heavytail" begin
    lp = UncertainTea.logpdf

    @testset "logpdf closed forms" begin
        # pareto(x_m, alpha): log(alpha) + alpha log(x_m) - (alpha+1) log(x), x >= x_m
        let xm = 2.0, al = 3.0, x = 5.0
            @test lp(pareto(xm, al), x) ≈ log(al) + al * log(xm) - (al + 1) * log(x) atol = 1e-12
        end
        @test lp(pareto(2.0, 3.0), 1.9) == -Inf
        # frechet(shape, scale): log(a) - log(s) - (1+a) log(x/s) - (x/s)^(-a), x > 0
        let sh = 2.5, sc = 1.3, x = 0.8, logz = log(0.8) - log(1.3)
            @test lp(frechet(sh, sc), x) ≈ log(sh) - log(sc) - (1 + sh) * logz - exp(-sh * logz) atol = 1e-12
        end
        @test lp(frechet(2.0, 1.0), -0.1) == -Inf
        @test lp(frechet(2.0, 1.0), 0.0) == -Inf
        # rayleigh(scale): log(x) - 2 log(s) - x^2/(2 s^2), x > 0
        let s = 1.7, x = 2.3
            @test lp(rayleigh(s), x) ≈ log(x) - 2 * log(s) - x^2 / (2 * s^2) atol = 1e-12
        end
        @test lp(rayleigh(1.0), 0.0) == -Inf
        @test lp(rayleigh(1.0), -0.5) == -Inf
        # inversegaussian(mu, lambda): sqrt(lambda/(2 pi x^3)) exp(-lambda (x-mu)^2/(2 mu^2 x))
        let mu = 1.5, lam = 2.0, x = 0.9
            @test lp(inversegaussian(mu, lam), x) ≈
                  0.5 * log(lam) - 0.5 * log(2pi) - 1.5 * log(x) - lam * (x - mu)^2 / (2 * mu^2 * x) atol = 1e-12
        end
        @test lp(inversegaussian(1.0, 1.0), 0.0) == -Inf
        @test lp(inversegaussian(1.0, 1.0), -0.5) == -Inf
        # constructors reject out-of-support parameters
        @test_throws ArgumentError pareto(-1.0, 2.0)
        @test_throws ArgumentError pareto(1.0, -2.0)
        @test_throws ArgumentError frechet(0.0, 1.0)
        @test_throws ArgumentError frechet(1.0, 0.0)
        @test_throws ArgumentError rayleigh(0.0)
        @test_throws ArgumentError inversegaussian(0.0, 1.0)
        @test_throws ArgumentError inversegaussian(1.0, -1.0)
    end

    @testset "value-gradient matches ForwardDiff" begin
        for (d, xs) in (
            (pareto(2.0, 3.0), (2.5, 4.0, 6.0)),
            (frechet(2.5, 1.3), (0.4, 1.0, 2.5)),
            (rayleigh(1.7), (0.5, 1.5, 3.0)),
            (inversegaussian(1.5, 2.0), (0.6, 1.2, 2.5)),
        )
            for x in xs
                fd = UncertainTea.ForwardDiff.derivative(t -> lp(d, t), x)
                h = 1e-6
                cd = (lp(d, x + h) - lp(d, x - h)) / (2h)
                @test isapprox(fd, cd; rtol=1e-4, atol=1e-6)
            end
        end
    end

    @testset "rand support and finiteness" begin
        rng = MersenneTwister(230)
        for d in (pareto(2.0, 3.0), frechet(2.5, 1.3), rayleigh(1.7), inversegaussian(1.5, 2.0))
            draws = [rand(rng, d) for _ = 1:20000]
            @test all(isfinite, draws)
            @test all(>(0), draws)
        end
        @test all(>=(2.0), [rand(rng, pareto(2.0, 3.0)) for _ = 1:5000])
        # rayleigh(scale) has mean scale*sqrt(pi/2)
        ray = [rand(rng, rayleigh(2.0)) for _ = 1:20000]
        @test abs(_pht_mean(ray) - 2.0 * sqrt(pi / 2)) < 0.1
        # inversegaussian(mu, lambda) has mean mu
        ig = [rand(rng, inversegaussian(1.5, 4.0)) for _ = 1:20000]
        @test abs(_pht_mean(ig) - 1.5) < 0.1
    end

    # ---- latent-slot wiring: transforms + backend plan step types ----
    @tea static function pht_pareto_model()
        x ~ pareto(1.0, 3.0)
        {:y} ~ normal(x, 0.5)
        return x
    end
    @tea static function pht_frechet_model()
        x ~ frechet(2.0, 1.3)
        {:y} ~ normal(x, 0.5)
        return x
    end
    @tea static function pht_rayleigh_model()
        x ~ rayleigh(1.5)
        {:y} ~ normal(x, 0.5)
        return x
    end
    @tea static function pht_ig_model()
        x ~ inversegaussian(1.5, 2.0)
        {:y} ~ normal(x, 0.5)
        return x
    end
    # pareto observation with a dynamic (latent-flowing) lower bound is allowed.
    @tea static function pht_pareto_obs_model()
        w ~ normal(0.0, 0.3)
        {:y} ~ pareto(0.2 + 0.1 * exp(w), 3.0)
        return w
    end

    @testset "latent transforms and backend plan steps" begin
        @test modelspec(pht_pareto_model).parameter_layout.slots[1].transform isa UncertainTea.LowerBoundedTransform
        @test modelspec(pht_frechet_model).parameter_layout.slots[1].transform isa LogTransform
        @test modelspec(pht_rayleigh_model).parameter_layout.slots[1].transform isa LogTransform
        @test modelspec(pht_ig_model).parameter_layout.slots[1].transform isa LogTransform

        @test backend_execution_plan(pht_frechet_model).steps[1] isa UncertainTea.BackendFrechetChoicePlanStep
        @test backend_execution_plan(pht_rayleigh_model).steps[1] isa UncertainTea.BackendRayleighChoicePlanStep
        @test backend_execution_plan(pht_ig_model).steps[1] isa UncertainTea.BackendInverseGaussianChoicePlanStep
        @test backend_execution_plan(pht_pareto_obs_model).steps[2] isa UncertainTea.BackendParetoChoicePlanStep

        # a dynamic x_m is rejected for a LATENT pareto at macro-expansion time
        @test_throws Exception @eval @tea static function pht_pareto_dyn(a)
            x ~ pareto(a, 3.0)
            return x
        end
    end

    # ---- analytic batched gradient vs ForwardDiff, and device parity ----
    device_families = (
        ("frechet", pht_frechet_model),
        ("rayleigh", pht_rayleigh_model),
        ("inversegaussian", pht_ig_model),
    )

    @testset "backend analytic gradient == ForwardDiff" begin
        for (name, model) in (device_families..., ("pareto_obs", pht_pareto_obs_model))
            tr, _ = generate(model, (), choicemap((:y, 1.35)); rng=MersenneTwister(160))
            p = parameter_vector(tr)
            bp = reshape(hcat(p .- 0.1, p, p .+ 0.15), length(p), 3)
            bcm = [choicemap((:y, 0.8)), choicemap((:y, 1.35)), choicemap((:y, 1.9))]
            g = batched_logjoint_gradient_unconstrained(model, bp, (), bcm)
            gref = hcat(
                [
                    UncertainTea.ForwardDiff.gradient(v -> logjoint_unconstrained(model, v, (), bcm[i]), bp[:, i]) for i = 1:3
                ]...,
            )
            @test isapprox(g, gref; rtol=1e-7, atol=1e-7)
        end
        # frechet/rayleigh/inversegaussian are backend-native; a latent pareto
        # (lower-bounded transform) honestly falls back to the ForwardDiff column path.
        @test backend_report(pht_frechet_model).supported
        @test backend_report(pht_rayleigh_model).supported
        @test backend_report(pht_ig_model).supported
        @test backend_report(pht_pareto_obs_model).supported
        @test !backend_report(pht_pareto_model).supported
    end

    @testset "device lowering parity (CPU Float64)" begin
        for (name, model) in device_families
            supported, issues = device_lowering_report(model)
            @test supported
            @test isempty(issues)
            tr, _ = generate(model, (), choicemap((:y, 1.35)); rng=MersenneTwister(160))
            p = parameter_vector(tr)
            bp = reshape(hcat(p .- 0.1, p, p .+ 0.15), length(p), 3)
            bcm = [choicemap((:y, 0.8)), choicemap((:y, 1.35)), choicemap((:y, 1.9))]
            vref = batched_logjoint_unconstrained(model, bp, (), bcm)
            gref = batched_logjoint_gradient_unconstrained(model, bp, (), bcm)
            v, g = device_batched_logjoint_gradient(model, bp, (), bcm)
            @test isapprox(v, vref; rtol=1e-10, atol=1e-10)
            @test isapprox(g, gref; rtol=1e-10, atol=1e-10)
        end
        # pareto is deliberately not device-lowered (LowerBoundedTransform is not
        # a device transform); the report must say so honestly.
        supported_p, issues_p = device_lowering_report(pht_pareto_obs_model)
        @test !supported_p
        @test !isempty(issues_p)
    end

    @testset "NUTS posterior recovery" begin
        # rayleigh scale latent recovers a sane positive scale.
        @tea static function pht_ray_scale(n)
            s ~ halfnormal(2.0)
            for i = 1:n
                {:y => i} ~ rayleigh(s)
            end
            return s
        end
        rng = MersenneTwister(9)
        true_s = 1.4
        ys = [rand(rng, rayleigh(true_s)) for _ = 1:60]
        ray_cm = choicemap((:y => i, ys[i]) for i = 1:60)
        ray_chains = nuts_chains(
            pht_ray_scale,
            (60,),
            ray_cm;
            num_chains=4,
            num_samples=250,
            num_warmup=250,
            rng=MersenneTwister(230),
        )
        @test all(<(1.05), rhat(ray_chains))
        ray_draws = vcat((vec(c.constrained_samples[1, :]) for c in ray_chains.chains)...)
        @test all(ray_draws .> 0)
        @test abs(_pht_mean(ray_draws) - true_s) < 0.3
        @test sum(sum(c.divergent) for c in ray_chains.chains) <
              0.1 * sum(length(c.divergent) for c in ray_chains.chains)

        # inversegaussian mean latent recovers mu.
        @tea static function pht_ig_mean(n)
            mu ~ lognormal(0.0, 0.5)
            for i = 1:n
                {:y => i} ~ inversegaussian(mu, 3.0)
            end
            return mu
        end
        true_mu = 2.0
        ig_ys = [rand(rng, inversegaussian(true_mu, 3.0)) for _ = 1:60]
        ig_cm = choicemap((:y => i, ig_ys[i]) for i = 1:60)
        ig_chains = nuts_chains(
            pht_ig_mean,
            (60,),
            ig_cm;
            num_chains=4,
            num_samples=250,
            num_warmup=250,
            rng=MersenneTwister(231),
        )
        @test all(<(1.05), rhat(ig_chains))
        ig_draws = vcat((vec(c.constrained_samples[1, :]) for c in ig_chains.chains)...)
        @test all(ig_draws .> 0)
        @test abs(_pht_mean(ig_draws) - true_mu) < 0.6
    end
end
