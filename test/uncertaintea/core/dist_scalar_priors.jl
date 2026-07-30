# Scalar prior families (issue #229): cauchy, halfnormal, halfcauchy, uniform,
# logistic, gumbel. Each is an elementary density with CPU-reference,
# backend-native analytic gradient, and (except uniform) device support.
#
# No `using Statistics` (CI hygiene): local mean/std helpers below.
_sp_mean(v) = sum(v) / length(v)
function _sp_std(v)
    m = _sp_mean(v)
    return sqrt(sum(x -> (x - m)^2, v) / (length(v) - 1))
end

@testset "dist_scalar_priors" begin
    lp = UncertainTea.logpdf

    @testset "logpdf closed forms" begin
        # cauchy(mu, sigma): -log(pi) - log(sigma) - log1p(z^2)
        let mu = 0.3, s = 2.0, x = 1.5, z = (1.5 - 0.3) / 2.0
            @test lp(cauchy(mu, s), x) ≈ -log(pi) - log(s) - log1p(z * z) atol = 1e-12
        end
        # halfnormal(sigma): log(2) + normal(0, sigma) on the nonnegative half
        @test lp(halfnormal(2.0), 1.5) ≈ log(2) - log(2.0) - log(2pi) / 2 - (1.5 / 2.0)^2 / 2 atol = 1e-12
        @test lp(halfnormal(2.0), -0.1) == -Inf
        # halfcauchy(scale): log(2) - log(pi) - log(scale) - log1p(z^2)
        @test lp(halfcauchy(2.0), 1.5) ≈ log(2) - log(pi) - log(2.0) - log1p((1.5 / 2.0)^2) atol = 1e-12
        @test lp(halfcauchy(2.0), -0.5) == -Inf
        # uniform(a, b): -log(b - a) inside, -Inf outside
        @test lp(uniform(-1.0, 3.0), 0.5) ≈ -log(4.0) atol = 1e-12
        @test lp(uniform(-1.0, 3.0), 3.5) == -Inf
        @test lp(uniform(-1.0, 3.0), -2.0) == -Inf
        # logistic(mu, s): -log(s) - z - 2 log(1 + exp(-z))
        let mu = 0.3, s = 2.0, z = (1.5 - 0.3) / 2.0
            @test lp(logistic(mu, s), 1.5) ≈ -log(s) - z - 2 * log(1 + exp(-z)) atol = 1e-12
        end
        # gumbel(mu, beta): -log(beta) - z - exp(-z)
        let mu = 0.3, b = 2.0, z = (1.5 - 0.3) / 2.0
            @test lp(gumbel(mu, b), 1.5) ≈ -log(b) - z - exp(-z) atol = 1e-12
        end
        # constructors reject out-of-support scale/bound parameters
        @test_throws ArgumentError cauchy(0.0, -1.0)
        @test_throws ArgumentError halfnormal(0.0)
        @test_throws ArgumentError halfcauchy(-2.0)
        @test_throws ArgumentError uniform(2.0, 1.0)
        @test_throws ArgumentError logistic(0.0, 0.0)
        @test_throws ArgumentError gumbel(0.0, -1.0)
    end

    @testset "value-gradient matches ForwardDiff" begin
        for d in (cauchy(0.3, 2.0), halfnormal(2.0), halfcauchy(2.0), uniform(-1.0, 3.0), logistic(0.3, 2.0), gumbel(0.3, 2.0))
            for x in (0.25, 1.5, 2.75)
                fd = UncertainTea.ForwardDiff.derivative(t -> lp(d, t), x)
                # tiny central difference cross-check
                h = 1e-6
                cd = (lp(d, x + h) - lp(d, x - h)) / (2h)
                @test isapprox(fd, cd; rtol=1e-4, atol=1e-6)
            end
        end
    end

    # ---- latent-slot wiring: transforms + backend plan step types ----
    @tea static function sp_cauchy_model()
        x ~ cauchy(0.3, 2.0)
        {:y} ~ normal(x, 0.5)
        return x
    end
    @tea static function sp_halfnormal_model()
        s ~ halfnormal(2.0)
        {:y} ~ normal(0.0, s)
        return s
    end
    @tea static function sp_halfcauchy_model()
        s ~ halfcauchy(2.0)
        {:y} ~ normal(0.0, s)
        return s
    end
    @tea static function sp_uniform_model()
        x ~ uniform(-2.0, 2.0)
        {:y} ~ normal(x, 0.4)
        return x
    end
    @tea static function sp_logistic_model()
        x ~ logistic(0.3, 2.0)
        {:y} ~ normal(x, 0.5)
        return x
    end
    @tea static function sp_gumbel_model()
        x ~ gumbel(0.3, 2.0)
        {:y} ~ normal(x, 0.5)
        return x
    end

    @testset "latent transforms and backend plan steps" begin
        @test modelspec(sp_cauchy_model).parameter_layout.slots[1].transform isa IdentityTransform
        @test modelspec(sp_halfnormal_model).parameter_layout.slots[1].transform isa LogTransform
        @test modelspec(sp_halfcauchy_model).parameter_layout.slots[1].transform isa LogTransform
        @test modelspec(sp_uniform_model).parameter_layout.slots[1].transform isa UncertainTea.BoundedTransform
        @test modelspec(sp_logistic_model).parameter_layout.slots[1].transform isa IdentityTransform
        @test modelspec(sp_gumbel_model).parameter_layout.slots[1].transform isa IdentityTransform

        @test backend_execution_plan(sp_cauchy_model).steps[1] isa UncertainTea.BackendCauchyChoicePlanStep
        @test backend_execution_plan(sp_halfnormal_model).steps[1] isa UncertainTea.BackendHalfNormalChoicePlanStep
        @test backend_execution_plan(sp_halfcauchy_model).steps[1] isa UncertainTea.BackendHalfCauchyChoicePlanStep
        @test backend_execution_plan(sp_logistic_model).steps[1] isa UncertainTea.BackendLogisticChoicePlanStep
        @test backend_execution_plan(sp_gumbel_model).steps[1] isa UncertainTea.BackendGumbelChoicePlanStep

        # dynamic uniform bounds are rejected for a LATENT at macro-expansion time
        @test_throws Exception @eval @tea static function sp_uniform_dyn(a)
            x ~ uniform(a, 3.0)
            return x
        end
    end

    # ---- analytic batched gradient vs ForwardDiff, and device parity ----
    device_families = (
        ("cauchy", sp_cauchy_model),
        ("halfnormal", sp_halfnormal_model),
        ("halfcauchy", sp_halfcauchy_model),
        ("logistic", sp_logistic_model),
        ("gumbel", sp_gumbel_model),
    )

    @testset "backend analytic gradient == ForwardDiff" begin
        for (name, model) in (device_families..., ("uniform", sp_uniform_model))
            tr, _ = generate(model, (), choicemap((:y, 0.35)); rng=MersenneTwister(160))
            p = parameter_vector(tr)
            bp = reshape(hcat(p .- 0.15, p, p .+ 0.2), length(p), 3)
            bcm = [choicemap((:y, 0.1)), choicemap((:y, 0.35)), choicemap((:y, 0.6))]
            g = batched_logjoint_gradient_unconstrained(model, bp, (), bcm)
            gref = hcat(
                [
                    UncertainTea.ForwardDiff.gradient(v -> logjoint_unconstrained(model, v, (), bcm[i]), bp[:, i]) for i = 1:3
                ]...,
            )
            @test isapprox(g, gref; rtol=1e-8, atol=1e-8)
        end
        # cauchy/logistic/gumbel/halfnormal/halfcauchy are backend-native; uniform
        # (bounded transform) honestly falls back to the ForwardDiff column path.
        @test backend_report(sp_cauchy_model).supported
        @test backend_report(sp_logistic_model).supported
        @test !backend_report(sp_uniform_model).supported
    end

    @testset "device lowering parity (CPU Float64)" begin
        for (name, model) in device_families
            supported, issues = device_lowering_report(model)
            @test supported
            @test isempty(issues)
            tr, _ = generate(model, (), choicemap((:y, 0.35)); rng=MersenneTwister(160))
            p = parameter_vector(tr)
            bp = reshape(hcat(p .- 0.15, p, p .+ 0.2), length(p), 3)
            bcm = [choicemap((:y, 0.1)), choicemap((:y, 0.35)), choicemap((:y, 0.6))]
            vref = batched_logjoint_unconstrained(model, bp, (), bcm)
            gref = batched_logjoint_gradient_unconstrained(model, bp, (), bcm)
            v, g = device_batched_logjoint_gradient(model, bp, (), bcm)
            @test isapprox(v, vref; rtol=1e-10, atol=1e-10)
            @test isapprox(g, gref; rtol=1e-10, atol=1e-10)
        end
        # uniform is deliberately not device-lowered (BoundedTransform is not a
        # device transform); the report must say so honestly.
        supported_u, issues_u = device_lowering_report(sp_uniform_model)
        @test !supported_u
        @test !isempty(issues_u)
    end

    @testset "NUTS posterior recovery" begin
        # uniform latent: bounded posterior concentrates near the observation.
        uni_chains = nuts_chains(
            sp_uniform_model,
            (),
            choicemap((:y, 0.7));
            num_chains=4,
            num_samples=250,
            num_warmup=250,
            rng=MersenneTwister(229),
        )
        uni_r = rhat(uni_chains)
        @test all(<(1.05), uni_r)
        uni_draws = vcat((vec(c.constrained_samples[1, :]) for c in uni_chains.chains)...)
        @test all(-2.0 .<= uni_draws .<= 2.0)
        @test abs(_sp_mean(uni_draws) - 0.62) < 0.25
        @test sum(sum(c.divergent) for c in uni_chains.chains) <
              0.1 * sum(length(c.divergent) for c in uni_chains.chains)

        # halfnormal scale latent recovers a sane positive scale.
        @tea static function sp_hn_scale(n)
            s ~ halfnormal(2.0)
            for i = 1:n
                {:y => i} ~ normal(0.0, s)
            end
            return s
        end
        ys = 1.3 .* randn(MersenneTwister(9), 40)
        hn_cm = choicemap((:y => i, ys[i]) for i = 1:40)
        hn_chains = nuts_chains(
            sp_hn_scale,
            (40,),
            hn_cm;
            num_chains=4,
            num_samples=250,
            num_warmup=250,
            rng=MersenneTwister(230),
        )
        @test all(<(1.05), rhat(hn_chains))
        hn_draws = vcat((vec(c.constrained_samples[1, :]) for c in hn_chains.chains)...)
        @test all(hn_draws .> 0)
        @test 0.9 < _sp_mean(hn_draws) < 1.8
    end
end
