# Ecosystem adapters (issue #340): the Distributions.jl registration one-liner
# (UncertainTeaDistributionsExt) and the Tables.jl interface on HMCChains /
# PredictiveDraws (UncertainTeaTablesExt).
#
# `import` (not `using`) on purpose: Distributions exports many names that
# collide with UncertainTea's distribution constructors (gamma, beta, ...) and
# Tables exports nothing we need unqualified. Importing the modules still
# triggers the package extensions.

import Distributions
import Tables

# Registrations must precede the @tea definitions that use them (the registry
# is consulted at model-definition time), and models must be top-level — same
# layout as custom_distribution_registration.jl.
eco_skewnormal_reg = register_distribution(:eco_skewnormal, Distributions.SkewNormal)
eco_chisq_reg = register_distribution(:eco_chisq, Distributions.Chisq)
eco_beta_reg = register_distribution(:eco_beta, Distributions.Beta)
eco_poisson_reg = register_distribution(:eco_poisson, Distributions.Poisson)

@tea static function eco_observed_model()
    x ~ normal(0.0, 1.0)
    {:y} ~ eco_skewnormal(x, 1.0, 2.0)
    return x
end

@tea static function eco_latent_model()
    x ~ eco_skewnormal(0.0, 1.0, 3.0)
    s ~ eco_chisq(3.0)
    {:y} ~ normal(x, s)
    return x
end

@tea static function eco_poisson_model()
    lambda ~ exponential(1.0)
    {:y} ~ eco_poisson(lambda)
    return lambda
end

@testset "ecosystem_adapters" begin
    eco_constraints = choicemap((:y, 0.4))

    @testset "eco_registration_and_support_mapping" begin
        # auto-detected supports from the Distributions.jl type
        @test eco_skewnormal_reg.transform isa IdentityTransform   # (-Inf, Inf)
        @test eco_chisq_reg.transform isa LogTransform             # (0, Inf)
        @test eco_beta_reg.transform isa LogitTransform            # (0, 1)
        @test eco_poisson_reg.transform === nothing                # discrete -> observation-only
        @test :eco_skewnormal in registered_distributions()

        # parameter-dependent support (Uniform) cannot be auto-detected
        @test_throws ArgumentError register_distribution(:eco_uniform_bad, Distributions.Uniform)
        # ... but explicit support specifications work
        @test register_distribution(:eco_unit, Distributions.Uniform; support=(0.0, 1.0)).transform isa
              LogitTransform
        @test register_distribution(:eco_bounded, Distributions.Uniform; support=(2.0, 5.0)).transform isa
              BoundedTransform
        @test register_distribution(:eco_pos, Distributions.LogNormal; support=:positive).transform isa
              LogTransform
        @test register_distribution(:eco_real, Distributions.SkewNormal; support=:real).transform isa
              IdentityTransform
        @test register_distribution(:eco_obsonly, Distributions.Uniform; support=:none).transform === nothing
        @test_throws ArgumentError register_distribution(:eco_bad_support, Distributions.SkewNormal; support=:nonsense)
        @test_throws ArgumentError register_distribution(:eco_bad_ctor, 1.0)

        # latent slots pick up the mapped transforms
        eco_layout = parameterlayout(eco_latent_model)
        @test parametercount(eco_layout) == 2
        @test eco_layout.slots[1].transform isa IdentityTransform
        @test eco_layout.slots[2].transform isa LogTransform
    end

    @testset "eco_wrap_distribution" begin
        eco_wrapped = wrap_distribution(Distributions.SkewNormal(0.0, 1.0, 2.0))
        @test eco_wrapped isa AbstractTeaDistribution
        @test UncertainTea.logpdf(eco_wrapped, 0.3) ≈
              Distributions.logpdf(Distributions.SkewNormal(0.0, 1.0, 2.0), 0.3) atol = 1e-12
        @test rand(MersenneTwister(1), eco_wrapped) isa Float64
        # univariate only: multivariate families are rejected with a clear error
        @test_throws ArgumentError wrap_distribution(Distributions.Dirichlet([1.0, 1.0]))
    end

    @testset "eco_observed_family" begin
        eco_trace, eco_logw = generate(eco_observed_model, (), eco_constraints; rng=MersenneTwister(340))
        @test isfinite(eco_logw)
        @test eco_trace[:y] == 0.4
        eco_params = parameter_vector(eco_trace)
        eco_manual =
            UncertainTea.logpdf(normal(0.0, 1.0), eco_trace[:x]) +
            Distributions.logpdf(Distributions.SkewNormal(eco_trace[:x], 1.0, 2.0), 0.4)
        @test logjoint(eco_observed_model, eco_params, (), eco_constraints) ≈ eco_manual atol = 1e-10
        @test logjoint(eco_observed_model, eco_params, (), eco_constraints) ≈
              assess(eco_observed_model, (), choicemap((:x, eco_trace[:x]), (:y, 0.4))) atol = 1e-10

        # discrete observed family: scored through Distributions.logpdf too
        eco_pois_constraints = choicemap((:y, 3))
        eco_pois_trace, _ = generate(eco_poisson_model, (), eco_pois_constraints; rng=MersenneTwister(341))
        eco_pois_manual =
            UncertainTea.logpdf(exponential(1.0), eco_pois_trace[:lambda]) +
            Distributions.logpdf(Distributions.Poisson(eco_pois_trace[:lambda]), 3)
        @test logjoint(eco_poisson_model, parameter_vector(eco_pois_trace), (), eco_pois_constraints) ≈
              eco_pois_manual atol = 1e-10
        # a latent from the transform-less (discrete) family gets no parameter slot
        @test parametercount(parameterlayout(eco_poisson_model)) == 1
    end

    @testset "eco_latent_gradient" begin
        eco_trace, _ = generate(eco_latent_model, (), eco_constraints; rng=MersenneTwister(342))
        eco_unconstrained = transform_to_unconstrained(eco_trace)
        eco_gradient = logjoint_gradient_unconstrained(eco_latent_model, eco_unconstrained, (), eco_constraints)
        for i in eachindex(eco_unconstrained)
            h = cbrt(eps(Float64)) * max(1.0, abs(eco_unconstrained[i]))
            up = copy(eco_unconstrained)
            up[i] += h
            down = copy(eco_unconstrained)
            down[i] -= h
            fd =
                (
                    logjoint_unconstrained(eco_latent_model, up, (), eco_constraints) -
                    logjoint_unconstrained(eco_latent_model, down, (), eco_constraints)
                ) / (2h)
            @test eco_gradient[i] ≈ fd atol = 5e-6
        end
    end

    # One sampled run shared by the posterior-sanity and Tables checks.
    eco_chains = nuts_chains(
        eco_latent_model,
        (),
        choicemap((:y, 0.8));
        num_chains=2,
        num_samples=30,
        num_warmup=30,
        rng=MersenneTwister(343),
    )

    @testset "eco_latent_nuts" begin
        for eco_chain in eco_chains
            @test all(isfinite, eco_chain.constrained_samples)
            @test all(eco_chain.constrained_samples[2, :] .> 0)  # Chisq latent stays positive
        end
        eco_draws = posterior_array(eco_chains)
        eco_x_mean = sum(eco_draws[:, :, 1]) / length(eco_draws[:, :, 1])
        @test -1.0 < eco_x_mean < 2.0  # pulled between the skew-normal prior mode and y=0.8
    end

    @testset "eco_tables_hmcchains" begin
        @test Tables.istable(typeof(eco_chains))
        @test Tables.columnaccess(typeof(eco_chains))
        eco_tbl = Tables.columntable(eco_chains)
        @test propertynames(eco_tbl) == (:chain, :draw, :x, :s)
        @test length(eco_tbl.chain) == 60
        @test eco_tbl.chain == vcat(fill(1, 30), fill(2, 30))
        @test eco_tbl.draw == vcat(1:30, 1:30)
        eco_draws = posterior_array(eco_chains)
        @test eco_tbl.x[1:30] == eco_draws[:, 1, 1]
        @test eco_tbl.s[31:60] == eco_draws[:, 2, 2]

        # single-chain results are HMCChains too (issue #337) and table the same way
        eco_single = nuts(
            eco_latent_model,
            (),
            choicemap((:y, 0.8));
            num_samples=10,
            num_warmup=10,
            rng=MersenneTwister(344),
        )
        eco_single_tbl = Tables.columntable(eco_single)
        @test propertynames(eco_single_tbl) == (:chain, :draw, :x, :s)
        @test length(eco_single_tbl.draw) == 10
    end

    @testset "eco_tables_predictive_draws" begin
        # prior-form predict keeps every address: latents and observations
        eco_pd = predict(eco_observed_model, (); num_draws=12, rng=MersenneTwister(345))
        @test Tables.istable(typeof(eco_pd))
        eco_pd_tbl = Tables.columntable(eco_pd)
        @test propertynames(eco_pd_tbl) == (:draw, :x, :y)
        @test eco_pd_tbl.draw == 1:12
        @test length(eco_pd_tbl.y) == 12
        @test all(isfinite, eco_pd_tbl.y)

        # loop addresses flatten to the ArviZ-style bracketed names
        eco_iid_pd = predict(iid_model, (3,); num_draws=5, rng=MersenneTwister(346))
        eco_iid_tbl = Tables.columntable(eco_iid_pd)
        @test propertynames(eco_iid_tbl) ==
              (:draw, :mu, Symbol("y[1]"), Symbol("y[2]"), Symbol("y[3]"))
        @test length(eco_iid_tbl[Symbol("y[2]")]) == 5

        # posterior-form predict tables the same way
        eco_post_pd = predict(eco_latent_model, (), eco_chains; num_draws=8, rng=MersenneTwister(347))
        eco_post_tbl = Tables.columntable(eco_post_pd)
        @test propertynames(eco_post_tbl) == (:draw, :y)
        @test length(eco_post_tbl.y) == 8
    end
end
