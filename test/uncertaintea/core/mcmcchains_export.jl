# Functional coverage for the UncertainTeaMCMCChainsExt package extension
# (issue #336). `posterior_array` is draw-major (num_samples, num_chains,
# num_params) while `MCMCChains.Chains` expects iterations x parameters x
# chains, so the extension must permute — the old pass-through either threw
# (3 chains x 2 params) or silently swapped parameters and chains.
#
# All UncertainTea calls are qualified on purpose: this file must stay immune
# to export reorganizations. Statistics is unavailable in the harness, so use
# a local mean helper.
#
# `import` (not `using`): MCMCChains exports `summarize`/`ess`/`rhat` too, so
# `using` it would make those unqualified names ambiguous for every later test
# file in this shared process — the exact collision the `to_mcmcchains`
# docstring warns about. Importing the module still triggers the extension.

import MCMCChains

mcx_mean(x) = sum(x) / length(x)

@testset "mcmcchains_export" begin
    @tea static function mcx_two_latent_model()
        a ~ normal(0.0f0, 1.0f0)
        b ~ normal(0.0f0, 1.0f0)
        {:y1} ~ normal(a, 1.0f0)
        {:y2} ~ normal(b, 1.0f0)
    end

    mcx_constraints = UncertainTea.choicemap((:y1, 1.5f0), (:y2, -0.5f0))
    # 3 chains x 2 params: the shape that made the unpermuted constructor throw.
    mcx_chains = UncertainTea.nuts_chains(
        mcx_two_latent_model,
        (),
        mcx_constraints;
        num_chains=3,
        num_samples=25,
        num_warmup=25,
        rng=MersenneTwister(336),
    )

    mcx_draws = UncertainTea.posterior_array(mcx_chains)
    mcx_num_samples, mcx_num_chains, mcx_num_params = size(mcx_draws)
    @test (mcx_num_samples, mcx_num_chains, mcx_num_params) == (25, 3, 2)

    # The extension is loaded (MCMCChains is a test dependency), so the
    # method-less core stub has gained its ::HMCChains method.
    @test length(methods(UncertainTea.to_mcmcchains)) >= 1

    mcx_chn = UncertainTea.to_mcmcchains(mcx_chains)
    @test mcx_chn isa MCMCChains.Chains

    mcx_internal_names = [:lp, :diverging, :energy, :tree_depth, :acceptance_rate]
    # iterations x (parameters + internals) x chains.
    @test size(mcx_chn) == (25, 2 + length(mcx_internal_names), 3)

    # Parameter names survive the conversion, in posterior_array order.
    mcx_param_names = Symbol.(UncertainTea.parameter_names(mcx_chains))
    @test collect(MCMCChains.names(mcx_chn, :parameters)) == mcx_param_names

    # The sampler statistics land in the :internals section.
    @test :internals in keys(mcx_chn.name_map)
    @test collect(MCMCChains.names(mcx_chn, :internals)) == mcx_internal_names

    # Per-(param, chain) posterior means match posterior_array exactly: any
    # draw/param/chain axis mix-up shifts these immediately.
    mcx_matrices = Array(mcx_chn, [:parameters]; append_chains=false)
    @test length(mcx_matrices) == mcx_num_chains
    for c = 1:mcx_num_chains, p = 1:mcx_num_params
        @test isapprox(
            mcx_mean(mcx_matrices[c][:, p]),
            mcx_mean(mcx_draws[:, c, p]);
            atol=1e-12,
        )
    end

    # Internals reproduce the to_arviz_dict sample_stats values.
    mcx_stats = UncertainTea.to_arviz_dict(mcx_chains)["sample_stats"]
    @test vec(mcx_chn[:, :lp, 2]) == mcx_stats["lp"][:, 2]
    @test vec(mcx_chn[:, :diverging, 1]) == Float64.(mcx_stats["diverging"][:, 1])
    @test vec(mcx_chn[:, :energy, 3]) == mcx_stats["energy"][:, 3]

    # The unconstrained space converts too (identity transforms here, so the
    # draws coincide with the constrained ones).
    mcx_chn_unconstrained = UncertainTea.to_mcmcchains(mcx_chains; space=:unconstrained)
    @test size(mcx_chn_unconstrained) == size(mcx_chn)
    @test collect(MCMCChains.names(mcx_chn_unconstrained, :parameters)) ==
          Symbol.(UncertainTea.parameter_names(mcx_chains; space=:unconstrained))
end
