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

# Issue #368: the extension extends the MCMCChains-side diagnostics generics
# (`MCMCDiagnosticTools.ess`/`rhat`, `MCMCChains.summarize`) for `HMCChains`.
# The sandbox module below reproduces a real user session — the full facade
# `using` plus `using MCMCChains` (which makes the bare names ambiguous) plus
# the ONE documented disambiguation line — in an isolated namespace, so the
# explicit imports cannot leak into the shared test process and change how
# later test files resolve `summarize`/`ess`/`rhat`.
module MCXGenericsConvention

using Test
using Random
using UncertainTea, UncertainTea.Inference, UncertainTea.Diagnostics
using MCMCChains
# The documented convention: one explicit import resolves the export clash
# toward the MCMCChains-side generics, whose HMCChains methods (added by the
# extension) forward to UncertainTea's implementations.
using MCMCChains: ess, rhat, summarize

@tea static function mcxgc_model()
    a ~ normal(0.0f0, 1.0f0)
    b ~ normal(0.0f0, 1.0f0)
    {:y1} ~ normal(a, 1.0f0)
    {:y2} ~ normal(b, 1.0f0)
end

@testset "mcmcchains_generics_convention" begin
    mcxgc_constraints = UncertainTea.choicemap((:y1, 1.5f0), (:y2, -0.5f0))
    # Even num_samples so UncertainTea's and MCMCDiagnosticTools' chain
    # splitting see identical halves in the round-trip comparison below.
    mcxgc_chains = nuts_chains(
        mcxgc_model,
        (),
        mcxgc_constraints;
        num_chains=4,
        num_samples=40,
        num_warmup=40,
        rng=MersenneTwister(368),
    )

    # Bare names work on HMCChains and accept UncertainTea's keywords.
    @test rhat(mcxgc_chains) == UncertainTea.rhat(mcxgc_chains)
    @test rhat(mcxgc_chains; method=:rank, space=:unconstrained) ==
          UncertainTea.rhat(mcxgc_chains; method=:rank, space=:unconstrained)
    @test ess(mcxgc_chains) == UncertainTea.ess(mcxgc_chains)
    @test ess(mcxgc_chains; space=:unconstrained) ==
          UncertainTea.ess(mcxgc_chains; space=:unconstrained)
    @test summarize(mcxgc_chains) isa UncertainTea.HMCSummary
    @test summarize(mcxgc_chains; per_chain=true, quantiles=(0.25, 0.75)) isa
          UncertainTea.HMCSummary

    # Keyword vocabularies are deliberately NOT aliased: MCMCDiagnosticTools'
    # `kind` keyword is rejected on HMCChains (its `kind=:basic` is the
    # non-split classic R-hat, which has no UncertainTea counterpart).
    @test_throws MethodError rhat(mcxgc_chains; kind=:rank)

    # The same (now unambiguous) generics still serve converted Chains objects
    # with the MCMCChains/MCMCDiagnosticTools behavior and keywords.
    mcxgc_chn = to_mcmcchains(mcxgc_chains)
    @test mcxgc_chn isa Chains
    mcxgc_mc_rhat = rhat(mcxgc_chn)                 # default kind=:rank
    mcxgc_mc_ess = ess(mcxgc_chn)
    @test summarize(mcxgc_chn) isa MCMCChains.ChainDataFrame

    # Round-trip agreement: UncertainTea's rank-normalized split-Rhat vs
    # MCMCDiagnosticTools' default `kind=:rank` on the converted chains. Both
    # implement Vehtari et al. (2021); tiny differences remain from
    # quantile-interpolation details in the folded statistic's median, so the
    # comparison uses atol=5e-3 (observed ~1e-5 on this seed with current MDT;
    # ~1.2e-3 on Julia 1.10's older resolved MCMCDiagnosticTools).
    mcxgc_ut_rank = UncertainTea.rhat(mcxgc_chains; method=:rank)
    for (index, name) in enumerate(Symbol.(UncertainTea.parameter_names(mcxgc_chains)))
        row = findfirst(==(name), mcxgc_mc_rhat[:, :parameters])
        @test row !== nothing
        @test isapprox(mcxgc_ut_rank[index], mcxgc_mc_rhat[row, :rhat]; atol=5e-3)
    end

    # Bulk ESS uses a different estimator on each side (UncertainTea: paired
    # autocorrelation sums; MCMCDiagnosticTools: rank-normalized FFT autocov),
    # so only loose agreement is expected — same order of magnitude.
    mcxgc_ut_ess = UncertainTea.ess(mcxgc_chains)
    for (index, name) in enumerate(Symbol.(UncertainTea.parameter_names(mcxgc_chains)))
        row = findfirst(==(name), mcxgc_mc_ess[:, :parameters])
        @test row !== nothing
        @test isapprox(mcxgc_ut_ess[index], mcxgc_mc_ess[row, :ess]; rtol=0.5)
    end

    # Qualified calls keep working unchanged alongside the merged generic.
    @test UncertainTea.rhat(mcxgc_chains) == rhat(mcxgc_chains)
end

end # module MCXGenericsConvention
