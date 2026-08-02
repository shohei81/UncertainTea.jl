# Namespace partition (issue #329): the flat pre-#329 export surface is split
# into a minimal modeling-language top level plus three facade submodules
# (Inference / Diagnostics / Device). This test pins the partition:
#   (a) the union of the four export sets equals the old flat export list,
#   (b) no name is exported from two places,
#   (c) moved names are gone from the top level and present in their submodule,
#   (d) every facade-exported name resolves to a parent binding.

# The complete `export` list of src/UncertainTea.jl immediately before #329
# (hardcoded on purpose: the whole point is to notice accidental drops/renames).
# Names introduced AFTER the partition are appended below with their issue.
const PRE_329_EXPORTS = Set{Symbol}([
    # posterior-draws interface (issue #337, exported from Diagnostics)
    :constrained_draws, :PosteriorDrawsResult,
    # modeling language
    Symbol("@tea"), :ChoiceMap, :TeaModel, :TeaTrace, :choicemap, :generate, :assess, :logjoint,
    # density / parameter machinery
    :logjoint_unconstrained, :logjoint_gradient_unconstrained,
    :batched_logjoint, :batched_logjoint_unconstrained, :batched_logjoint_gradient_unconstrained,
    :BatchedLogjointGradientCache, :batched_logjoint_gradient_unconstrained!,
    :initialparameters, :parameter_vector, :parameterchoicemap,
    :transform_to_constrained, :transform_to_unconstrained, :transform_to_constrained_with_logabsdet,
    # samplers, fitters, and result types
    :HMCChain, :HMCChains, :HMCMassAdaptationWindowSummary, :HMCMassAdaptationSummary,
    :HMCDiagnosticsSummary, :HMCParameterSummary, :HMCSummary, :SamplerWarnings,
    :ADVIResult, :ImportanceSamplingResult, :SIRResult, :SMCStageSummary, :SMCResult,
    :hmc, :hmc_chains, :nuts, :nuts_chains, :batched_hmc, :batched_nuts, :batched_chees,
    :batched_meads, :batched_advi,
    :batched_svgd, :SVGDResult, :particle_mean, :particle_covariance,
    :gibbs, :GibbsChain, :discrete_ess,
    :batched_importance_sampling, :batched_sir, :batched_smc,
    # diagnostics and export helpers
    :acceptancerate, :divergencerate, :massadaptationwindows, :treedepths, :integrationsteps,
    :nchains, :numsamples, :numstages, :rhat, :ess, :summarize,
    :check_diagnostics, :has_warnings,
    :posterior_array, :parameter_names, :to_arviz_dict, :to_mcmcchains,
    :reverse_mode_gradient,
    :pointwise_loglikelihood, :observation_addresses, :waic, :psis_loo, :loo, :WAICResult, :LOOResult,
    :sbc, :SBCResult,
    :map_estimate, :laplace_approximation, :MAPResult, :LaplaceResult,
    :pathfinder, :PathfinderResult,
    :variational_mean, :variational_samples, :variational_covariance,
    :predict, :PredictiveDraws, :addresses, :log_evidence,
    :prior_predictive,
    :nested_sampling, :NestedSamplingResult, :log_evidence_error, :information,
    :elliptical_slice, :EllipticalSliceResult,
    # distributions
    :normal, :lognormal, :laplace, :exponential, :gamma, :inversegamma, :weibull, :beta,
    :dirichlet, :mvnormal, :bernoulli, :bernoullilogit, :geometric, :negativebinomial,
    :poisson, :studentt, :categorical, :betabinomial, :multinomial, :discreteuniform,
    :cauchy, :halfnormal, :halfcauchy, :uniform, :logistic, :gumbel,
    :pareto, :frechet, :rayleigh, :inversegaussian,
    :truncatednormal, :truncatedstudentt,
    :mixture,
    :mvnormaldense, :gaussianprocess, :sparsegaussianprocess, :gp_cholesky, :hmm,
    :rbf_kernel, :matern32_kernel, :matern52_kernel, :periodic_kernel, :kernel_sum, :kernel_product,
    :orderedlogistic,
    :zeroinflatedpoisson, :zeroinflatednegativebinomial,
    :vonmises,
    :mvstudentt, :mvstudenttdense,
    :wishart, :inversewishart,
    :lkjcholesky, :scale_cholesky,
    :iid,
    :AbstractTeaDistribution, :register_distribution, :registered_distributions,
    # device backend
    :device_batched_logjoint, :device_batched_logjoint!, :device_lowering_report,
    :DeviceBatchedWorkspace, :DeviceExecutionPlan,
    :device_batched_logjoint_gradient, :device_batched_logjoint_gradient!,
    :DeviceHMCWorkspace,
])

# names(M) includes the module's own name; strip module self-names everywhere.
_exports(m::Module) = setdiff(Set(names(m)), Set([nameof(m)]))

@testset "namespace partition (issue #329)" begin
    top = _exports(UncertainTea)
    inference = _exports(UncertainTea.Inference)
    diagnostics = _exports(UncertainTea.Diagnostics)
    device = _exports(UncertainTea.Device)

    # (a) nothing dropped, nothing invented: the union is exactly the old surface
    new_union = union(top, inference, diagnostics, device)
    @test isempty(setdiff(PRE_329_EXPORTS, new_union))   # dropped names
    @test isempty(setdiff(new_union, PRE_329_EXPORTS))   # invented names

    # (b) the partition is disjoint: no name exported from two places
    sets = [top, inference, diagnostics, device]
    for i = 1:4, j = (i+1):4
        @test isempty(intersect(sets[i], sets[j]))
    end

    # (c) spot checks: moved names left the top level and landed in one submodule
    @test !Base.isexported(UncertainTea, :nuts)
    @test Base.isexported(UncertainTea.Inference, :nuts)
    @test !Base.isexported(UncertainTea, :summarize)
    @test Base.isexported(UncertainTea.Diagnostics, :summarize)
    @test !Base.isexported(UncertainTea, :device_batched_logjoint)
    @test Base.isexported(UncertainTea.Device, :device_batched_logjoint)
    @test !Base.isexported(UncertainTea, :logjoint_unconstrained)
    @test Base.isexported(UncertainTea.Inference, :logjoint_unconstrained)
    # ... while the modeling language stayed put
    @test Base.isexported(UncertainTea, Symbol("@tea"))
    @test Base.isexported(UncertainTea, :choicemap)
    @test Base.isexported(UncertainTea, :normal)
    @test Base.isexported(UncertainTea, :logjoint)

    # (d) every facade-exported name resolves, and to the parent's binding
    for (mod, exported) in
        ((UncertainTea.Inference, inference), (UncertainTea.Diagnostics, diagnostics), (UncertainTea.Device, device))
        for name in exported
            @test isdefined(mod, name)
            @test getfield(mod, name) === getfield(UncertainTea, name)
        end
    end
end
