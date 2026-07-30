module UncertainTea

using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SpecialFunctions: digamma, loggamma, erf, erfc, erfcx, erfinv, erfcinv, logerfc, beta_inc, gamma_inc

export @tea
export AddressSpec, ChoiceSpec, ModelSpec
export AddressLiteralPart, AddressDynamicPart
export DistributionSpec, GenerativeCallSpec, RawChoiceRhsSpec, BroadcastDistributionSpec
export LoopScopeSpec
export ParameterLayout, ParameterSlotSpec
export ExecutionPlan, ChoicePlanStep
export DeterministicPlanStep, LoopPlanStep
export IdentityTransform, VectorIdentityTransform, LogTransform, LogitTransform, SimplexTransform, CholeskyCorrTransform
export VectorLogTransform, VectorLogitTransform
export BoundedTransform, LowerBoundedTransform, UpperBoundedTransform
export ChoiceMap, TeaModel, TeaTrace
export StaticMode, DynamicMode
export modelspec, isstaticaddress, isaddresstemplate, isrepeatedchoice, hasrepeatedchoices
export parameterlayout, parametercount, parametervaluecount
export executionplan
export choicemap, generate, assess, logjoint, logjoint_unconstrained, logjoint_gradient_unconstrained
export BackendExecutionPlan, BackendLoweringReport, backend_report, backend_execution_plan
export batched_logjoint, batched_logjoint_unconstrained, batched_logjoint_gradient_unconstrained
export BatchedLogjointGradientCache, batched_logjoint_gradient_unconstrained!
export initialparameters, parameter_vector, parameterchoicemap
export transform_to_constrained, transform_to_unconstrained, transform_to_constrained_with_logabsdet
export HMCChain, HMCChains, HMCMassAdaptationWindowSummary, HMCMassAdaptationSummary, HMCDiagnosticsSummary, HMCParameterSummary,
    HMCSummary, SamplerWarnings
export ADVIResult, ImportanceSamplingResult, SIRResult, SMCStageSummary, SMCResult
export hmc, hmc_chains, nuts, nuts_chains, batched_hmc, batched_nuts, batched_chees, batched_advi
export gibbs, GibbsChain, discrete_ess
export batched_importance_sampling, batched_sir, batched_smc
export acceptancerate,
    divergencerate, massadaptationwindows, treedepths, integrationsteps, nchains, numsamples, numstages, rhat, ess, summarize
export check_diagnostics, has_warnings
export posterior_array, parameter_names, to_arviz_dict, to_mcmcchains
export pointwise_loglikelihood, observation_addresses, waic, psis_loo, loo, WAICResult, LOOResult
export sbc, SBCResult
export map_estimate, laplace_approximation, MAPResult, LaplaceResult
export pathfinder, PathfinderResult
export variational_mean, variational_samples, variational_covariance
export predict, PredictiveDraws, addresses, log_evidence
export normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, dirichlet, mvnormal, bernoulli,
    bernoullilogit, geometric, negativebinomial, poisson, studentt, categorical
export truncatednormal, truncatedstudentt
export mixture
export mvnormaldense
export lkjcholesky, scale_cholesky
export iid
export AbstractTeaDistribution, register_distribution, registered_distributions
export device_batched_logjoint, device_batched_logjoint!, device_lowering_report, DeviceBatchedWorkspace, DeviceExecutionPlan
export device_batched_logjoint_gradient, device_batched_logjoint_gradient!
export DeviceHMCWorkspace
# binomial is intentionally not exported: it would shadow Base.binomial for users.
# Inside @tea models the name resolves to UncertainTea.binomial automatically.

include("ir.jl")
include("core.jl")
include("choicemaps.jl")
include("distributions.jl")
include("runtime.jl")
include("parameters.jl")
include("evaluator.jl")
include("generated_scorer.jl")
include("evaluator_pointwise.jl")
include("backend.jl")
include("batched.jl")
include("inference.jl")
include("frontend.jl")
include("device.jl")
include("precompile.jl")

end
