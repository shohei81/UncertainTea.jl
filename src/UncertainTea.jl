module UncertainTea

using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SpecialFunctions: digamma, loggamma, erf, erfc, erfcx, erfinv, erfcinv, logerfc, beta_inc, gamma_inc

export @tea
export ChoiceMap, TeaModel, TeaTrace
export choicemap, generate, assess, logjoint, logjoint_unconstrained, logjoint_gradient_unconstrained

# --- Internal / advanced interfaces (issue #283) ----------------------------
# The IR, execution-plan, transform, and lowering-introspection names below are
# implementation details. They remain reachable QUALIFIED (`UncertainTea.ExecutionPlan`,
# `UncertainTea.backend_report`, ...) for white-box tests and power users, but are
# deliberately NOT exported, so `using UncertainTea` no longer floods the namespace
# with ~45 compiler internals. They carry NO semver stability guarantee; the exported
# surface (see docs/src/api.md) is the supported public API.
#   IR specs         : AddressSpec, ChoiceSpec, ModelSpec, AddressLiteralPart,
#                      AddressDynamicPart, DistributionSpec, GenerativeCallSpec,
#                      RawChoiceRhsSpec, BroadcastDistributionSpec, LoopScopeSpec
#   Execution plan   : ParameterLayout, ParameterSlotSpec, ExecutionPlan, ChoicePlanStep,
#                      DeterministicPlanStep, LoopPlanStep, executionplan
#   Transforms       : IdentityTransform, VectorIdentityTransform, LogTransform,
#                      LogitTransform, SimplexTransform, CholeskyCorrTransform,
#                      CholeskyCovTransform, VectorLogTransform, VectorLogitTransform,
#                      BoundedTransform, LowerBoundedTransform, UpperBoundedTransform
#   Modes            : StaticMode, DynamicMode
#   Introspection    : modelspec, isstaticaddress, isaddresstemplate, isrepeatedchoice,
#                      hasrepeatedchoices, parameterlayout, parametercount, parametervaluecount
#   Lowering report  : BackendExecutionPlan, BackendLoweringReport, backend_report,
#                      backend_execution_plan

export batched_logjoint, batched_logjoint_unconstrained, batched_logjoint_gradient_unconstrained
export BatchedLogjointGradientCache, batched_logjoint_gradient_unconstrained!
export initialparameters, parameter_vector, parameterchoicemap
export transform_to_constrained, transform_to_unconstrained, transform_to_constrained_with_logabsdet
export HMCChain, HMCChains, HMCMassAdaptationWindowSummary, HMCMassAdaptationSummary, HMCDiagnosticsSummary, HMCParameterSummary,
    HMCSummary, SamplerWarnings
export ADVIResult, ImportanceSamplingResult, SIRResult, SMCStageSummary, SMCResult
export hmc, hmc_chains, nuts, nuts_chains, batched_hmc, batched_nuts, batched_chees, batched_meads, batched_advi
export batched_svgd, SVGDResult, particle_mean, particle_covariance
export gibbs, GibbsChain, discrete_ess
export batched_importance_sampling, batched_sir, batched_smc
export acceptancerate,
    divergencerate, massadaptationwindows, treedepths, integrationsteps, nchains, numsamples, numstages, rhat, ess, summarize
export check_diagnostics, has_warnings
export posterior_array, parameter_names, to_arviz_dict, to_mcmcchains
export reverse_mode_gradient
export pointwise_loglikelihood, observation_addresses, waic, psis_loo, loo, WAICResult, LOOResult
export sbc, SBCResult
export map_estimate, laplace_approximation, MAPResult, LaplaceResult
export pathfinder, PathfinderResult
export variational_mean, variational_samples, variational_covariance
export predict, PredictiveDraws, addresses, log_evidence
export nested_sampling, NestedSamplingResult, log_evidence_error, information
export normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, dirichlet, mvnormal, bernoulli,
    bernoullilogit, geometric, negativebinomial, poisson, studentt, categorical, betabinomial, multinomial,
    discreteuniform
export cauchy, halfnormal, halfcauchy, uniform, logistic, gumbel
export pareto, frechet, rayleigh, inversegaussian
export truncatednormal, truncatedstudentt
export mixture
export mvnormaldense, gaussianprocess, sparsegaussianprocess, gp_cholesky, hmm
export rbf_kernel, matern32_kernel, matern52_kernel, periodic_kernel, kernel_sum, kernel_product
export orderedlogistic
export zeroinflatedpoisson, zeroinflatednegativebinomial
export vonmises
export mvstudentt, mvstudenttdense
export wishart, inversewishart
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
