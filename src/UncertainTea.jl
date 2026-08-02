module UncertainTea

using ForwardDiff
using KernelAbstractions
using LinearAlgebra
using Random
using SpecialFunctions: digamma, loggamma, erf, erfc, erfcx, erfinv, erfcinv, logerfc, beta_inc, gamma_inc

# --- Top-level exports: the modeling language only (issue #329) --------------
# `using UncertainTea` brings in the `@tea` DSL, the trace/choicemap types, the
# core evaluation entry points, and the distribution constructors. Everything
# else lives in the facade submodules defined at the bottom of this file:
#   UncertainTea.Inference   — samplers, fitters, result types, density/parameter machinery
#   UncertainTea.Diagnostics — chain summaries, convergence checks, export helpers,
#                              model comparison, predictive checks
#   UncertainTea.Device      — the KernelAbstractions device-resident density APIs
export @tea
export ChoiceMap, TeaModel, TeaTrace
export choicemap, generate, assess, logjoint

# --- Internal / advanced interfaces (issue #283) ----------------------------
# The IR, execution-plan, transform, and lowering-introspection names below are
# implementation details. They remain reachable QUALIFIED (`UncertainTea.ExecutionPlan`,
# `UncertainTea.backend_report`, ...) for white-box tests and power users, but are
# deliberately NOT exported, so `using UncertainTea` no longer floods the namespace
# with ~45 compiler internals. They carry NO semver stability guarantee; the exported
# surface (top-level names plus the Inference/Diagnostics/Device submodule exports,
# see docs/src/api.md) is the supported public API.
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

# --- Facade submodules (issue #329) ------------------------------------------
# All definitions live in the flat parent module above; these submodules only
# import the parent bindings and re-export them, giving the public API a
# namespaced surface (`using UncertainTea.Inference`, ...) while every name
# stays reachable qualified as `UncertainTea.<name>` for white-box tests and
# extensions. Physically moving the definitions is tracked separately (#332).

"""
    UncertainTea.Inference

The inference surface: the samplers and fitters (`nuts`, `hmc`, `batched_nuts`,
`batched_advi`, `pathfinder`, `gibbs`, ...), their result types (`HMCChains`,
`ADVIResult`, ...), result accessors (`variational_mean`, `log_evidence`, ...),
and the density/parameter machinery they are built on (`logjoint_unconstrained`,
`batched_logjoint_gradient_unconstrained`, `parameter_vector`,
`transform_to_unconstrained`, ...). Load it alongside the modeling language with
`using UncertainTea, UncertainTea.Inference`; all names remain reachable
qualified as `UncertainTea.<name>`.
"""
module Inference

using ..UncertainTea:
    hmc, hmc_chains, nuts, nuts_chains, batched_hmc, batched_nuts, batched_chees,
    batched_meads, batched_advi, batched_svgd, gibbs, batched_importance_sampling,
    batched_sir, batched_smc, pathfinder, elliptical_slice, nested_sampling, sbc,
    map_estimate, laplace_approximation,
    HMCChain, HMCChains, GibbsChain, ADVIResult, ImportanceSamplingResult, SIRResult,
    SMCResult, SMCStageSummary, SVGDResult, PathfinderResult, SBCResult, MAPResult,
    LaplaceResult, NestedSamplingResult, EllipticalSliceResult,
    variational_mean, variational_samples, variational_covariance, particle_mean,
    particle_covariance, log_evidence, log_evidence_error, information,
    logjoint_unconstrained, logjoint_gradient_unconstrained, batched_logjoint,
    batched_logjoint_unconstrained, batched_logjoint_gradient_unconstrained,
    batched_logjoint_gradient_unconstrained!, BatchedLogjointGradientCache,
    reverse_mode_gradient, initialparameters, parameter_vector, parameterchoicemap,
    transform_to_constrained, transform_to_unconstrained,
    transform_to_constrained_with_logabsdet

export hmc, hmc_chains, nuts, nuts_chains, batched_hmc, batched_nuts, batched_chees,
    batched_meads, batched_advi, batched_svgd, gibbs, batched_importance_sampling,
    batched_sir, batched_smc, pathfinder, elliptical_slice, nested_sampling, sbc,
    map_estimate, laplace_approximation
export HMCChain, HMCChains, GibbsChain, ADVIResult, ImportanceSamplingResult, SIRResult,
    SMCResult, SMCStageSummary, SVGDResult, PathfinderResult, SBCResult, MAPResult,
    LaplaceResult, NestedSamplingResult, EllipticalSliceResult
export variational_mean,
    variational_samples, variational_covariance, particle_mean,
    particle_covariance, log_evidence, log_evidence_error, information
export logjoint_unconstrained, logjoint_gradient_unconstrained, batched_logjoint,
    batched_logjoint_unconstrained, batched_logjoint_gradient_unconstrained,
    batched_logjoint_gradient_unconstrained!, BatchedLogjointGradientCache,
    reverse_mode_gradient, initialparameters, parameter_vector, parameterchoicemap,
    transform_to_constrained, transform_to_unconstrained,
    transform_to_constrained_with_logabsdet

end # module Inference

"""
    UncertainTea.Diagnostics

The chain-inspection and model-checking surface: convergence summaries and
sampler-health checks (`summarize`, `rhat`, `ess`, `check_diagnostics`, ...),
draw export helpers (`posterior_array`, `to_arviz_dict`, `to_mcmcchains`, ...),
model comparison (`waic`, `psis_loo`, `loo`, `pointwise_loglikelihood`, ...),
and predictive checks (`predict`, `prior_predictive`, ...). Load it with
`using UncertainTea.Diagnostics`; all names remain reachable qualified as
`UncertainTea.<name>`.
"""
module Diagnostics

using ..UncertainTea:
    summarize, rhat, ess, discrete_ess, acceptancerate, divergencerate,
    massadaptationwindows, treedepths, integrationsteps, nchains, numsamples,
    numstages, check_diagnostics, has_warnings, SamplerWarnings,
    HMCMassAdaptationWindowSummary, HMCMassAdaptationSummary, HMCDiagnosticsSummary,
    HMCParameterSummary, HMCSummary,
    posterior_array, parameter_names, to_arviz_dict, to_mcmcchains,
    pointwise_loglikelihood, observation_addresses, waic, psis_loo, loo, WAICResult,
    LOOResult,
    predict, PredictiveDraws, addresses, prior_predictive

export summarize, rhat, ess, discrete_ess, acceptancerate, divergencerate,
    massadaptationwindows, treedepths, integrationsteps, nchains, numsamples,
    numstages, check_diagnostics, has_warnings, SamplerWarnings,
    HMCMassAdaptationWindowSummary, HMCMassAdaptationSummary, HMCDiagnosticsSummary,
    HMCParameterSummary, HMCSummary
export posterior_array, parameter_names, to_arviz_dict, to_mcmcchains
export pointwise_loglikelihood, observation_addresses, waic, psis_loo, loo, WAICResult,
    LOOResult
export predict, PredictiveDraws, addresses, prior_predictive

end # module Diagnostics

"""
    UncertainTea.Device

The device-backend surface: the KernelAbstractions device-resident batched
density APIs (`device_batched_logjoint`, `device_batched_logjoint_gradient`, and
their in-place variants), the lowering-support report (`device_lowering_report`),
and the device workspace/plan types (`DeviceBatchedWorkspace`,
`DeviceExecutionPlan`, `DeviceHMCWorkspace`). Load it with
`using UncertainTea.Device`; all names remain reachable qualified as
`UncertainTea.<name>`.
"""
module Device

using ..UncertainTea:
    device_batched_logjoint, device_batched_logjoint!, device_batched_logjoint_gradient,
    device_batched_logjoint_gradient!, device_lowering_report, DeviceBatchedWorkspace,
    DeviceExecutionPlan, DeviceHMCWorkspace

export device_batched_logjoint, device_batched_logjoint!, device_batched_logjoint_gradient,
    device_batched_logjoint_gradient!, device_lowering_report, DeviceBatchedWorkspace,
    DeviceExecutionPlan, DeviceHMCWorkspace

end # module Device

end
