# Shared fixtures: models and values used by more than one test file.
# core.jl includes this file first, before the core suite. To run any core
# test file standalone:
#
#   using Test, Random, UncertainTea
#   include("test/uncertaintea/fixtures.jl")
#   include("test/uncertaintea/core/<file>.jl")
#
# Everything here is deterministic: every RNG-consuming call seeds its own
# MersenneTwister, so fixture values do not depend on evaluation order.

using Test
using Random
using UncertainTea

# The public namespace is split into facade submodules (issue #329); pull their
# exports in so every test file keeps using the historical unqualified names.
using UncertainTea.Inference
using UncertainTea.Diagnostics
using UncertainTea.Device

# Internal (unexported) names the white-box suite reaches for. They are qualified-
# access-only in the public API (issue #283); import them explicitly here so the
# core tests keep referring to them unqualified.
using UncertainTea:
    AddressSpec, ChoiceSpec, ModelSpec, AddressLiteralPart, AddressDynamicPart,
    DistributionSpec, GenerativeCallSpec, RawChoiceRhsSpec, BroadcastDistributionSpec,
    LoopScopeSpec, ParameterLayout, ParameterSlotSpec, ExecutionPlan, ChoicePlanStep,
    DeterministicPlanStep, LoopPlanStep, executionplan, IdentityTransform,
    VectorIdentityTransform, LogTransform, LogitTransform, SimplexTransform,
    CholeskyCorrTransform, CholeskyCovTransform, VectorLogTransform, VectorLogitTransform,
    BoundedTransform, LowerBoundedTransform, UpperBoundedTransform, StaticMode, DynamicMode,
    modelspec, isstaticaddress, isaddresstemplate, isrepeatedchoice, hasrepeatedchoices,
    parameterlayout, parametercount, parametervaluecount, BackendExecutionPlan,
    BackendLoweringReport, backend_report, backend_execution_plan

@tea static function gaussian_mean()
    mu ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(mu, 1.0f0)
    return mu
end

constraints = choicemap((:y, 0.3f0))

@tea static function iid_model(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0f0)
    end
    return mu
end

ys = Float32[0.1f0, -0.2f0, 0.4f0]
repeated = choicemap((:y => i, ys[i]) for i in eachindex(ys))
trace2, logw2 = generate(iid_model, (length(ys),), repeated; rng=MersenneTwister(2))
params2 = [Float64(trace2[:mu])]

# `cbrt` stays outside the backend primitive allowlist (sin joined it in issue
# #286), so this model still exercises the honest-unsupported reporting path.
@tea static function unsupported_backend_model()
    mu ~ normal(0.0f0, 1.0f0)
    sigma = cbrt(mu)
    {:y} ~ normal(mu, 1.0f0)
    return sigma
end

@tea static function positive_step()
    sigma ~ lognormal(0.0f0, 0.5f0)
    return sigma
end

@tea static function observed_positive_step()
    sigma = ({:state} ~ positive_step())
    {:y} ~ normal(sigma, 1.0f0)
    return sigma
end

positive_step_trace, _ = generate(observed_positive_step, (), choicemap((:y, 1.2f0)); rng=MersenneTwister(14))
positive_step_unconstrained = transform_to_unconstrained(positive_step_trace)

gaussian_batch_params = reshape(Float64[-0.4, 0.0, 0.6], 1, 3)
gaussian_batch_constraints = [
    choicemap((:y, -0.1f0)),
    choicemap((:y, 0.3f0)),
    choicemap((:y, 0.7f0)),
]
gaussian_batch_gradient_cache = BatchedLogjointGradientCache(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints)

iid_batch_params = reshape(Float64[-0.2, 0.4], 1, 2)
iid_batch_args = [(2,), (3,)]
iid_batch_constraints = [
    choicemap((:y => 1, 0.1f0), (:y => 2, -0.2f0)),
    choicemap((:y => 1, 0.5f0), (:y => 2, 0.2f0), (:y => 3, -0.1f0)),
]

positive_batch_unconstrained = reshape(
    [
        positive_step_unconstrained[1] - 0.2,
        positive_step_unconstrained[1],
        positive_step_unconstrained[1] + 0.2,
    ],
    1,
    3,
)
positive_batch_constraints = [
    choicemap((:y, 0.8f0)),
    choicemap((:y, 1.2f0)),
    choicemap((:y, 1.5f0)),
]

# Seeded exact-value pins for warmup adaptation hold only on the toolchain
# family they were captured on: Julia 1.12 rewrote the randn ziggurat, so
# MersenneTwister normal draws differ on older versions, and dual averaging
# amplifies that into macroscopically different adapted values. On older
# supported versions the same tests assert version-independent invariants
# instead. If the pins drift on a future stable (the CI julia "1" jobs start
# failing), re-pin the values there and raise this bound.
adaptation_pins_exact = VERSION >= v"1.12"

@tea static function shifted_iid_model(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i = 1:n
        {:y => i + 1} ~ normal(mu, 1.0f0)
    end
    return mu
end

@tea static function offset_iid_model(n, offset)
    mu ~ normal(0.0f0, 1.0f0)
    for i = 1:n
        {:y => i + offset} ~ normal(mu, 1.0f0)
    end
    return mu
end

@tea static function indexed_scale_model(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i = 1:n
        {:y => i} ~ normal(mu, exp(mu + i / 10))
    end
    return mu
end

@tea static function chain_step(prev)
    z ~ normal(prev, 1.0f0)
    return z
end

@tea static function chain_model(T)
    z = ({:z => 1} ~ chain_step(0.0f0))
    for t = 2:T
        z = ({:z => t} ~ chain_step(z))
    end
    return z
end

@tea static function deterministic_scale()
    mu ~ normal(0.0f0, 1.0f0)
    log_sigma ~ normal(0.0f0, 1.0f0)
    sigma = exp(log_sigma)
    {:y} ~ normal(mu, sigma)
    return (; mu, sigma)
end

deterministic_trace, _ = generate(deterministic_scale, (), choicemap((:y, 0.4f0)); rng=MersenneTwister(12))
deterministic_params = parameter_vector(deterministic_trace)

@tea static function abs_scale_model()
    log_sigma ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(0.0f0, exp(abs(log_sigma)))
    return log_sigma
end

abs_scale_trace, _ = generate(abs_scale_model, (), choicemap((:y, 0.6f0)); rng=MersenneTwister(16))
abs_scale_params = parameter_vector(abs_scale_trace)

@tea static function power_scale_model()
    log_sigma ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(0.0f0, exp(log_sigma) ^ 2)
    return log_sigma
end

power_scale_trace, _ = generate(power_scale_model, (), choicemap((:y, 0.4f0)); rng=MersenneTwister(17))
power_scale_params = parameter_vector(power_scale_trace)

@tea static function min_scale_model()
    log_sigma ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(0.0f0, exp(min(log_sigma, 0.15f0)))
    return log_sigma
end

min_scale_trace, _ = generate(min_scale_model, (), choicemap((:y, 0.5f0)); rng=MersenneTwister(18))
min_scale_params = parameter_vector(min_scale_trace)

@tea static function max_scale_model()
    log_sigma ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(0.0f0, exp(max(log_sigma, -0.25f0)))
    return log_sigma
end

max_scale_trace, _ = generate(max_scale_model, (), choicemap((:y, 0.5f0)); rng=MersenneTwister(19))
max_scale_params = parameter_vector(max_scale_trace)

@tea static function mod_scale_model()
    log_sigma ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(0.0f0, exp(log_sigma % 0.5f0))
    return log_sigma
end

@tea static function clamp_scale_model()
    log_sigma ~ normal(0.0f0, 1.0f0)
    {:y} ~ normal(0.0f0, exp(clamp(log_sigma, -0.2f0, 0.3f0)))
    return log_sigma
end

clamp_scale_trace, _ = generate(clamp_scale_model, (), choicemap((:y, 0.5f0)); rng=MersenneTwister(21))
clamp_scale_params = parameter_vector(clamp_scale_trace)

@tea static function observed_coin()
    {:y} ~ bernoulli(0.25f0)
    return nothing
end

gaussian_batch_logjoint = batched_logjoint(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints)
gaussian_batch_gradient =
    batched_logjoint_gradient_unconstrained(gaussian_mean, gaussian_batch_params, (), gaussian_batch_constraints)

gaussian_shared_batch_logjoint = batched_logjoint_unconstrained(
    gaussian_mean,
    gaussian_batch_params,
    (),
    choicemap((:y, 0.4)),
)
gaussian_shared_batch_gradient = batched_logjoint_gradient_unconstrained(
    gaussian_mean,
    gaussian_batch_params,
    (),
    choicemap((:y, 0.4)),
)
