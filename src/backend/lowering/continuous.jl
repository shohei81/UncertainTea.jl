# Lowering of static models to the backend expression IR: continuous scalar families (normal, lognormal, laplace, exponential, gamma, inversegamma, weibull, beta, studentt).

const GPU_BACKEND_SUPPORTED_DISTRIBUTIONS = Symbol[
    :normal,
    :lognormal,
    :laplace,
    :exponential,
    :gamma,
    :inversegamma,
    :weibull,
    :beta,
    :dirichlet,
    :bernoulli,
    :bernoullilogit,
    :binomial,
    :geometric,
    :negativebinomial,
    :poisson,
    :studentt,
    :categorical,
    :mvnormal,
    :truncatednormal,
    :truncatedstudentt,
    :mixture,
    :mvnormaldense,
    :lkjcholesky,
]

struct BackendNormalChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
end

# reparam=:noncentered normal latent: the raw slot value is the standardized
# z scored against N(0, 1); the binding materializes theta = mu + sigma * z,
# so downstream expressions and slot gradients flow through the affine form
# (docs/noncentered-reparam.md, PR-4).
struct BackendNoncenteredNormalChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
end

struct BackendLognormalChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
end

struct BackendExponentialChoicePlanStep{R<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    rate::R
    parameter_slot::Union{Nothing,Int}
end

struct BackendGammaChoicePlanStep{SH<:AbstractBackendExpr,R<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    shape::SH
    rate::R
    parameter_slot::Union{Nothing,Int}
end

struct BackendBetaChoicePlanStep{A<:AbstractBackendExpr,B<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    alpha::A
    beta::B
    parameter_slot::Union{Nothing,Int}
end

struct BackendStudentTChoicePlanStep{
    N<:AbstractBackendExpr,
    M<:AbstractBackendExpr,
    S<:AbstractBackendExpr,
    AD<:BackendAddressSpec,
} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    nu::N
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
end

function _collect_backend_slot_kinds!(
    step::BackendNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendNoncenteredNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendLognormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendExponentialChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.rate, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendGammaChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.shape, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.rate, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendBetaChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.alpha, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.beta, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendStudentTChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.nu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end
struct BackendLaplaceChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    scale::S
    parameter_slot::Union{Nothing,Int}
end

struct BackendInverseGammaChoicePlanStep{SH<:AbstractBackendExpr,SC<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    shape::SH
    scale::SC
    parameter_slot::Union{Nothing,Int}
end

struct BackendWeibullChoicePlanStep{SH<:AbstractBackendExpr,SC<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    shape::SH
    scale::SC
    parameter_slot::Union{Nothing,Int}
end

function _collect_backend_slot_kinds!(
    step::BackendLaplaceChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.scale, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendInverseGammaChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.shape, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.scale, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendWeibullChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.shape, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.scale, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end
