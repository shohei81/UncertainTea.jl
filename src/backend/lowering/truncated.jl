# Lowering of static models to the backend expression IR: truncated families (truncatednormal, truncatedstudentt).

# Backend-native truncated normal: mu/sigma are standard numeric argument
# expressions; lower/upper are numeric expressions too, but for the common latent
# case they lower to literal constants (Inf/-Inf via `_static_bound_value`). The
# scoring subtracts log(Phi(zb) - Phi(za)) from the base normal logpdf, matching
# the CPU `TruncatedNormalDist` reference exactly.
struct BackendTruncatedNormalChoicePlanStep{
    M<:AbstractBackendExpr,
    S<:AbstractBackendExpr,
    L<:AbstractBackendExpr,
    U<:AbstractBackendExpr,
    AD<:BackendAddressSpec,
} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    lower::L
    upper::U
    parameter_slot::Union{Nothing,Int}
end

# Backend-native truncated Student-t. `nu` (degrees of freedom) must lower to a
# literal constant: the normalizer `Z = T_cdf(zb, nu) - T_cdf(za, nu)` uses the
# regularized incomplete beta, whose `nu`-derivative is analytically intractable,
# so the analytic gradient omits any `d/dnu` term and is only correct when `nu`
# carries no latent dependence. A latent- or argument-flowing `nu` falls back to
# the compiled logjoint. `mu`/`sigma` are standard numeric expressions and
# `lower`/`upper` reuse the truncated-bound lowering (static bounds become
# literals, including `Inf`/`-Inf`). Scoring subtracts `log Z` from the base
# Student-t logpdf, matching the CPU `TruncatedStudentTDist` reference exactly.
struct BackendTruncatedStudentTChoicePlanStep{
    N<:AbstractBackendExpr,
    M<:AbstractBackendExpr,
    S<:AbstractBackendExpr,
    L<:AbstractBackendExpr,
    U<:AbstractBackendExpr,
    AD<:BackendAddressSpec,
} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    nu::N
    mu::M
    sigma::S
    lower::L
    upper::U
    parameter_slot::Union{Nothing,Int}
end

function _backend_lower_truncatednormal_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 4 || begin
        _backend_issue!(issues, "truncatednormal expects exactly 4 backend arguments")
        return nothing
    end
    # A latent truncatednormal draws through a bounded parameter transform that the
    # batched backend unconstrained-transform layer does not implement, so fall back
    # for latents. Observations with latent-flowing arguments stay backend-native.
    isnothing(step.parameter_slot) || begin
        _backend_issue!(issues, "latent truncatednormal is not supported in backend lowering (bounded transform)")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    mu = _backend_lower_expr(model, layout, step.rhs.arguments[1], issues, "truncatednormal mean")
    sigma = _backend_lower_expr(model, layout, step.rhs.arguments[2], issues, "truncatednormal scale")
    lower = _backend_lower_bound_expr(model, layout, step.rhs.arguments[3], issues, "truncatednormal lower bound")
    upper = _backend_lower_bound_expr(model, layout, step.rhs.arguments[4], issues, "truncatednormal upper bound")
    (isnothing(address) || isnothing(mu) || isnothing(sigma) || isnothing(lower) || isnothing(upper)) && return nothing
    return BackendTruncatedNormalChoicePlanStep(
        step.binding_slot,
        address,
        mu,
        sigma,
        lower,
        upper,
        _backend_scalar_parameter_row(parameter_layout, step.parameter_slot, issues)[1],
    )
end

function _backend_lower_truncatedstudentt_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 5 || begin
        _backend_issue!(issues, "truncatedstudentt expects exactly 5 backend arguments")
        return nothing
    end
    # A latent truncatedstudentt draws through a bounded parameter transform that
    # the batched backend unconstrained-transform layer does not implement, so fall
    # back for latents. Observations with latent-flowing mu/sigma stay backend-native.
    isnothing(step.parameter_slot) || begin
        _backend_issue!(issues, "latent truncatedstudentt is not supported in backend lowering (bounded transform)")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    nu = _backend_lower_expr(model, layout, step.rhs.arguments[1], issues, "truncatedstudentt degrees of freedom")
    # The analytic gradient omits d/dnu (the incomplete-beta nu-derivative is
    # intractable), so backend support is restricted to a constant nu. A literal
    # nu carries no latent dependence, keeping the omitted term genuinely zero;
    # any other nu expression falls back to the compiled logjoint.
    isnothing(nu) || nu isa BackendLiteralExpr ||
        begin
            _backend_issue!(
                issues,
                "truncatedstudentt backend lowering requires a constant (literal) nu; latent or argument-flowing nu falls back",
            )
            return nothing
        end
    mu = _backend_lower_expr(model, layout, step.rhs.arguments[2], issues, "truncatedstudentt mean")
    sigma = _backend_lower_expr(model, layout, step.rhs.arguments[3], issues, "truncatedstudentt scale")
    lower = _backend_lower_bound_expr(model, layout, step.rhs.arguments[4], issues, "truncatedstudentt lower bound")
    upper = _backend_lower_bound_expr(model, layout, step.rhs.arguments[5], issues, "truncatedstudentt upper bound")
    (isnothing(address) || isnothing(nu) || isnothing(mu) || isnothing(sigma) || isnothing(lower) || isnothing(upper)) &&
        return nothing
    return BackendTruncatedStudentTChoicePlanStep(
        step.binding_slot,
        address,
        nu,
        mu,
        sigma,
        lower,
        upper,
        _backend_scalar_parameter_row(parameter_layout, step.parameter_slot, issues)[1],
    )
end

function _collect_backend_slot_kinds!(
    step::BackendTruncatedNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.lower, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.upper, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendTruncatedStudentTChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.nu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.mu, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.sigma, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.lower, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.upper, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end
