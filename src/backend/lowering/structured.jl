# Lowering of static models to the backend expression IR: structured families (mvnormal, mvnormaldense, dirichlet, lkjcholesky, mixture, broadcast/iid vector machinery).

struct BackendMvNormalChoicePlanStep{M<:Tuple,S<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
    value_index::Union{Nothing,Int}
    value_length::Int
end

struct BackendDirichletChoicePlanStep{A<:Tuple,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    alpha::A
    parameter_slot::Union{Nothing,Int}
    parameter_index::Union{Nothing,Int}
    value_index::Union{Nothing,Int}
    value_length::Int
end

# LKJ prior over a packed correlation Cholesky factor. `d` is the macro-time
# literal dimension; `eta` stays a free numeric expression so latent-dependent
# concentrations remain differentiable. Like dirichlet, the step carries both
# the unconstrained gradient-seed row (`parameter_index`) and the constrained
# value row (`value_index`) because the packed value has d extra diagonal rows.
struct BackendLKJCholeskyChoicePlanStep{E<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    d::Int
    eta::E
    parameter_slot::Union{Nothing,Int}
    parameter_index::Union{Nothing,Int}
    value_index::Union{Nothing,Int}
    value_length::Int
end

# Backend-native broadcast (vectorized) normal observation: one dense per-element
# scoring op over the observed vector instead of N loop-addressed choices. `mu` and
# `sigma` are broadcast argument expressions (scalar or vector-valued); the observed
# vector length is resolved at scoring time from the constraint value.
struct BackendBroadcastNormalChoicePlanStep{M<:AbstractBackendExpr,S<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
end

# Backend-native finite marginalized mixture of normal components. `weights` is a
# tuple of numeric argument expressions (a latent simplex flows in as slot
# expressions); `mus`/`sigmas` are per-component tuples of numeric expressions.
# Only all-normal-component mixtures are lowered here; other component families
# fall back to the compiled logjoint.
struct BackendMixtureNormalChoicePlanStep{
    W<:Tuple,
    M<:Tuple,
    S<:Tuple,
    AD<:BackendAddressSpec,
} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    weights::W
    mus::M
    sigmas::S
    parameter_slot::Union{Nothing,Int}
    weight_parameter_index::Union{Nothing,Int}
end

# Backend-native dense-covariance multivariate normal parameterized by a lower
# triangular Cholesky factor. `mu` is a tuple of numeric expressions (length d);
# `scale_tril` is a single generic expression evaluating to a whole d x d matrix
# (a model argument or captured constant — the compiled CPU path never accepts an
# inline matrix literal, so the factor is treated as constant data with zero
# gradient). Scoring solves L z = x - mu by forward substitution, matching the CPU
# `MvNormalDenseDist` reference; gradients flow only through `mu` and the value.
struct BackendMvNormalDenseChoicePlanStep{M<:Tuple,L<:AbstractBackendExpr,AD<:BackendAddressSpec} <: BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    mu::M
    scale_tril::L
    parameter_slot::Union{Nothing,Int}
    value_index::Union{Nothing,Int}
    value_length::Int
end

# Diagonal-scale multivariate Student-t (heavy-tailed `mvnormal`). Mirrors
# `BackendMvNormalChoicePlanStep` with an added degrees-of-freedom expression
# `nu`; the d dimensions couple through one quadratic form so scoring is not a
# per-component fold.
struct BackendMvStudentTChoicePlanStep{N<:AbstractBackendExpr,M<:Tuple,S<:Tuple,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    nu::N
    mu::M
    sigma::S
    parameter_slot::Union{Nothing,Int}
    value_index::Union{Nothing,Int}
    value_length::Int
end

# Dense-scale multivariate Student-t (heavy-tailed `mvnormaldense`). Mirrors
# `BackendMvNormalDenseChoicePlanStep` with an added `nu` expression.
struct BackendMvStudentTDenseChoicePlanStep{N<:AbstractBackendExpr,M<:Tuple,L<:AbstractBackendExpr,AD<:BackendAddressSpec} <:
       BackendChoicePlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    nu::N
    mu::M
    scale_tril::L
    parameter_slot::Union{Nothing,Int}
    value_index::Union{Nothing,Int}
    value_length::Int
end

# Underlying scalar operator symbol for a (possibly dotted) broadcast operator, e.g.
# `.*` -> `:*`. Returns `nothing` when the callee is not a supported broadcast op.
function _backend_broadcast_op(callee)
    callee isa Symbol || return nothing
    name = string(callee)
    base = (length(name) >= 2 && name[1] == '.') ? Symbol(name[2:end]) : callee
    return _supported_backend_primitive(base) ? base : nothing
end

# Lower a broadcast argument expression. Dotted operators are mapped to their scalar
# base op; leaves (slots, literals) reuse the standard backend lowering. Referenced
# vector model arguments stay generic environment slots (never forced numeric).
function _backend_lower_broadcast_arg(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    expr,
    issues::Vector{String},
    context::String,
)
    if expr isa Expr && expr.head == :call && !isempty(expr.args)
        op = _backend_broadcast_op(expr.args[1])
        if isnothing(op)
            _backend_issue!(issues, "unsupported broadcast argument call `$(expr.args[1])` in $context")
            return nothing
        end
        arguments = map(arg -> _backend_lower_broadcast_arg(model, layout, arg, issues, context), expr.args[2:end])
        any(isnothing, arguments) && return nothing
        return BackendPrimitiveExpr(op, tuple(arguments...))
    end
    return _backend_lower_expr(model, layout, expr, issues, context)
end

function _backend_lower_broadcast_normal_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    step.rhs.family === :normal || begin
        _backend_issue!(issues, "broadcast observations only support the normal family in backend lowering")
        return nothing
    end
    length(step.rhs.arguments) == 2 || begin
        _backend_issue!(issues, "broadcast normal expects exactly 2 backend arguments")
        return nothing
    end
    isnothing(step.parameter_slot) || begin
        _backend_issue!(issues, "broadcast normal latents are not supported in backend lowering")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    mu = _backend_lower_broadcast_arg(model, layout, step.rhs.arguments[1], issues, "broadcast normal mean")
    sigma = _backend_lower_broadcast_arg(model, layout, step.rhs.arguments[2], issues, "broadcast normal scale")
    (isnothing(address) || isnothing(mu) || isnothing(sigma)) && return nothing
    return BackendBroadcastNormalChoicePlanStep(step.binding_slot, address, mu, sigma, step.parameter_slot)
end

function _backend_lower_tuple_argument(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    expr,
    issues::Vector{String},
    context::String,
)
    if expr isa Expr && expr.head in (:vect, :tuple)
        lowered = map(arg -> _backend_lower_expr(model, layout, arg, issues, context), expr.args)
        any(isnothing, lowered) && return nothing
        return tuple(lowered...)
    elseif expr isa QuoteNode && (expr.value isa Tuple || expr.value isa AbstractVector)
        lowered = map(arg -> _backend_lower_expr(model, layout, QuoteNode(arg), issues, context), collect(expr.value))
        any(isnothing, lowered) && return nothing
        return tuple(lowered...)
    elseif expr isa Tuple || expr isa AbstractVector
        lowered = map(arg -> _backend_lower_expr(model, layout, QuoteNode(arg), issues, context), collect(expr))
        any(isnothing, lowered) && return nothing
        return tuple(lowered...)
    end

    lowered = _backend_lower_expr(model, layout, expr, issues, context)
    lowered isa BackendTupleExpr || begin
        _backend_issue!(issues, "mvnormal requires tuple-like backend arguments")
        return nothing
    end
    return lowered.arguments
end

function _backend_lower_mvnormal_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 2 || begin
        _backend_issue!(issues, "mvnormal expects exactly 2 backend arguments")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    mu = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[1], issues, "mvnormal mean")
    sigma = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[2], issues, "mvnormal scale")
    (isnothing(mu) || isnothing(sigma)) && return nothing
    length(mu) == length(sigma) || begin
        _backend_issue!(issues, "mvnormal requires mean and scale vectors with the same backend length")
        return nothing
    end
    isempty(mu) && begin
        _backend_issue!(issues, "mvnormal requires at least one backend dimension")
        return nothing
    end

    value_index = nothing
    if !isnothing(step.parameter_slot)
        slot = parameter_layout.slots[step.parameter_slot]
        slot.transform isa VectorIdentityTransform || begin
            _backend_issue!(issues, "mvnormal backend lowering expects a vector identity transform")
            return nothing
        end
        slot.value_length == length(mu) || begin
            _backend_issue!(issues, "mvnormal backend lowering requires a parameter slot with matching vector length")
            return nothing
        end
        value_index = slot.value_index
    end
    return BackendMvNormalChoicePlanStep(
        step.binding_slot,
        address,
        mu,
        sigma,
        step.parameter_slot,
        value_index,
        length(mu),
    )
end

function _backend_lower_dirichlet_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 1 || begin
        _backend_issue!(issues, "dirichlet expects exactly 1 backend argument")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    alpha = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[1], issues, "dirichlet concentration")
    isnothing(alpha) && return nothing
    length(alpha) >= 2 || begin
        _backend_issue!(issues, "dirichlet requires at least two backend dimensions")
        return nothing
    end

    parameter_index = nothing
    value_index = nothing
    if !isnothing(step.parameter_slot)
        slot = parameter_layout.slots[step.parameter_slot]
        slot.transform isa SimplexTransform || begin
            _backend_issue!(issues, "dirichlet backend lowering expects a simplex transform")
            return nothing
        end
        slot.value_length == length(alpha) || begin
            _backend_issue!(issues, "dirichlet backend lowering requires a parameter slot with matching simplex length")
            return nothing
        end
        parameter_index = slot.index
        value_index = slot.value_index
    end
    return BackendDirichletChoicePlanStep(
        step.binding_slot,
        address,
        alpha,
        step.parameter_slot,
        parameter_index,
        value_index,
        length(alpha),
    )
end

function _backend_lower_lkjcholesky_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    d = _lkjcholesky_static_size(step.rhs.arguments)
    isnothing(d) && begin
        _backend_issue!(issues, "lkjcholesky requires a literal integer dimension d >= 2 in backend lowering")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    eta = _backend_lower_expr(model, layout, step.rhs.arguments[2], issues, "lkjcholesky concentration")
    isnothing(eta) && return nothing
    value_length = (d * (d + 1)) ÷ 2

    parameter_index = nothing
    value_index = nothing
    if !isnothing(step.parameter_slot)
        slot = parameter_layout.slots[step.parameter_slot]
        slot.transform isa CholeskyCorrTransform || begin
            _backend_issue!(issues, "lkjcholesky backend lowering expects a cholesky correlation transform")
            return nothing
        end
        slot.value_length == value_length || begin
            _backend_issue!(issues, "lkjcholesky backend lowering requires a parameter slot with matching packed length")
            return nothing
        end
        parameter_index = slot.index
        value_index = slot.value_index
    end
    return BackendLKJCholeskyChoicePlanStep(
        step.binding_slot,
        address,
        d,
        eta,
        step.parameter_slot,
        parameter_index,
        value_index,
        value_length,
    )
end

# Lower one `mixture` component constructor call into `(family, mu_expr, sigma_expr)`.
# Only `normal(mu, sigma)` components are supported for backend lowering.
function _backend_lower_mixture_component(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    component,
    issues::Vector{String},
)
    if component isa Expr && component.head == :call && length(component.args) == 3 && component.args[1] === :normal
        mu = _backend_lower_expr(model, layout, component.args[2], issues, "mixture normal mean")
        sigma = _backend_lower_expr(model, layout, component.args[3], issues, "mixture normal scale")
        (isnothing(mu) || isnothing(sigma)) && return nothing
        return (mu, sigma)
    end
    _backend_issue!(issues, "mixture backend lowering only supports normal(mu, sigma) components")
    return nothing
end

function _backend_lower_mixture_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) >= 2 || begin
        _backend_issue!(issues, "mixture expects a weights argument and at least one component")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    weights = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[1], issues, "mixture weights")
    (isnothing(address) || isnothing(weights)) && return nothing
    component_count = length(step.rhs.arguments) - 1
    length(weights) == component_count || begin
        _backend_issue!(issues, "mixture requires one weight per component in backend lowering")
        return nothing
    end
    components = map(component -> _backend_lower_mixture_component(model, layout, component, issues), step.rhs.arguments[2:end])
    any(isnothing, components) && return nothing
    mus = tuple((component[1] for component in components)...)
    sigmas = tuple((component[2] for component in components)...)

    weight_parameter_index = nothing
    if !isnothing(step.parameter_slot)
        # A latent mixture value uses an IdentityTransform over the drawn scalar; the
        # weights are not a latent simplex in that case. Backend gradients only need
        # the weight-parameter index when the weights themselves are a latent simplex,
        # which is not representable through the scalar mixture parameter slot, so
        # leave it `nothing` and fall back if a genuine latent-weight case arises.
        weight_parameter_index = nothing
    end
    # like every scalar-valued step, store the slot's VALUE row (issue #36),
    # not its ordinal: scoring reads the constrained matrix with this number
    parameter_row, parameter_row_ok = _backend_scalar_parameter_row(parameter_layout, step.parameter_slot, issues)
    parameter_row_ok || return nothing
    return BackendMixtureNormalChoicePlanStep(
        step.binding_slot,
        address,
        weights,
        mus,
        sigmas,
        parameter_row,
        weight_parameter_index,
    )
end

function _backend_lower_mvnormaldense_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 2 || begin
        _backend_issue!(issues, "mvnormaldense expects exactly 2 backend arguments")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    mu = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[1], issues, "mvnormaldense mean")
    # scale_tril is a single expression evaluating to a whole matrix (a model
    # argument slot or captured constant), never an inline element grid.
    scale_tril = _backend_lower_expr(model, layout, step.rhs.arguments[2], issues, "mvnormaldense scale_tril")
    (isnothing(mu) || isnothing(scale_tril)) && return nothing
    scale_tril isa BackendSlotExpr || begin
        _backend_issue!(issues, "mvnormaldense backend lowering requires a scale_tril model argument or captured matrix")
        return nothing
    end
    dimension = length(mu)
    dimension >= 1 || begin
        _backend_issue!(issues, "mvnormaldense requires at least one backend dimension")
        return nothing
    end

    value_index = nothing
    if !isnothing(step.parameter_slot)
        slot = parameter_layout.slots[step.parameter_slot]
        slot.transform isa VectorIdentityTransform || begin
            _backend_issue!(issues, "mvnormaldense backend lowering expects a vector identity transform")
            return nothing
        end
        slot.value_length == dimension || begin
            _backend_issue!(issues, "mvnormaldense backend lowering requires a parameter slot with matching vector length")
            return nothing
        end
        value_index = slot.value_index
    end
    return BackendMvNormalDenseChoicePlanStep(
        step.binding_slot,
        address,
        mu,
        scale_tril,
        step.parameter_slot,
        value_index,
        dimension,
    )
end

function _backend_lower_mvstudentt_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 3 || begin
        _backend_issue!(issues, "mvstudentt expects exactly 3 backend arguments")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    nu = _backend_lower_expr(model, layout, step.rhs.arguments[1], issues, "mvstudentt degrees of freedom")
    mu = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[2], issues, "mvstudentt mean")
    sigma = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[3], issues, "mvstudentt scale")
    (isnothing(nu) || isnothing(mu) || isnothing(sigma)) && return nothing
    length(mu) == length(sigma) || begin
        _backend_issue!(issues, "mvstudentt requires mean and scale vectors with the same backend length")
        return nothing
    end
    isempty(mu) && begin
        _backend_issue!(issues, "mvstudentt requires at least one backend dimension")
        return nothing
    end
    value_index = nothing
    if !isnothing(step.parameter_slot)
        slot = parameter_layout.slots[step.parameter_slot]
        slot.transform isa VectorIdentityTransform || begin
            _backend_issue!(issues, "mvstudentt backend lowering expects a vector identity transform")
            return nothing
        end
        slot.value_length == length(mu) || begin
            _backend_issue!(issues, "mvstudentt backend lowering requires a parameter slot with matching vector length")
            return nothing
        end
        value_index = slot.value_index
    end
    return BackendMvStudentTChoicePlanStep(
        step.binding_slot, address, nu, mu, sigma, step.parameter_slot, value_index, length(mu),
    )
end

function _backend_lower_mvstudenttdense_choice_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    issues::Vector{String},
)
    length(step.rhs.arguments) == 3 || begin
        _backend_issue!(issues, "mvstudenttdense expects exactly 3 backend arguments")
        return nothing
    end
    address = _backend_lower_address(model, layout, step.address, issues)
    isnothing(address) && return nothing
    nu = _backend_lower_expr(model, layout, step.rhs.arguments[1], issues, "mvstudenttdense degrees of freedom")
    mu = _backend_lower_tuple_argument(model, layout, step.rhs.arguments[2], issues, "mvstudenttdense mean")
    scale_tril = _backend_lower_expr(model, layout, step.rhs.arguments[3], issues, "mvstudenttdense scale_tril")
    (isnothing(nu) || isnothing(mu) || isnothing(scale_tril)) && return nothing
    scale_tril isa BackendSlotExpr || begin
        _backend_issue!(issues, "mvstudenttdense backend lowering requires a scale_tril model argument or captured matrix")
        return nothing
    end
    dimension = length(mu)
    dimension >= 1 || begin
        _backend_issue!(issues, "mvstudenttdense requires at least one backend dimension")
        return nothing
    end
    value_index = nothing
    if !isnothing(step.parameter_slot)
        slot = parameter_layout.slots[step.parameter_slot]
        slot.transform isa VectorIdentityTransform || begin
            _backend_issue!(issues, "mvstudenttdense backend lowering expects a vector identity transform")
            return nothing
        end
        slot.value_length == dimension || begin
            _backend_issue!(issues, "mvstudenttdense backend lowering requires a parameter slot with matching vector length")
            return nothing
        end
        value_index = slot.value_index
    end
    return BackendMvStudentTDenseChoicePlanStep(
        step.binding_slot, address, nu, mu, scale_tril, step.parameter_slot, value_index, dimension,
    )
end

function _collect_backend_slot_kinds!(
    step::BackendMvStudentTChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.nu, numeric_slots, index_slots, generic_slots)
    for expr in step.mu
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    for expr in step.sigma
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendMvStudentTDenseChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.nu, numeric_slots, index_slots, generic_slots)
    for expr in step.mu
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    _mark_backend_generic_expr_slots!(step.scale_tril, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendMixtureNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for expr in step.weights
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    for expr in step.mus
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    for expr in step.sigmas
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    isnothing(step.binding_slot) || _mark_backend_numeric_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendMvNormalDenseChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for expr in step.mu
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    _mark_backend_generic_expr_slots!(step.scale_tril, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendMvNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for expr in step.mu
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    for expr in step.sigma
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendDirichletChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    for expr in step.alpha
        _mark_backend_numeric_expr_slots!(expr, numeric_slots, index_slots, generic_slots)
    end
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendLKJCholeskyChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    _mark_backend_numeric_expr_slots!(step.eta, numeric_slots, index_slots, generic_slots)
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end

function _collect_backend_slot_kinds!(
    step::BackendBroadcastNormalChoicePlanStep,
    numeric_slots::BitVector,
    index_slots::BitVector,
    generic_slots::BitVector,
)
    _mark_backend_choice_address_slots!(step.address, numeric_slots, index_slots, generic_slots)
    # Broadcast mu/sigma slots are intentionally left unmarked here: scalar latents
    # receive their numeric marking from their own defining steps, while vector model
    # arguments must stay generic (stored as whole vectors), so they default to the
    # generic environment storage. The broadcast evaluators read either kind.
    isnothing(step.binding_slot) || _mark_backend_generic_slot!(numeric_slots, index_slots, generic_slots, step.binding_slot)
    return nothing
end
