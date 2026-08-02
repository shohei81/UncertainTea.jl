function _evaluation_module(model::TeaModel)
    return parentmodule(model.impl)
end

mutable struct PlanEnvironment
    layout::EnvironmentLayout
    values::Vector{Any}
    assigned::BitVector
    # Stan-style reject semantics for sampling-context evaluation (issue #157):
    # when true, invalid distribution parameters at a choice step (a constructor
    # ArgumentError/DomainError, e.g. `normal` with an exp-underflowed sigma == 0
    # reached mid-trajectory) score as log-density -Inf for THAT evaluation
    # instead of throwing, so one bad chain/lane registers a divergence rather
    # than killing a whole batched run. Off (throwing) for the public logjoint
    # APIs, where an invalid parameter is a model/user error to surface.
    reject_invalid_parameters::Bool
end

function PlanEnvironment(layout::EnvironmentLayout; reject_invalid_parameters::Bool=false)
    return PlanEnvironment(
        layout,
        Vector{Any}(undef, length(layout.symbols)),
        falses(length(layout.symbols)),
        reject_invalid_parameters,
    )
end

function _environment_slot(layout::EnvironmentLayout, symbol::Symbol)
    return get(layout.slot_by_symbol, symbol, nothing)
end

function _environment_hasvalue(env::PlanEnvironment, slot::Int)
    return env.assigned[slot]
end

function _environment_hasvalue(env::PlanEnvironment, symbol::Symbol)
    slot = _environment_slot(env.layout, symbol)
    return !isnothing(slot) && _environment_hasvalue(env, slot)
end

function _environment_value(env::PlanEnvironment, slot::Int)
    env.assigned[slot] || throw(ArgumentError("environment slot $slot is not assigned"))
    return env.values[slot]
end

function _environment_value(env::PlanEnvironment, symbol::Symbol)
    slot = _environment_slot(env.layout, symbol)
    isnothing(slot) && throw(ArgumentError("environment does not track symbol `$symbol`"))
    return _environment_value(env, slot)
end

function _environment_set!(env::PlanEnvironment, slot::Int, value)
    env.values[slot] = value
    env.assigned[slot] = true
    return value
end

function _environment_restore!(env::PlanEnvironment, slot::Int, previous_value, was_assigned::Bool)
    if was_assigned
        env.values[slot] = previous_value
        env.assigned[slot] = true
    else
        env.assigned[slot] = false
    end
    return nothing
end

# Full-environment snapshot/restore for suffix re-evaluation under
# enumeration: suffix steps may rebind slots from their own prior values
# (`x = x + 1`), so restoring only the enumerated binding would leak one
# branch's mutations into the next.
_environment_snapshot(env::PlanEnvironment) = (copy(env.values), copy(env.assigned))

function _environment_restore_snapshot!(env::PlanEnvironment, snapshot::Tuple{Vector{Any},BitVector})
    copyto!(env.values, snapshot[1])
    copyto!(env.assigned, snapshot[2])
    return nothing
end

abstract type AbstractCompiledExpr end
abstract type AbstractCompiledAddressPart end
abstract type AbstractCompiledPlanStep end

struct CompiledLiteralExpr{T} <: AbstractCompiledExpr
    value::T
end

struct CompiledSlotExpr <: AbstractCompiledExpr
    slot::Int
end

struct CompiledCallExpr{C<:AbstractCompiledExpr,A<:Tuple} <: AbstractCompiledExpr
    callee::C
    arguments::A
end

struct CompiledTupleExpr{A<:Tuple} <: AbstractCompiledExpr
    arguments::A
end

struct CompiledVectorExpr{A<:Tuple} <: AbstractCompiledExpr
    arguments::A
end

struct CompiledBlockExpr{A<:Tuple} <: AbstractCompiledExpr
    arguments::A
end

struct CompiledAddressLiteralPart{T} <: AbstractCompiledAddressPart
    value::T
end

struct CompiledAddressDynamicPart{E<:AbstractCompiledExpr} <: AbstractCompiledAddressPart
    expr::E
end

struct CompiledAddressSpec{P<:Tuple}
    parts::P
end

# reparam=:noncentered walk data: theta = location + scale * z, or
# exp(location + scale * z) for the log-space (lognormal) variant.
struct CompiledNoncentered
    location::Any
    scale::Any
    logspace::Bool
end

# marginalize=:enumerate walk data (docs/discrete-enumeration.md): the
# compile-time support values to enumerate. bernoulli: (false, true);
# categorical: (1, ..., K) from the literal probability-vector length.
struct CompiledMarginalize
    support::Tuple
end

struct CompiledChoicePlanStep{A<:Tuple,AD<:CompiledAddressSpec,C} <: AbstractCompiledPlanStep
    binding_slot::Union{Nothing,Int}
    address::AD
    constructor::C
    arguments::A
    parameter_value_indices::Union{Nothing,UnitRange{Int}}
    parameter_slot::Union{Nothing,Int}
    # compiled location/scale (plus log-space flag) for reparam=:noncentered
    # latents; `nothing` for centered choices
    noncentered::Union{Nothing,CompiledNoncentered}
    # compile-time enumeration support for marginalize=:enumerate latents;
    # `nothing` for ordinary choices
    marginalize::Union{Nothing,CompiledMarginalize}
    # Dense observed-value staging (issue #145). A loop observation whose
    # address is `(literal..., loop-index)` gets a per-plan stage index; a
    # LogjointGradientCache pre-resolves its constrained values into a dense
    # Float64 vector indexed by the loop index, and the scoring loop reads
    # `values[i]` instead of assembling an address tuple and hashing into the
    # ChoiceMap. 0 when the step does not qualify. `stage_iterator_slot` is
    # the environment slot of the enclosing loop iterator the address's final
    # dynamic part reads (0 when `stage_index` is 0).
    stage_index::Int
    stage_iterator_slot::Int
end

struct CompiledDeterministicPlanStep{E<:AbstractCompiledExpr} <: AbstractCompiledPlanStep
    binding_slot::Int
    expr::E
end

struct CompiledLoopPlanStep{I<:AbstractCompiledExpr,B<:Tuple} <: AbstractCompiledPlanStep
    iterator_slot::Int
    iterable::I
    body::B
end

struct CompiledExecutionPlan{S<:Tuple}
    steps::S
    # Environment slots that some reparam=:noncentered location/scale expression
    # transitively depends on (issue #100). The dependent-transform walk evaluates
    # deterministic steps and loops only when they populate one of these slots;
    # steps outside this dependency cone are irrelevant to the change of variables,
    # and eagerly evaluating them would poison the walk whenever an unrelated
    # slotless choice (a marginalize=:enumerate site or a loop-scoped binding)
    # feeds them. Empty when the plan has no noncentered site.
    required_walk_slots::Set{Int}
    # Number of stageable loop-observation sites in the plan (issue #145); the
    # per-cache dense observation stage sizes its site vector from this count.
    stage_count::Int
end

# If `callee` is a dotted operator symbol (e.g. `.*`, `.+`), return the underlying
# scalar operator function; otherwise `nothing`. Used to lower broadcast argument
# expressions of broadcast (dot-call) distribution observations.
function _broadcast_operator_function(callee)
    callee isa Symbol || return nothing
    name = string(callee)
    (length(name) >= 2 && name[1] == '.') || return nothing
    base = Symbol(name[2:end])
    isdefined(Base, base) || return nothing
    value = getfield(Base, base)
    return value isa Function ? value : nothing
end

# Per-model plan-BUILD de-specialization (issue #155 part 2). The plan
# compilers below run ONCE per (model, signature) -- their results are memoized
# in `model.evaluator_cache`/`model.signature_cache` -- yet Julia used to
# recompile them for every distinct `TeaModel{M,F,S}` because the `impl` field
# `F` is a unique closure type per `@tea` model. That per-model JIT was the
# ~7-11 s tax #155 measured. `model` is used here only as a DATA source (its
# evaluation module, name, and IR); it never enters the type of the
# `Compiled*` plan the compiler returns -- that type is driven entirely by the
# `expr`/`step` structure. So `@nospecialize(model)` lets these builders
# compile a SINGLE time (against the abstract `TeaModel`) and be reused across
# every model without changing the emitted plan or any numeric result. This is
# deliberately confined to the plan-build layer: the generated scorer (#144,
# `_gen_*`) still specializes on the concrete plan for run-time speed, and the
# hot batched/scalar scoring walks are untouched.
function _resolve_compile_symbol(@nospecialize(model::TeaModel), layout::EnvironmentLayout, sym::Symbol)
    slot = _environment_slot(layout, sym)
    if !isnothing(slot)
        return CompiledSlotExpr(slot)
    elseif sym === Symbol(":")
        return CompiledLiteralExpr(getfield(Base, Symbol(":")))
    elseif sym === Symbol("=>")
        return CompiledLiteralExpr(getfield(Base, Symbol("=>")))
    end

    module_ = _evaluation_module(model)
    if isdefined(module_, sym)
        return CompiledLiteralExpr(getfield(module_, sym))
    elseif isdefined(@__MODULE__, sym)
        return CompiledLiteralExpr(getfield(@__MODULE__, sym))
    elseif isdefined(Base, sym)
        return CompiledLiteralExpr(getfield(Base, sym))
    end

    throw(ArgumentError("lower-level logjoint could not resolve symbol `$sym` in model $(model.name)"))
end

# `sum(a .* b)` compiled without the intermediate broadcast array (issue #145).
# Below 16 elements Base's `mapreduce` reduces sequentially without `@simd`, so
# summing the lazy `Broadcasted` visits the identical element values in the
# identical order as summing the materialized array and the result is
# bit-identical while skipping the per-call allocation -- this covers the
# per-observation `sum(beta .* X[:, i])` dots the rewrite targets. From 16
# elements up, `@simd` may associate array loads and lazily computed products
# differently (observed to diverge in the last ulp), so materialize exactly as
# the unrewritten `sum(broadcast(*, a, b))` would.
function _sum_broadcast_multiply(a, b)
    lazy = Broadcast.instantiate(Broadcast.broadcasted(*, a, b))
    length(lazy) < 16 && return sum(lazy)
    return sum(Broadcast.materialize(lazy))
end

_compiled_literal_is(expr::AbstractCompiledExpr, value) =
    expr isa CompiledLiteralExpr && expr.value === value

# Post-compilation call rewrites (issue #145): `X[:, i]` (a `getindex` with a
# literal `Colon` index) becomes a `view`, so per-observation column reads stop
# materializing fresh vectors, and `sum(a .* b)` becomes an allocation-free
# broadcast reduction. Both rewrites read the same element values in the same
# order as the original calls.
function _rewrite_compiled_call(expr::CompiledCallExpr)
    if _compiled_literal_is(expr.callee, getindex) &&
       any(arg -> arg isa CompiledLiteralExpr && arg.value isa Colon, expr.arguments)
        return CompiledCallExpr(CompiledLiteralExpr(view), expr.arguments)
    end
    if _compiled_literal_is(expr.callee, sum) && length(expr.arguments) == 1
        inner = expr.arguments[1]
        if inner isa CompiledCallExpr &&
           _compiled_literal_is(inner.callee, broadcast) &&
           length(inner.arguments) == 3 &&
           _compiled_literal_is(inner.arguments[1], *)
            return CompiledCallExpr(
                CompiledLiteralExpr(_sum_broadcast_multiply),
                (inner.arguments[2], inner.arguments[3]),
            )
        end
    end
    return expr
end

# Build a dense `nrows` x `ncols` matrix from row-major elements. `map(identity,
# ...)` narrows a `Vector{Any}` (e.g. elements carrying ForwardDiff Duals) to the
# promoted element type so downstream arithmetic stays differentiable.
function _build_matrix(nrows::Int, ncols::Int, elements...)
    narrowed = map(identity, collect(elements))
    result = Matrix{eltype(narrowed)}(undef, nrows, ncols)
    index = 1
    for row = 1:nrows, col = 1:ncols
        result[row, col] = narrowed[index]
        index += 1
    end
    return result
end

function _compile_plan_expr(@nospecialize(model::TeaModel), layout::EnvironmentLayout, expr)
    if expr isa QuoteNode
        return CompiledLiteralExpr(expr.value)
    elseif expr isa Symbol
        return _resolve_compile_symbol(model, layout, expr)
    elseif expr isa GlobalRef
        return CompiledLiteralExpr(getfield(expr.mod, expr.name))
    elseif expr isa Expr
        if expr.head == :call
            base_op = _broadcast_operator_function(expr.args[1])
            if !isnothing(base_op)
                # A dotted operator like `.*`: evaluate as `broadcast(op, args...)` so
                # scalar/vector broadcast arguments combine with standard rules.
                arguments = tuple(
                    CompiledLiteralExpr(base_op),
                    (_compile_plan_expr(model, layout, arg) for arg in expr.args[2:end])...,
                )
                return CompiledCallExpr(CompiledLiteralExpr(broadcast), arguments)
            end
            callee = _compile_plan_expr(model, layout, expr.args[1])
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args[2:end])...)
            return _rewrite_compiled_call(CompiledCallExpr(callee, arguments))
        elseif expr.head == :. && length(expr.args) == 2 &&
               expr.args[2] isa Expr && expr.args[2].head == :tuple
            # dotted FUNCTION call `f.(args...)` (issue #287): evaluate as
            # `broadcast(f, args...)`, the same rule the dotted operators above
            # use, so vectorized deterministic expressions compile instead of
            # rejecting the plan.
            callee = _compile_plan_expr(model, layout, expr.args[1])
            arguments = tuple(
                callee,
                (_compile_plan_expr(model, layout, arg) for arg in expr.args[2].args)...,
            )
            return CompiledCallExpr(CompiledLiteralExpr(broadcast), arguments)
        elseif expr.head == :block
            arguments = tuple((
                _compile_plan_expr(model, layout, arg) for arg in expr.args if !(arg isa LineNumberNode)
            )...)
            return CompiledBlockExpr(arguments)
        elseif expr.head == :tuple
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args)...)
            return CompiledTupleExpr(arguments)
        elseif expr.head == :vect
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args)...)
            return CompiledVectorExpr(arguments)
        elseif expr.head == :vcat || expr.head == :hcat
            # matrix / row-vector literal, e.g. `[a b; c d]` (a `:vcat` of `:row`s)
            # or `[a b]` (an `:hcat`). Reconstruct row-major into a dense matrix so
            # `mvnormaldense`/`mvstudenttdense`/`wishart` scale-matrix literals score.
            rows =
                expr.head == :hcat ? Any[expr.args] :
                Any[row isa Expr && row.head == :row ? row.args : Any[row] for row in expr.args]
            ncols = length(first(rows))
            all(row -> length(row) == ncols, rows) ||
                throw(ArgumentError("matrix literal rows must all have the same length: $expr"))
            nrows = length(rows)
            elements = Any[]
            for row in rows, element in row
                push!(elements, _compile_plan_expr(model, layout, element))
            end
            return CompiledCallExpr(
                CompiledLiteralExpr(_build_matrix),
                tuple(CompiledLiteralExpr(nrows), CompiledLiteralExpr(ncols), elements...),
            )
        elseif expr.head == :ref
            callee = CompiledLiteralExpr(getindex)
            arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in expr.args)...)
            return _rewrite_compiled_call(CompiledCallExpr(callee, arguments))
        end

        throw(ArgumentError("unsupported expression in lower-level logjoint compilation: $expr"))
    end

    return CompiledLiteralExpr(expr)
end

function _compile_address(layout::EnvironmentLayout, @nospecialize(model::TeaModel), address::AddressSpec)
    parts = tuple((
        begin
            if part isa AddressLiteralPart
                CompiledAddressLiteralPart(part.value)
            else
                CompiledAddressDynamicPart(_compile_plan_expr(model, layout, part.value))
            end
        end for part in address.parts
    )...)
    return CompiledAddressSpec(parts)
end

# A choice step qualifies for dense observed-value staging (issue #145) when
# its value comes from the constraints (no parameter slot), it is not an
# enumerated latent, and its address is `(literal..., loop-index)` -- every
# part literal except a final dynamic part that reads an enclosing loop's
# iterator slot. Returns `(stage_index, iterator_slot)` or `(0, 0)`.
function _stage_observation_marker(
    address::CompiledAddressSpec,
    parameter_slot::Union{Nothing,Int},
    marginalize::Union{Nothing,CompiledMarginalize},
    loop_iterator_slots::Tuple{Vararg{Int}},
    stage_counter::Base.RefValue{Int},
)
    isnothing(parameter_slot) || return (0, 0)
    isnothing(marginalize) || return (0, 0)
    parts = address.parts
    length(parts) >= 2 || return (0, 0)
    all(part -> part isa CompiledAddressLiteralPart, Base.front(parts)) || return (0, 0)
    tail = last(parts)
    tail isa CompiledAddressDynamicPart || return (0, 0)
    tail.expr isa CompiledSlotExpr || return (0, 0)
    tail.expr.slot in loop_iterator_slots || return (0, 0)
    stage_counter[] += 1
    return (stage_counter[], tail.expr.slot)
end

_static_vector_obs_constructor(constructor) =
    constructor === gaussianprocess ||
    constructor === sparsegaussianprocess ||
    constructor === hmm ||
    constructor === BroadcastNormalDist ||
    (constructor isa UnionAll && constructor <: BroadcastScalarDist)

function _compile_plan_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::ChoicePlanStep,
    loop_iterator_slots::Tuple{Vararg{Int}},
    stage_counter::Base.RefValue{Int},
)
    step.rhs isa DistributionSpec || step.rhs isa BroadcastDistributionSpec ||
        throw(ArgumentError("compiled lower-level logjoint only supports distribution choice steps"))
    arguments = tuple((_compile_plan_expr(model, layout, arg) for arg in step.rhs.arguments)...)
    constructor = if step.rhs isa BroadcastDistributionSpec
        # normal keeps its dedicated runtime dist; every other broadcast family
        # constructs the generic BroadcastScalarDist{family} (issue #287)
        step.rhs.family === :normal ? getfield(@__MODULE__, :BroadcastNormalDist) :
        BroadcastScalarDist{step.rhs.family}
    elseif !isnothing(step.rhs.builder)
        step.rhs.builder
    else
        getfield(@__MODULE__, step.rhs.family)
    end
    parameter_value_indices =
        isnothing(step.parameter_slot) ? nothing : parametervalueindices(parameter_layout.slots[step.parameter_slot])
    noncentered = nothing
    if step.rhs isa DistributionSpec && step.rhs.reparam === :noncentered
        if step.rhs.family === :iid
            base = step.rhs.arguments[1]
            location_index, scale_index = _noncentered_location_scale_indices(base.args[1])
            noncentered = CompiledNoncentered(
                _compile_plan_expr(model, layout, base.args[location_index+1]),
                _compile_plan_expr(model, layout, base.args[scale_index+1]),
                base.args[1] === :lognormal,
            )
        else
            location_index, scale_index = _noncentered_location_scale_indices(step.rhs.family)
            noncentered = CompiledNoncentered(
                arguments[location_index],
                arguments[scale_index],
                step.rhs.family === :lognormal,
            )
        end
    end
    marginalize = nothing
    if step.rhs isa DistributionSpec && step.rhs.marginalize === :enumerate
        marginalize = CompiledMarginalize(_marginalize_support(step.rhs))
    end
    address = _compile_address(layout, model, step.address)
    stage_index, stage_iterator_slot =
        _stage_observation_marker(address, step.parameter_slot, marginalize, loop_iterator_slots, stage_counter)
    # Static single-address VECTOR observation (issue #288): the whole observed
    # vector lives at one all-literal address (`{:y} ~ gaussianprocess(...)`,
    # `{:y} ~ poisson.(...)`, ...). It stages like a loop site but with the
    # sentinel iterator slot -1 (the emitted scorer reads the dense site vector
    # whole instead of indexing it), which puts GP/HMM hyperparameter models and
    # the broadcast GLMs on the generated-scorer path -- and therefore on the
    # Enzyme reverse-mode path (#268). The conservative constructor allowlist
    # keeps custom builders and transform-coupled vector families interpreted.
    if stage_index == 0 && isnothing(step.parameter_slot) && isnothing(marginalize) &&
       all(part -> part isa CompiledAddressLiteralPart, address.parts) &&
       _static_vector_obs_constructor(constructor)
        stage_counter[] += 1
        stage_index, stage_iterator_slot = stage_counter[], -1
    end
    return CompiledChoicePlanStep(
        step.binding_slot,
        address,
        constructor,
        arguments,
        parameter_value_indices,
        step.parameter_slot,
        noncentered,
        marginalize,
        stage_index,
        stage_iterator_slot,
    )
end

function _marginalize_support(rhs::DistributionSpec)
    rhs.family === :bernoulli && return (false, true)
    if rhs.family === :categorical
        probabilities = rhs.arguments[1]
        support_size = if probabilities isa Expr && probabilities.head == :vect
            length(probabilities.args)
        elseif probabilities isa AbstractVector
            length(probabilities)
        else
            throw(
                ArgumentError(
                    "marginalize=:enumerate requires a literal categorical probability vector, got `$probabilities`",
                ),
            )
        end
        return ntuple(identity, support_size)
    end
    throw(ArgumentError("marginalize=:enumerate is not supported for family `$(rhs.family)`"))
end

function _compile_plan_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::DeterministicPlanStep,
    loop_iterator_slots::Tuple{Vararg{Int}},
    stage_counter::Base.RefValue{Int},
)
    return CompiledDeterministicPlanStep(step.binding_slot, _compile_plan_expr(model, layout, step.expr))
end

function _compile_plan_step(
    @nospecialize(model::TeaModel),
    layout::EnvironmentLayout,
    parameter_layout::ParameterLayout,
    step::LoopPlanStep,
    loop_iterator_slots::Tuple{Vararg{Int}},
    stage_counter::Base.RefValue{Int},
)
    body_slots = (loop_iterator_slots..., step.iterator_slot)
    body = tuple((
        _compile_plan_step(model, layout, parameter_layout, inner, body_slots, stage_counter) for inner in step.body
    )...)
    return CompiledLoopPlanStep(step.iterator_slot, _compile_plan_expr(model, layout, step.iterable), body)
end

# A dynamic-mode model body may contain if/else control flow that the linear
# execution plan cannot represent (the plan appends BOTH branches' choices and
# assignments). generate/assess execute the model body directly and stay
# correct, but every compiled scoring path must refuse the linearized plan.
function _reject_branchful_compiled_scoring(model::TeaModel)
    model.branchful || return nothing
    throw(
        ArgumentError(
            "model `$(model.name)` contains branchful control flow (`if`/`else`, ternary, or a " *
            "short-circuit `&&`/`||` with a choice or assignment inside); the compiled scoring " *
            "paths (logjoint, gradients, batched and device evaluation) linearize the execution " *
            "plan and would score both branches unconditionally -- use `generate`/`assess`, which " *
            "execute the model body directly",
        ),
    )
end

function _compile_execution_plan(@nospecialize(model::TeaModel), raw_plan::ExecutionPlan)
    stage_counter = Ref(0)
    compiled_steps = tuple(
        (
            _compile_plan_step(model, raw_plan.environment_layout, raw_plan.parameter_layout, step, (), stage_counter) for
            step in raw_plan.steps
        )...,
    )
    return CompiledExecutionPlan(compiled_steps, _required_walk_slots(compiled_steps), stage_counter[])
end

_compile_execution_plan(@nospecialize(model::TeaModel)) = _compile_execution_plan(model, executionplan(model))

# Serializes the lazy plan memoizations (`model.evaluator_cache` below and
# `model.signature_cache` in `_resolve_signature_plan`) so concurrent inference
# -- e.g. `hmc_chains`/`nuts_chains` running chains on threads -- cannot insert
# into a memo Dict while another thread reads it. One process-wide reentrant
# lock: compilation is rare (once per model / per signature) and the guarded
# lookup is cheap next to a logjoint evaluation.
const _PLAN_MEMO_LOCK = ReentrantLock()

function _compiled_execution_plan(@nospecialize(model::TeaModel))
    _reject_branchful_compiled_scoring(model)
    return lock(_PLAN_MEMO_LOCK) do
        cached = model.evaluator_cache[]
        if isnothing(cached)
            cached = _compile_execution_plan(model)
            model.evaluator_cache[] = cached
        end
        cached::CompiledExecutionPlan
    end
end

# --- conditioning-signature resolution (issue #95) ---------------------------
#
# The compiled scoring path classifies latents/observations from the CONDITIONING
# SIGNATURE (docs/constraint-driven-conditioning.md): the set of constrained
# addresses that name a static, unscoped choice. The signature-specific execution
# plan (and its parameter layout) is memoized per `(model, signature)` in the
# model's `signature_cache`, so re-running inference with new data at the same
# observed addresses reuses the compiled plan.

mutable struct ResolvedSignaturePlan
    plan::ExecutionPlan
    compiled::CompiledExecutionPlan
    # Backend lowering of `plan`, lowered lazily on first batched/device use and
    # cached here (this struct is memoized in `model.signature_cache`). Holds a
    # `BackendLoweringResult` once populated; `nothing` until then. Untyped so
    # `evaluator.jl` need not see the backend types (defined later, in backend.jl).
    backend_lowering::Base.RefValue{Any}
    # Memoized `ForwardDiff.GradientConfig`s for the public
    # `logjoint_gradient_unconstrained` (issue #145), keyed by
    # `(objective type, eltype, length)` so a config is only reused for the
    # closure/chunk shape it was built for. Holds a `Dict` once populated;
    # `nothing` until then. Untyped for the same reason as `backend_lowering`.
    gradient_config_cache::Base.RefValue{Any}
    # Single-slot memo of the last dense `ObservationStage` built by the public
    # `logjoint_gradient_unconstrained` (issue #145): repeated calls with the
    # SAME (unmutated) constraints object skip the staging walk. Validated per
    # call via `_stage_is_current`, so a different or mutated ChoiceMap simply
    # rebuilds. `nothing` until first populated.
    observation_stage_cache::Base.RefValue{Any}
    # Lazily emitted type-stable straight-line scorers (issue #144), keyed by
    # the reject-mode flag. Holds a `_GeneratedScorerCache` once analyzed;
    # `nothing` until then. See src/generated_scorer.jl. Untyped so this struct
    # need not see the generated-scorer types.
    generated_scorer_cache::Base.RefValue{Any}
    # Single-slot memo of the dense obs vectors and their sufficient statistics
    # (issue #138) derived from the last memoized `ObservationStage` in the
    # public `logjoint_gradient_unconstrained`. Holds `(stage, obs, stats)`;
    # reused while the same stage object is current, so repeated calls on an
    # unmutated ChoiceMap skip the O(observations) statistics scan and the
    # entry point becomes O(1) in the observation count. `nothing` until first
    # populated. Untyped for the same reason as the other caches.
    generated_obs_stats_cache::Base.RefValue{Any}
end

ResolvedSignaturePlan(plan::ExecutionPlan, compiled::CompiledExecutionPlan) = ResolvedSignaturePlan(
    plan,
    compiled,
    Ref{Any}(nothing),
    Ref{Any}(nothing),
    Ref{Any}(nothing),
    Ref{Any}(nothing),
    Ref{Any}(nothing),
)

# A representative single ChoiceMap for the conditioning signature. The batched
# and device paths accept either a shared ChoiceMap or a per-column vector of
# ChoiceMaps; the parameter layout is static per signature, so every column of a
# batch shares one signature and the first entry is representative.
_representative_constraints(constraints::ChoiceMap) = constraints
_representative_constraints(constraints::AbstractVector) =
    isempty(constraints) ? choicemap() : first(constraints)

# The observed set is the canonical signature: the normalized addresses of the
# model's static, unscoped choices that are present in `constraints`. Values are
# never part of it, so it is a value-independent memoization key. Constrained
# addresses that do not name a static choice (templated loop indices, unknown
# addresses) never carry a parameter slot and so do not enter the signature.
function _conditioning_signature(model::TeaModel, constraints::ChoiceMap)
    observed = Set{Address}()
    for step in executionplan(model).steps
        step isa ChoicePlanStep || continue
        (isempty(step.scopes) && isstaticaddress(step.address)) || continue
        address = _static_choice_address(step)
        found, _ = _choice_tryget_normalized(constraints, address)
        found && push!(observed, address)
    end
    return observed
end

function _resolve_signature_plan(@nospecialize(model::TeaModel), signature::Set{Address})
    _reject_branchful_compiled_scoring(model)
    return lock(_PLAN_MEMO_LOCK) do
        cache = model.signature_cache[]
        if isnothing(cache)
            cache = Dict{Set{Address},ResolvedSignaturePlan}()
            model.signature_cache[] = cache
        end
        store = cache::Dict{Set{Address},ResolvedSignaturePlan}
        existing = get(store, signature, nothing)
        isnothing(existing) && begin
            plan = _signature_execution_plan(executionplan(model), signature)
            existing = ResolvedSignaturePlan(plan, _compile_execution_plan(model, plan))
            store[signature] = existing
        end
        existing
    end
end

# --- misconditioning guard (issue #310) --------------------------------------
#
# A constraint address that matches NO model choice is silently dropped by the
# conditioning rule: the intended observation stays a latent and sampling
# quietly targets the prior -- the worst silent failure a typo can produce
# (`choicemap((:yy, 0.3))` against a model observing `:y`). Warn on the first
# encounter of a conditioning signature; the check is static template matching
# against the execution plan (no model execution), and running it only on the
# signature-cache MISS keeps it off every hot path.

# The model's static choice-address templates: one vector per choice, with
# `Some(value)` for literal parts and `nothing` for dynamic (loop-index) parts.
# Returns `nothing` when the plan contains shapes we cannot enumerate
# statically (generative subcalls), so the guard skips rather than risking
# false positives.
function _choice_address_templates(@nospecialize(model::TeaModel))
    templates = Vector{Vector{Union{Nothing,Some{Any}}}}()
    ok = _collect_address_templates!(templates, executionplan(model).steps)
    return ok ? templates : nothing
end

function _collect_address_templates!(templates, steps)
    for step in steps
        if step isa ChoicePlanStep
            step.rhs isa GenerativeCallSpec && return false
            parts = step.address.parts
            template = Vector{Union{Nothing,Some{Any}}}(undef, length(parts))
            for (index, part) in enumerate(parts)
                template[index] = part isa AddressLiteralPart ? Some{Any}(part.value) : nothing
            end
            push!(templates, template)
        elseif step isa LoopPlanStep
            _collect_address_templates!(templates, step.body) || return false
        end
    end
    return true
end

function _address_matches_template(address::Tuple, template)
    length(address) == length(template) || return false
    for (part, slot) in zip(address, template)
        slot === nothing && continue                    # dynamic part matches anything
        something(slot) == part || return false
    end
    return true
end

function _warn_unmatched_constraint_addresses(@nospecialize(model::TeaModel), constraints::ChoiceMap)
    isempty(constraints.entries) && return nothing
    templates = _choice_address_templates(model)
    isnothing(templates) && return nothing
    for entry in constraints.entries
        address = first(entry)
        address isa Tuple || continue
        any(template -> _address_matches_template(address, template), templates) && continue
        @warn "constraint address $(address) does not match any choice in model `$(model.name)` and will be IGNORED -- the intended observation (if any) stays a latent and sampling targets the prior. Check the address for typos; a loop observation is addressed as `(:y => i)`." _id =
            Symbol(:uncertaintea_unmatched_constraint_, model.name, :_, hash(address)) maxlog = 1
    end
    return nothing
end

function _resolve_signature_plan(model::TeaModel, constraints::ChoiceMap)
    signature = _conditioning_signature(model, constraints)
    # first-encounter validation only: peek the memo unlocked (a benign race
    # merely repeats the cheap static check)
    cache = model.signature_cache[]
    if isnothing(cache) || !haskey(cache::Dict{Set{Address},ResolvedSignaturePlan}, signature)
        _warn_unmatched_constraint_addresses(model, constraints)
    end
    return _resolve_signature_plan(model, signature)
end

# The signature-specific parameter layout for one conditioning. The CPU
# inference entry points size, initialize, and reconstruct against THIS layout
# (issue #95, PR-6), not the syntactic default `parameterlayout(model)`: the
# latent set is a function of which addresses are constrained, so a constrained
# bound choice drops its slot and an unconstrained unbound choice gains one.
_conditioned_parameter_layout(model::TeaModel, constraints::ChoiceMap) =
    _resolve_signature_plan(model, constraints).plan.parameter_layout

# Per-column conditioning (a chain/result may carry one ChoiceMap per column):
# every column shares the same signature, so the representative constraints fix
# the (static) layout, matching how `_batched_signature_layout` resolves it.
_conditioned_parameter_layout(model::TeaModel, constraints::AbstractVector) =
    _conditioned_parameter_layout(model, _representative_constraints(constraints))

# Human-readable description of a conditioning signature, used by the
# raw-parameter-vector APIs when a length check fails. The parameter-vector
# length is a function of the conditioning signature (which addresses are
# observed), not of the model alone (see docs/constraint-driven-conditioning.md),
# so a bare "expected N got M" hides the reason for the count; this names the
# observed addresses and the latent slots the conditioning implies.
function _describe_conditioning(model::TeaModel, layout::ParameterLayout, constraints::ChoiceMap)
    observed = sort!([string(address) for address in _conditioning_signature(model, constraints)])
    latents = [slot.binding === Symbol("") ? "<unbound>" : string(slot.binding) for slot in layout.slots]
    obs_desc =
        isempty(observed) ? "no addresses observed" :
        "$(length(observed)) observed: {$(join(observed, ", "))}"
    lat_desc =
        isempty(latents) ? "no latent slots" :
        "$(length(latents)) latent slot(s): [$(join(latents, ", "))]"
    return "$obs_desc; $lat_desc"
end

# DimensionMismatch for a signature-specific length check that names the
# conditioning so the required length is self-explanatory. `space` distinguishes
# unconstrained ("parameters") from constrained ("constrained-space parameters").
function _signature_length_error(
    model::TeaModel,
    layout::ParameterLayout,
    constraints::ChoiceMap,
    expected::Integer,
    got::Integer;
    space::AbstractString="parameters",
)
    return DimensionMismatch(
        "expected $expected $space for this conditioning " *
        "($(_describe_conditioning(model, layout, constraints))), got $got. The parameter-vector " *
        "length is conditioning-dependent (constraining or unconstraining an address changes it); " *
        "see docs/constraint-driven-conditioning.md.",
    )
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledLiteralExpr)
    return expr.value
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledSlotExpr)
    return _environment_value(env, expr.slot)
end

# Recursive tuple map over compiled argument tuples. A generator splat
# (`tuple((_eval_compiled_expr(env, a) for a in args)...)`) loses the tuple's
# element types and routes every call through dynamic apply-iterate machinery
# (~60% of logistic scoring samples, issue #145); the head/tail recursion
# keeps the argument tuple type concrete so calls specialize.
@inline _eval_compiled_exprs(env::PlanEnvironment, ::Tuple{}) = ()
@inline function _eval_compiled_exprs(env::PlanEnvironment, exprs::Tuple)
    return (_eval_compiled_expr(env, first(exprs)), _eval_compiled_exprs(env, Base.tail(exprs))...)
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledCallExpr)
    callee = _eval_compiled_expr(env, expr.callee)
    arguments = _eval_compiled_exprs(env, expr.arguments)
    return callee(arguments...)
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledTupleExpr)
    return _eval_compiled_exprs(env, expr.arguments)
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledVectorExpr)
    return Any[_eval_compiled_expr(env, arg) for arg in expr.arguments]
end

function _eval_compiled_expr(env::PlanEnvironment, expr::CompiledBlockExpr)
    value = nothing
    for arg in expr.arguments
        value = _eval_compiled_expr(env, arg)
    end
    return value
end

_concrete_compiled_address_parts(env::PlanEnvironment, ::Tuple{}) = ()

function _concrete_compiled_address_parts(env::PlanEnvironment, parts::Tuple)
    part = first(parts)
    head = if part isa CompiledAddressLiteralPart
        part.value
    else
        _eval_compiled_expr(env, part.expr)
    end
    return (head, _concrete_compiled_address_parts(env, Base.tail(parts))...)
end

function _concrete_address(env::PlanEnvironment, address::CompiledAddressSpec)
    return _normalize_concrete_address(_concrete_compiled_address_parts(env, address.parts))
end

function _construct_compiled_distribution(step::CompiledChoicePlanStep, env::PlanEnvironment)
    arguments = _eval_compiled_exprs(env, step.arguments)
    return step.constructor(arguments...)
end

# Returns the step's distribution, or `nothing` when the environment is in
# reject mode and the parameters are invalid (issue #157). The catch is narrow
# -- only the argument evaluation + constructor of this one choice step, and
# only the parameter-validation error types (`ArgumentError` from the CPU
# distribution constructors, `DomainError` from math in an argument
# expression) -- so structural errors (BoundsError, MethodError, dimension
# mismatches, ...) still propagate. Catching here instead of duplicating every
# family's validity predicate keeps registered custom distributions covered.
function _compiled_distribution(step::CompiledChoicePlanStep, env::PlanEnvironment)
    env.reject_invalid_parameters || return _construct_compiled_distribution(step, env)
    try
        return _construct_compiled_distribution(step, env)
    catch err
        (err isa ArgumentError || err isa DomainError) && return nothing
        rethrow()
    end
end

function _parameter_slot_value(layout::ParameterLayout, slot_index::Int, params::AbstractVector)
    slot = layout.slots[slot_index]
    indices = parametervalueindices(slot)
    length(indices) == 1 && return params[first(indices)]
    return collect(view(params, indices))
end

function _parameter_slot_value(indices::UnitRange{Int}, params::AbstractVector)
    length(indices) == 1 && return params[first(indices)]
    return collect(view(params, indices))
end

# --- dense observed-value staging (issue #145) --------------------------------
#
# A dense `Vector{Float64}` read costs ~0.5 ns; the equivalent per-observation
# ChoiceMap lookup assembles a `(literal..., i)` tuple and hashes it (~137 ns +
# 5 allocs). Constraints are bound once per LogjointGradientCache and assumed
# immutable for its lifetime, so the cache pre-resolves every stage-marked
# loop observation (`CompiledChoicePlanStep.stage_index`) into a dense
# per-site vector indexed by the loop index. Anything dynamic -- non-Int loop
# indices, non-Float64 values, indices the staging walk never visited -- falls
# back to the live ChoiceMap lookup, and the assumed immutability is verified
# per evaluation via the ChoiceMap mutation counter (see `_stage_is_current`).

struct StagedObservationSite
    values::Vector{Float64}
    filled::BitVector
end

struct ObservationStage
    sites::Vector{StagedObservationSite}
    constraints::ChoiceMap
    constraints_mutation_count::Int
end

# The staging contract: the stage may only be consulted for the exact
# constraints object it snapshot, and only while that object has not been
# mutated since (gibbs replaces discrete-site values in its merged constraints
# in place between gradient evaluations -- staged values would be stale).
_stage_is_current(stage::ObservationStage, constraints::ChoiceMap) =
    stage.constraints === constraints && stage.constraints_mutation_count == constraints.mutation_count

# Staged value for a stage-marked choice step, or `nothing` when the site (or
# this particular index) must fall back to the ChoiceMap.
@inline function _staged_observation_value(stage::ObservationStage, step::CompiledChoicePlanStep, env::PlanEnvironment)
    # static whole-vector sites (iterator slot -1, issue #288) are consumed only
    # by the generated scorer; the interpreter keeps its ChoiceMap lookup
    step.stage_iterator_slot == -1 && return nothing
    site = @inbounds stage.sites[step.stage_index]
    index = _environment_value(env, step.stage_iterator_slot)
    index isa Int || return nothing
    (1 <= index <= length(site.filled) && @inbounds(site.filled[index])) || return nothing
    return @inbounds site.values[index]
end

const _MaybeObservationStage = Union{Nothing,ObservationStage}

_score_compiled_steps(steps::Tuple, env::PlanEnvironment, params::AbstractVector, constraints::ChoiceMap) =
    _score_compiled_steps(steps, env, params, constraints, nothing)

_score_compiled_steps(
    ::Tuple{},
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
) = 0.0

function _score_compiled_steps(
    steps::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
)
    head = first(steps)
    tail = Base.tail(steps)
    if head isa CompiledChoicePlanStep && !isnothing(head.marginalize)
        return _score_marginalized_choice!(head, tail, env, params, constraints, stage)
    end
    return _score_plan_step!(head, env, params, constraints, stage) +
           _score_compiled_steps(tail, env, params, constraints, stage)
end

# marginalize=:enumerate (docs/discrete-enumeration.md): the fold above makes
# a marginalized choice own its suffix -- bind each compile-time support
# value, score the remaining steps, and logsumexp-combine, so the returned
# density is the marginal over the discrete latent. A constrained value
# short-circuits to the plain joint (conditioning on the latent stays free).
# Nested marginalized latents recurse through the suffix scoring (product
# enumeration).
function _score_marginalized_choice!(
    step::CompiledChoicePlanStep,
    tail::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
)
    address = _concrete_address(env, step.address)
    dist = _compiled_distribution(step, env)
    # reject mode (issue #157): invalid parameters make the whole marginal -Inf.
    # The marginalized choice owns its suffix, so returning here is complete.
    isnothing(dist) && return -Inf
    found, constrained_value = _choice_tryget_normalized(constraints, address)
    if found
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, constrained_value)
        return logpdf(dist, constrained_value) + _score_compiled_steps(tail, env, params, constraints, stage)
    end

    snapshot = _environment_snapshot(env)
    terms =
        _marginalized_suffix_terms(step.marginalize.support, snapshot, dist, step, tail, env, params, constraints, stage)
    _environment_restore_snapshot!(env, snapshot)

    # max-shifted logsumexp, mirroring `logpdf(::MixtureDist, x)`
    shift = maximum(terms)
    isfinite(shift) || return oftype(shift, -Inf)
    total = zero(shift)
    for term in terms
        total += exp(term - shift)
    end
    return shift + log(total)
end

function _marginalized_suffix_terms(
    ::Tuple{},
    snapshot::Tuple{Vector{Any},BitVector},
    dist,
    step::CompiledChoicePlanStep,
    tail::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
)
    return ()
end

function _marginalized_suffix_terms(
    support::Tuple,
    snapshot::Tuple{Vector{Any},BitVector},
    dist,
    step::CompiledChoicePlanStep,
    tail::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
)
    # every branch re-runs the suffix from the same pre-branch environment
    # (suffix rebinds like `x = x + 1` must not leak across branches)
    _environment_restore_snapshot!(env, snapshot)
    value = first(support)
    choice_logpdf = logpdf(dist, value)
    # a zero-mass support value (bernoulli(1.0) at false, a zero categorical
    # weight) contributes nothing to the marginal, and its suffix may be
    # unevaluable (branch-dependent invalid parameters) -- skip it, with
    # clean zero partials so an infinite pmf derivative cannot poison the
    # logsumexp gradient
    term = if isfinite(choice_logpdf)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        choice_logpdf + _score_compiled_steps(tail, env, params, constraints, stage)
    else
        oftype(choice_logpdf, -Inf)
    end
    return (
        term,
        _marginalized_suffix_terms(Base.tail(support), snapshot, dist, step, tail, env, params, constraints, stage)...,
    )
end

function _score_plan_step!(
    step::CompiledChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
)
    if !isnothing(stage) && step.stage_index != 0
        staged_value = _staged_observation_value(stage, step, env)
        if staged_value isa Float64
            dist = _compiled_distribution(step, env)
            isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, staged_value)
            # reject mode (issue #157): staged reads bypass constraint lookups,
            # not distribution construction -- an invalid parameter still
            # rejects the draw as -Inf here
            isnothing(dist) && return -Inf
            return logpdf(dist, staged_value)
        end
    end
    address = _concrete_address(env, step.address)
    value = if !isnothing(step.parameter_value_indices)
        _parameter_slot_value(step.parameter_value_indices, params)
    else
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        found || throw(ArgumentError("lower-level logjoint requires a provided value for choice $(address)"))
        constrained_value
    end

    dist = _compiled_distribution(step, env)
    # the binding still happens when the parameters reject (issue #157):
    # subsequent steps may reference the value, and it never depends on the
    # distribution
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    # reject mode (issue #157): a plain -Inf constant carries clean (zero)
    # ForwardDiff partials, so the lane's value is non-finite -- which the
    # leapfrog guards already turn into a per-chain divergence -- without
    # poisoning other lanes' gradients
    isnothing(dist) && return -Inf
    return logpdf(dist, value)
end

function _score_plan_step!(
    step::CompiledDeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
)
    _environment_set!(env, step.binding_slot, _eval_compiled_expr(env, step.expr))
    return 0.0
end

function _score_plan_step!(
    step::CompiledLoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage,
)
    iterable = _eval_compiled_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    total = 0.0

    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        total += _score_compiled_steps(step.body, env, params, constraints, stage)
    end

    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return total
end

# --- signature-aware prior draw (issue #156) ----------------------------------
#
# One prior draw of the SIGNATURE latents, walked over the compiled signature
# plan instead of a traced `generate`: the traced walk re-visits every
# observation address (normalize_address/_pushchoice! dominate with many
# observations), so drawing initial positions per chain paid O(observations)
# per chain before sampling started. The walk mirrors the traced execution
# site-for-site -- a constrained site reads its value from `constraints`
# without touching the RNG (exactly like `choice` in runtime.jl), an
# unconstrained site draws `rand(rng, dist)` from the same distribution in the
# same execution order -- so the drawn values are bitwise identical to the
# traced path, while trace recording, duplicate-address checks, and
# observation scoring are skipped.
function _sample_prior_steps!(
    ::Tuple{},
    env::PlanEnvironment,
    layout::ParameterLayout,
    params::AbstractVector,
    constraints::ChoiceMap,
    rng::AbstractRNG,
)
    return nothing
end

function _sample_prior_steps!(
    steps::Tuple,
    env::PlanEnvironment,
    layout::ParameterLayout,
    params::AbstractVector,
    constraints::ChoiceMap,
    rng::AbstractRNG,
)
    _sample_prior_step!(first(steps), env, layout, params, constraints, rng)
    return _sample_prior_steps!(Base.tail(steps), env, layout, params, constraints, rng)
end

function _sample_prior_step!(
    step::CompiledChoicePlanStep,
    env::PlanEnvironment,
    layout::ParameterLayout,
    params::AbstractVector,
    constraints::ChoiceMap,
    rng::AbstractRNG,
)
    address = _concrete_address(env, step.address)
    found, value = _choice_tryget_normalized(constraints, address)
    if !found
        value = rand(rng, _compiled_distribution(step, env))
    end
    isnothing(step.parameter_slot) || _write_slot_value!(params, layout.slots[step.parameter_slot], value)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return nothing
end

# --- staging walk (issue #145) -------------------------------------------------
#
# Build the dense per-site observation vectors by walking the compiled plan
# once, exactly like a scoring pass but recording constrained values instead of
# summing logpdfs (distributions are never constructed). Latent bindings come
# from the constrained-space `params`; a loop iterable that somehow yields
# different indices at other parameter values is harmless because scoring falls
# back to the ChoiceMap for any index the walk did not fill. Any anomaly aborts
# the walk (`nothing` -- full ChoiceMap fallback); a non-Int index or a
# non-Float64 value deactivates just that site.

function _stage_observations(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    unconstrained_seed::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    compiled = resolved.compiled
    compiled.stage_count == 0 && return nothing
    mutation_count = constraints.mutation_count
    sites = [StagedObservationSite(Float64[], BitVector()) for _ = 1:compiled.stage_count]
    active = trues(compiled.stage_count)
    try
        seed = collect(Float64, unconstrained_seed)
        constrained, _ = _transform_to_constrained_with_logabsdet(model, resolved, seed, args, constraints)
        complete_args = _complete_model_args(model, args)
        env = PlanEnvironment(resolved.plan.environment_layout)
        for (slot, value) in zip(resolved.plan.environment_layout.argument_slots, complete_args)
            _environment_set!(env, slot, value)
        end
        _stage_walk_steps!(sites, active, compiled.steps, env, constrained, constraints)
    catch
        return nothing
    end
    for site_index in eachindex(active)
        if !active[site_index]
            empty!(sites[site_index].values)
            empty!(sites[site_index].filled)
        end
    end
    return ObservationStage(sites, constraints, mutation_count)
end

function _stage_walk_steps!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    steps::Tuple,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    for step in steps
        _stage_walk_step!(sites, active, step, env, params, constraints)
    end
    return nothing
end

function _stage_walk_step!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    step::CompiledChoicePlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    value = if !isnothing(step.parameter_value_indices)
        _parameter_slot_value(step.parameter_value_indices, params)
    else
        address = _concrete_address(env, step.address)
        found, constrained_value = _choice_tryget_normalized(constraints, address)
        if found
            constrained_value
        elseif !isnothing(step.marginalize)
            # unconstrained enumerated latent: bind a representative support
            # value so the walk stays linear -- staged values are a pure
            # function of (site, loop index) and the fallback covers any
            # branch-dependent loop pattern
            first(step.marginalize.support)
        else
            # scoring would throw here; abort staging so it does
            throw(ArgumentError("staging walk found no value for choice $(address)"))
        end
    end
    if step.stage_index != 0 && active[step.stage_index] && step.stage_iterator_slot == -1
        # static whole-vector observation site (issue #288): the constrained
        # value is the dense Float64 vector itself (the same Float64-only rule
        # the loop sites apply; anything else deactivates the site)
        site = sites[step.stage_index]
        if value isa Vector{Float64} && !isempty(value)
            n = length(value)
            resize!(site.values, n)
            copyto!(site.values, value)
            resize!(site.filled, n)
            fill!(site.filled, true)
        else
            active[step.stage_index] = false
        end
    elseif step.stage_index != 0 && active[step.stage_index]
        index = _environment_value(env, step.stage_iterator_slot)
        if index isa Int && index >= 1 && value isa Float64
            site = sites[step.stage_index]
            if index > length(site.values)
                previous_length = length(site.values)
                resize!(site.values, index)
                resize!(site.filled, index)
                for unfilled = (previous_length+1):index
                    site.filled[unfilled] = false
                end
            end
            site.values[index] = value
            site.filled[index] = true
        else
            active[step.stage_index] = false
        end
    end
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return nothing
end

function _sample_prior_step!(
    step::CompiledDeterministicPlanStep,
    env::PlanEnvironment,
    layout::ParameterLayout,
    params::AbstractVector,
    constraints::ChoiceMap,
    rng::AbstractRNG,
)
    _environment_set!(env, step.binding_slot, _eval_compiled_expr(env, step.expr))
    return nothing
end

function _stage_walk_step!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    step::CompiledDeterministicPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    _environment_set!(env, step.binding_slot, _eval_compiled_expr(env, step.expr))
    return nothing
end

function _sample_prior_step!(
    step::CompiledLoopPlanStep,
    env::PlanEnvironment,
    layout::ParameterLayout,
    params::AbstractVector,
    constraints::ChoiceMap,
    rng::AbstractRNG,
)
    iterable = _eval_compiled_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing

    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        _sample_prior_steps!(step.body, env, layout, params, constraints, rng)
    end

    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return nothing
end

function _stage_walk_step!(
    sites::Vector{StagedObservationSite},
    active::BitVector,
    step::CompiledLoopPlanStep,
    env::PlanEnvironment,
    params::AbstractVector,
    constraints::ChoiceMap,
)
    iterable = _eval_compiled_expr(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        _stage_walk_steps!(sites, active, step.body, env, params, constraints)
    end
    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return nothing
end

function logjoint(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    resolved = _resolve_signature_plan(model, constraints)
    return _logjoint(model, resolved, params, args, constraints)
end

function _logjoint(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage=nothing;
    reject_invalid_parameters::Bool=false,
)
    plan = resolved.plan
    expected = parametervaluecount(plan.parameter_layout)
    length(params) == expected || throw(
        _signature_length_error(
            model,
            plan.parameter_layout,
            constraints,
            expected,
            length(params);
            space="constrained-space parameters",
        ),
    )
    args = _complete_model_args(model, args)

    env = PlanEnvironment(plan.environment_layout; reject_invalid_parameters=reject_invalid_parameters)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end

    # verify the staging assumption (immutable constraints per cache lifetime)
    # every evaluation: a mutated or different ChoiceMap silently drops back to
    # live lookups instead of scoring stale staged values
    active_stage = (stage isa ObservationStage && _stage_is_current(stage, constraints)) ? stage : nothing

    # Type-stable generated scorer (issue #144) for the scalar scoring path.
    # Any plan the generator or the dense observation-vector build cannot
    # represent (Float32/scalar/broadcast observations, marginalize, ...) leaves
    # `scorer`/`obs` as `nothing` and falls through to the interpreter, which
    # stays the numerical source of truth. `invokelatest` crosses the world-age
    # boundary from the (possibly just-emitted) scorer method; the scalar result
    # need not be type-inferred here.
    scorer = _generated_scorer(resolved, reject_invalid_parameters)
    if scorer !== nothing
        obs =
            isnothing(active_stage) ? _gen_obs_from_params(resolved, params, args, constraints) :
            _gen_obs_from_stage(active_stage)
        if !isnothing(obs)
            _GEN_SCORER_LAST_USED[] = true
            # sufficient statistics for fusable observation loops (issue #138),
            # computed from the dense obs data alone (empty for non-fusable
            # stages / data the closed form cannot represent); the scorer reads
            # them in O(1) for a fusable loop and ignores them otherwise
            stats = _gen_stats(resolved, obs)
            return Base.invokelatest(scorer, args, params, obs, stats)
        end
    end
    _GEN_SCORER_LAST_USED[] = false
    return _score_compiled_steps(resolved.compiled.steps, env, params, constraints, active_stage)
end

# `reject_invalid_parameters=true` selects Stan-style reject semantics (issue
# #157): invalid distribution parameters yield -Inf instead of throwing. The
# samplers evaluate with it on; the default keeps the public throwing contract.
function logjoint_unconstrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    reject_invalid_parameters::Bool=false,
)
    resolved = _resolve_signature_plan(model, constraints)
    return _logjoint_unconstrained(
        model, resolved, params, args, constraints, nothing;
        reject_invalid_parameters=reject_invalid_parameters,
    )
end

function _logjoint_unconstrained(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage;
    reject_invalid_parameters::Bool=false,
)
    constrained, logabsdet = _transform_to_constrained_with_logabsdet(
        model, resolved, params, args, constraints;
        reject_invalid_parameters=reject_invalid_parameters,
    )
    return _logjoint(
        model,
        resolved,
        constrained,
        args,
        constraints,
        stage;
        reject_invalid_parameters=reject_invalid_parameters,
    ) + logabsdet
end

# --- dependent-transform plan walk (reparam=:noncentered) ---------------------
#
# Noncentered slots hold the standardized z while the constrained space keeps
# theta = location + scale * z, where location/scale are expressions over
# model arguments and earlier latents. The transform therefore runs as a walk
# over the compiled plan with an environment, visiting slots in execution
# order; centered slots use the ordinary per-slot transforms. Observations
# are bound as NaN poison: a location/scale that depends on one fails loudly.

# positions of (location, scale) among the family's arguments
_noncentered_location_scale_indices(family::Symbol) = family === :studentt ? (2, 3) : (1, 2)

# Poison bound to slotless choices during the walk. Unlike NaN it cannot be
# swallowed by comparisons or branching (`NaN > 0` is silently false): it is
# not a Number, so arithmetic and ordered comparisons raise MethodError, and
# equality is overloaded to throw outright.
struct _TransformUnknownValue end

const _TRANSFORM_UNKNOWN_MESSAGE =
    "reparam=:noncentered location/scale expressions may only depend on model arguments, " *
    "earlier latents with parameter slots, and observed (constrained) values; this model " *
    "routes a choice with no fixed value during the change of variables -- a " *
    "marginalize=:enumerate discrete latent or a loop-scoped binding -- into a noncentered " *
    "location/scale"

Base.:(==)(::_TransformUnknownValue, ::Any) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.:(==)(::Any, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.:(==)(::_TransformUnknownValue, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::_TransformUnknownValue, ::Any) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::Any, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
Base.isequal(::_TransformUnknownValue, ::_TransformUnknownValue) = throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))

# Evaluate an expression during the walk, converting the poison's MethodError
# into the informative rejection.
function _walk_transform_eval(env::PlanEnvironment, expr)
    try
        return _eval_compiled_expr(env, expr)
    catch err
        if err isa MethodError && any(arg isa _TransformUnknownValue for arg in err.args)
            throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
        end
        rethrow()
    end
end

# --- reachability: which bindings a noncentered loc/scale actually needs ------
#
# Only expressions reachable from a reparam=:noncentered location/scale take part
# in the change of variables, so only those deterministic steps and loops need to
# run during the transform walk (issue #100). We compute, over the compiled plan,
# the set of environment slots a noncentered loc/scale transitively depends on;
# the walk then evaluates a deterministic step (or a loop) only when it populates
# one of those slots. A step that a poisoned (slotless) binding feeds but that
# never reaches a noncentered loc/scale is left unevaluated -- exactly the steps
# whose eager evaluation used to throw spuriously.

_collect_slot_refs!(::Set{Int}, ::CompiledLiteralExpr) = nothing
function _collect_slot_refs!(acc::Set{Int}, expr::CompiledSlotExpr)
    push!(acc, expr.slot)
    return nothing
end
function _collect_slot_refs!(acc::Set{Int}, expr::CompiledCallExpr)
    _collect_slot_refs!(acc, expr.callee)
    for arg in expr.arguments
        _collect_slot_refs!(acc, arg)
    end
    return nothing
end
function _collect_slot_refs!(acc::Set{Int}, expr::Union{CompiledTupleExpr,CompiledVectorExpr,CompiledBlockExpr})
    for arg in expr.arguments
        _collect_slot_refs!(acc, arg)
    end
    return nothing
end
# CompiledNoncentered stores loc/scale untyped; dispatch resolves on the concrete
# compiled-expr subtype at runtime.
_collect_slot_refs!(acc::Set{Int}, expr) = nothing

# Seed = slots read directly by any noncentered loc/scale; `defs` maps a slot to
# the slots the step producing it reads (deterministic bindings and loop
# iterators), recursing into loop bodies.
_collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, steps::Tuple) =
    (foreach(step -> _collect_walk_dependencies!(seed, defs, step), steps); nothing)
function _collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, step::CompiledDeterministicPlanStep)
    refs = Set{Int}()
    _collect_slot_refs!(refs, step.expr)
    defs[step.binding_slot] = refs
    return nothing
end
function _collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, step::CompiledChoicePlanStep)
    if !isnothing(step.noncentered)
        _collect_slot_refs!(seed, step.noncentered.location)
        _collect_slot_refs!(seed, step.noncentered.scale)
    end
    return nothing
end
function _collect_walk_dependencies!(seed::Set{Int}, defs::Dict{Int,Set{Int}}, step::CompiledLoopPlanStep)
    refs = Set{Int}()
    _collect_slot_refs!(refs, step.iterable)
    defs[step.iterator_slot] = refs
    _collect_walk_dependencies!(seed, defs, step.body)
    return nothing
end

function _required_walk_slots(steps::Tuple)
    seed = Set{Int}()
    defs = Dict{Int,Set{Int}}()
    _collect_walk_dependencies!(seed, defs, steps)
    required = Set{Int}()
    isempty(seed) && return required
    worklist = collect(seed)
    while !isempty(worklist)
        slot = pop!(worklist)
        slot in required && continue
        push!(required, slot)
        edges = get(defs, slot, nothing)
        isnothing(edges) && continue
        for dep in edges
            dep in required || push!(worklist, dep)
        end
    end
    return required
end

# Whether walking this step (or its loop subtree) can populate a required slot.
_subtree_defines_required(step::CompiledDeterministicPlanStep, required::Set{Int}) =
    step.binding_slot in required
_subtree_defines_required(step::CompiledChoicePlanStep, required::Set{Int}) =
    !isnothing(step.binding_slot) && step.binding_slot in required
function _subtree_defines_required(step::CompiledLoopPlanStep, required::Set{Int})
    step.iterator_slot in required && return true
    for inner in step.body
        _subtree_defines_required(inner, required) && return true
    end
    return false
end

function _dependent_transform_walk!(
    destination::AbstractVector,
    model::TeaModel,
    plan::ExecutionPlan,
    compiled_plan::CompiledExecutionPlan,
    params::AbstractVector,
    args::Tuple,
    inverse::Bool,
    constraints::ChoiceMap=choicemap();
    reject_invalid_parameters::Bool=false,
)
    args = _complete_model_args(model, args)
    env = PlanEnvironment(plan.environment_layout; reject_invalid_parameters=reject_invalid_parameters)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end
    return _walk_transform_steps!(
        destination,
        compiled_plan.steps,
        env,
        plan.parameter_layout,
        params,
        inverse,
        constraints,
        compiled_plan.required_walk_slots,
    )
end

function _dependent_transform_walk!(
    destination::AbstractVector,
    model::TeaModel,
    params::AbstractVector,
    args::Tuple,
    inverse::Bool,
    constraints::ChoiceMap=choicemap();
    reject_invalid_parameters::Bool=false,
)
    return _dependent_transform_walk!(
        destination,
        model,
        executionplan(model),
        _compiled_execution_plan(model),
        params,
        args,
        inverse,
        constraints;
        reject_invalid_parameters=reject_invalid_parameters,
    )
end

# Signature-aware constrained transform used by `logjoint_unconstrained`: it
# transforms against the resolved signature layout so the unconstrained
# parameter vector and the scored plan agree on which choices are latent.
function _transform_to_constrained_with_logabsdet(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap=choicemap();
    reject_invalid_parameters::Bool=false,
)
    layout = resolved.plan.parameter_layout
    expected = parametercount(layout)
    length(params) == expected ||
        throw(_signature_length_error(model, layout, constraints, expected, length(params)))

    constrained = similar(params, parametervaluecount(layout))
    if _has_dependent_transforms(layout)
        logabsdet = _dependent_transform_walk!(
            constrained,
            model,
            resolved.plan,
            resolved.compiled,
            params,
            args,
            false,
            constraints;
            reject_invalid_parameters=reject_invalid_parameters,
        )
        return constrained, logabsdet
    end
    logabsdet = expected == 0 ? 0.0 : zero(params[firstindex(params)])
    for slot in layout.slots
        logabsdet += _transform_slot_to_constrained!(constrained, slot, params)
    end
    return constrained, logabsdet
end

_walk_transform_steps!(destination, ::Tuple{}, env, layout, params, inverse, constraints, required) =
    zero(eltype(destination))

function _walk_transform_steps!(destination, steps::Tuple, env, layout, params, inverse, constraints, required)
    return _walk_transform_step!(destination, first(steps), env, layout, params, inverse, constraints, required) +
           _walk_transform_steps!(destination, Base.tail(steps), env, layout, params, inverse, constraints, required)
end

function _walk_transform_step!(
    destination,
    step::CompiledDeterministicPlanStep,
    env,
    layout,
    params,
    inverse,
    constraints,
    required,
)
    # Only evaluate when the binding feeds a noncentered loc/scale (issue #100);
    # otherwise the step is irrelevant to the change of variables and evaluating
    # it would poison the walk on unrelated slotless choices.
    if step.binding_slot in required
        _environment_set!(env, step.binding_slot, _walk_transform_eval(env, step.expr))
    end
    return zero(eltype(destination))
end

function _walk_transform_step!(destination, step::CompiledLoopPlanStep, env, layout, params, inverse, constraints, required)
    # Skip the loop entirely unless walking it populates a required binding; a
    # loop-scoped binding never reaches a noncentered loc/scale otherwise.
    _subtree_defines_required(step, required) || return zero(eltype(destination))
    iterable = _walk_transform_eval(env, step.iterable)
    had_previous = _environment_hasvalue(env, step.iterator_slot)
    previous_value = had_previous ? _environment_value(env, step.iterator_slot) : nothing
    total = zero(eltype(destination))
    for item in iterable
        _environment_set!(env, step.iterator_slot, item)
        total += _walk_transform_steps!(destination, step.body, env, layout, params, inverse, constraints, required)
    end
    _environment_restore!(env, step.iterator_slot, previous_value, had_previous)
    return total
end

# Reject-mode handling for a noncentered step whose location/scale evaluated to a
# non-finite Real (issue #202). Mirrors `_compiled_distribution` returning
# `nothing` (scored as -Inf): the change of variables cannot be evaluated here, so
# this step contributes -Inf to the total unconstrained log-joint, which the
# sampler already treats as a divergent/rejected proposal. The destination slot is
# still filled with a finite placeholder so the downstream (reject-mode) scoring
# walk reads valid memory and returns a finite value -- `finite + (-Inf)` is -Inf,
# whereas leaving the non-finite affine in place could make it `NaN`. The binding
# is left unset: a genuinely-unknown binding is only consulted when it feeds
# another noncentered loc/scale (it would be in `required` and poison loudly), and
# the -Inf already forces rejection.
function _reject_noncentered_transform_step!(destination, slot, inverse)
    placeholder = zero(eltype(destination))
    indices = inverse ? parameterindices(slot) : parametervalueindices(slot)
    for index in indices
        destination[index] = placeholder
    end
    return -oftype(placeholder, Inf)
end

function _walk_transform_step!(destination, step::CompiledChoicePlanStep, env, layout, params, inverse, constraints, required)
    if isnothing(step.parameter_slot)
        # Slotless choice. Unified value resolution (issue #95, doc section 2):
        # an OBSERVATION resolves to its constrained value and is bound so
        # downstream deterministic steps and noncentered loc/scale expressions
        # see the real value; a genuinely-unknown slotless latent (a
        # marginalized discrete site) has no single value during the transform
        # and is poisoned so any dependence fails loudly.
        if !isnothing(step.binding_slot)
            address = _concrete_address(env, step.address)
            found, constrained_value = _choice_tryget_normalized(constraints, address)
            _environment_set!(env, step.binding_slot, found ? constrained_value : _TransformUnknownValue())
        end
        return zero(eltype(destination))
    end
    slot = layout.slots[step.parameter_slot]
    if isnothing(step.noncentered)
        if inverse
            _transform_slot_to_unconstrained!(destination, slot, params)
            value = _parameter_slot_value(parametervalueindices(slot), params)
            isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
            return zero(eltype(destination))
        end
        logabsdet = _transform_slot_to_constrained!(destination, slot, params)
        value = _parameter_slot_value(parametervalueindices(slot), destination)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        return logabsdet
    end

    location = _walk_transform_eval(env, step.noncentered.location)
    scale = _walk_transform_eval(env, step.noncentered.scale)
    (location isa _TransformUnknownValue || scale isa _TransformUnknownValue) &&
        throw(ArgumentError(_TRANSFORM_UNKNOWN_MESSAGE))
    if !(location isa Real && scale isa Real && isfinite(location) && isfinite(scale))
        # A non-finite (Inf/NaN) location/scale is a #157-class parameter-VALUE
        # failure: a leapfrog trajectory on a funnel/heavy-tailed model overshoots
        # to where a finite unconstrained tau maps to a non-finite constrained one,
        # and the noncentered change of variables can no longer be evaluated. In
        # reject mode (sampler-owned workspaces) treat it exactly like
        # `_compiled_distribution` treats an invalid distribution: drive the
        # log-joint to -Inf so the sampler rejects the proposal as divergent,
        # instead of throwing out of the batched gradient. Gated strictly on
        # `env.reject_invalid_parameters`, and -- mirroring the narrowness of the
        # #157 catch (ArgumentError/DomainError only, never structural errors) --
        # only when both are Reals, so a non-Real loc/scale (a genuine structural
        # bug) still throws. Outside reject mode the check throws unchanged, so it
        # keeps catching real model bugs.
        if env.reject_invalid_parameters && location isa Real && scale isa Real
            return _reject_noncentered_transform_step!(destination, slot, inverse)
        end
        throw(
            ArgumentError(
                "reparam=:noncentered location/scale must evaluate to finite reals; got " *
                "location=$location, scale=$scale for the choice bound to :$(slot.binding)",
            ),
        )
    end
    logspace = step.noncentered.logspace
    if inverse
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            constrained_value = params[value_index]
            unconstrained_value = logspace ? log(constrained_value) : constrained_value
            destination[parameter_index] = (unconstrained_value - location) / scale
        end
        value = _parameter_slot_value(parametervalueindices(slot), params)
        isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
        return zero(eltype(destination))
    end
    logabsdet = zero(eltype(destination))
    for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
        affine = location + scale * params[parameter_index]
        destination[value_index] = logspace ? exp(affine) : affine
        logabsdet += log(scale) + (logspace ? affine : zero(affine))
    end
    value = _parameter_slot_value(parametervalueindices(slot), destination)
    isnothing(step.binding_slot) || _environment_set!(env, step.binding_slot, value)
    return logabsdet
end

struct LogjointGradientCache{F,C,V}
    objective::F
    config::C
    buffer::V
end

# The gradient objective binds the resolved signature plan and the dense
# observation stage once, so per-evaluation work is the compiled scoring walk
# alone (no signature re-resolution, no per-observation ChoiceMap hashing for
# staged sites). Building it through a named helper keeps the closure type a
# pure function of the captured types, which the public gradient-config
# memoization keys on.
function _gradient_objective(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    args::Tuple,
    constraints::ChoiceMap,
    stage::_MaybeObservationStage;
    reject_invalid_parameters::Bool=false,
)
    return theta -> _logjoint_unconstrained(
        model, resolved, theta, args, constraints, stage;
        reject_invalid_parameters=reject_invalid_parameters,
    )
end

function _logjoint_gradient_cache(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
    buffer::AbstractVector=similar(collect(params));
    reject_invalid_parameters::Bool=false,
)
    seed = collect(params)
    length(buffer) == length(seed) ||
        throw(DimensionMismatch("expected gradient buffer of length $(length(seed)), got $(length(buffer))"))
    resolved = _resolve_signature_plan(model, constraints)
    stage = _stage_observations(model, resolved, seed, args, constraints)
    # Prefer the type-stable generated scorer (issue #144) when the plan and its
    # dense observation vectors are representable; otherwise the interpreter
    # objective. The choice is fixed for the cache's lifetime, so its objective
    # type -- and the `LogjointGradientCache{F}` -- stays concrete.
    scorer = _generated_scorer(resolved, reject_invalid_parameters)
    obs = isnothing(scorer) ? nothing : _gen_obs_from_stage(stage)
    objective = if !isnothing(scorer) && !isnothing(obs)
        _gen_gradient_objective(
            scorer, model, resolved, _complete_model_args(model, args), constraints, seed, obs;
            reject=reject_invalid_parameters,
        )
    else
        _gradient_objective(
            model, resolved, args, constraints, stage;
            reject_invalid_parameters=reject_invalid_parameters,
        )
    end
    config = ForwardDiff.GradientConfig(objective, seed)
    return LogjointGradientCache(objective, config, buffer)
end

function _logjoint_gradient!(cache::LogjointGradientCache, params::AbstractVector)
    # The generated objective's scorer method is emitted after this caller's
    # world (see generated_scorer.jl); `invokelatest` runs the differentiation
    # in the latest world so the concrete scorer resolves and stays type-stable.
    # The interpreter objective keeps its direct, world-stable call path. A
    # cache reused across an in-place constraint mutation (gibbs) re-stages its
    # dense observations first; if the mutated constraints are no longer
    # densifiable, it falls back to the interpreter for that call.
    if cache.objective isa _GenGradientObjective
        if _gen_refresh_obs!(cache.objective)
            Base.invokelatest(ForwardDiff.gradient!, cache.buffer, cache.objective, params, cache.config)
        else
            Base.invokelatest(_gen_interpreter_gradient!, cache.objective, cache.buffer, params)
        end
    else
        ForwardDiff.gradient!(cache.buffer, cache.objective, params, cache.config)
    end
    return cache.buffer
end

# Memoized `ForwardDiff.GradientConfig` for the public gradient entry point
# (issue #145): the config (seed duals + chunk buffers, ~80 kB for the audit
# models) depends only on the objective's type, the seed eltype, and the
# parameter count, all of which are stable per resolved signature -- not on
# the constraint or argument VALUES -- so it lives on the memoized
# `ResolvedSignaturePlan` instead of being rebuilt per call.
function _cached_gradient_config(resolved::ResolvedSignaturePlan, objective, seed::AbstractVector)
    store = resolved.gradient_config_cache[]
    if isnothing(store)
        store = Dict{Tuple{DataType,DataType,Int},Any}()
        resolved.gradient_config_cache[] = store
    end
    dict = store::Dict{Tuple{DataType,DataType,Int},Any}
    key = (typeof(objective), eltype(seed), length(seed))
    config = get(dict, key, nothing)
    isnothing(config) || return config
    config = ForwardDiff.GradientConfig(objective, seed)
    dict[key] = config
    return config
end

# Stage memo for the public gradient entry point: reuse the last stage while
# it is still current (same constraints object, no mutations since); rebuild
# otherwise. A cache miss costs one plan walk -- about half a gradient
# evaluation -- and every hit removes the per-observation ChoiceMap lookups
# from every subsequent call.
function _memoized_observation_stage(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    seed::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    resolved.compiled.stage_count == 0 && return nothing
    cached = resolved.observation_stage_cache[]
    if cached isa ObservationStage && _stage_is_current(cached, constraints)
        return cached
    end
    stage = _stage_observations(model, resolved, seed, args, constraints)
    isnothing(stage) || (resolved.observation_stage_cache[] = stage)
    return stage
end

function logjoint_gradient_unconstrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    seed = collect(params)
    resolved = _resolve_signature_plan(model, constraints)
    stage = _memoized_observation_stage(model, resolved, seed, args, constraints)
    scorer = _generated_scorer(resolved, false)
    obs_stats = isnothing(scorer) ? nothing : _gen_obs_and_stats_for_stage(resolved, stage)
    if !isnothing(scorer) && !isnothing(obs_stats)
        obs, stats = obs_stats
        _GEN_SCORER_LAST_USED[] = true
        objective = _gen_gradient_objective(
            scorer, model, resolved, _complete_model_args(model, args), constraints, seed, obs; stats=stats,
        )
        config = _cached_gradient_config(resolved, objective, seed)
        gradient = similar(seed)
        Base.invokelatest(ForwardDiff.gradient!, gradient, objective, seed, config)
        return gradient
    end
    _GEN_SCORER_LAST_USED[] = false
    objective = _gradient_objective(model, resolved, args, constraints, stage)
    config = _cached_gradient_config(resolved, objective, seed)
    gradient = similar(seed)
    ForwardDiff.gradient!(gradient, objective, seed, config)
    return gradient
end

# Reverse-mode gradient of a model's unconstrained logjoint (issue #268, part A).
#
# The forward-mode `logjoint_gradient_unconstrained` above is `O(P)` in the
# parameter count; on high-dimensional non-analytic models a reverse-mode pass is
# `O(1)`. The type-stable GENERATED scorer (`_GenGradientObjective`, the same
# objective the forward path differentiates) is Enzyme-differentiable, so the
# reverse-mode gradient is exactly the forward-mode gradient computed the cheaper
# way — no separate evaluator. The interpreter fallback objective is NOT
# type-stable enough for Enzyme, so a model that falls off the generated-scorer
# path is rejected with a clear message rather than silently taking a slow path.
#
# The actual Enzyme call lives in `reverse_mode_gradient(f, x)`, supplied by
# UncertainTeaEnzymeExt (loaded via `using Enzyme`); it is reached through
# `invokelatest` because the generated scorer is emitted at runtime (the
# forward path crosses the same world-age boundary). Without Enzyme loaded the
# inner call raises a MethodError — the intended "load Enzyme" signal.
function reverse_mode_gradient(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    seed = collect(params)
    objective = _generated_gradient_objective_or_nothing(model, seed, args, constraints)
    if isnothing(objective)
        throw(
            ArgumentError(
                "reverse_mode_gradient(model, ...) currently supports only models on the " *
                "type-stable generated-scorer path; this model falls back to the interpreter. " *
                "Use logjoint_gradient_unconstrained (forward-mode) instead.",
            ),
        )
    end
    return Base.invokelatest(reverse_mode_gradient, objective, seed)
end
