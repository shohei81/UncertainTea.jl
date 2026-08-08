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
# exp(location + scale * z) for the log-space (lognormal) variant. Concretely
# typed on the compiled loc/scale expressions (issue #326): abstract `Any`
# fields here made every noncentered CompiledChoicePlanStep layout opaque to
# Enzyme's type analysis, which rejected the whole dependent-transform walk
# ("bad enzyme_type"), silently locking noncentered models out of the reverse
# tier.
struct CompiledNoncentered{L<:AbstractCompiledExpr,S<:AbstractCompiledExpr}
    location::L
    scale::S
    logspace::Bool
end

# marginalize=:enumerate walk data (docs/discrete-enumeration.md): the
# compile-time support values to enumerate. bernoulli: (false, true);
# categorical: (1, ..., K) from the literal probability-vector length.
struct CompiledMarginalize{S<:Tuple}
    support::S
end

# Every optional field is a type parameter constrained to its small Union
# (issue #326): with plain `Union{Nothing,...}` FIELDS the LLVM layout carries
# union selector bytes that Enzyme's type analysis cannot classify, and the
# reverse-mode probe fails on any plan whose steps flow through the
# dependent-transform walk (reparam=:noncentered). Parametrizing makes each
# instantiation's layout fully concrete; plan compilation already specializes
# per step, so this adds no new dynamic dispatch.
struct CompiledChoicePlanStep{
    A<:Tuple,
    AD<:CompiledAddressSpec,
    C,
    B<:Union{Nothing,Int},
    PV<:Union{Nothing,UnitRange{Int}},
    PS<:Union{Nothing,Int},
    N<:Union{Nothing,CompiledNoncentered},
    M<:Union{Nothing,CompiledMarginalize},
} <: AbstractCompiledPlanStep
    binding_slot::B
    address::AD
    constructor::C
    arguments::A
    parameter_value_indices::PV
    parameter_slot::PS
    # compiled location/scale (plus log-space flag) for reparam=:noncentered
    # latents; `nothing` for centered choices
    noncentered::N
    # compile-time enumeration support for marginalize=:enumerate latents;
    # `nothing` for ordinary choices
    marginalize::M
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

    throw(
        ArgumentError(
            "could not resolve the name `$sym` while compiling model `$(model.name)`: " *
            "it is not a model argument, a bound choice, a local binding, or a name " *
            "defined in the model's module.",
        ),
    )
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

        throw(
            ArgumentError(
                "expression not supported in compiled @tea static model code: `$expr`. " *
                "Model bodies compile a fixed expression subset (arithmetic, dotted " *
                "operators and function calls, indexing, tuples); move other computation " *
                "outside the model and pass its result in as a model argument.",
            ),
        )
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
        throw(
            ArgumentError(
                "the compiled scorer only supports distribution choice steps; this model " *
                "contains a generative subcall, which evaluates on the interpreted path.",
            ),
        )
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

include("evaluator/signature.jl")

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

include("evaluator/staging.jl")

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
        found || throw(
            ArgumentError(
                "no constraint value provided for the observed choice `$(address)`. " *
                "Constraining any index of a repeated site classifies the whole site " *
                "as observed, so every index needs a value — supply all of them, or " *
                "none to keep the site latent.",
            ),
        )
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

"""
    logjoint(model::TeaModel, params::AbstractVector, args=(), constraints=choicemap()) -> Float64

Log joint density of `model` via the compiled execution plan: the latent
choices (those NOT in `constraints`) take their values from the flat vector
`params` — constrained-space values in the plan's parameter layout order (see
`parameter_vector` / `initialparameters`) — while `constraints` fixes the
observed choices. For the unconstrained-space counterpart used by the samplers
see `logjoint_unconstrained`.
"""
function logjoint(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap(),
)
    resolved = _resolve_signature_plan(model, constraints, args)
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
    resolved = _resolve_signature_plan(model, constraints, args)
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

include("evaluator/noncentered_walk.jl")

include("evaluator/gradient_cache.jl")
