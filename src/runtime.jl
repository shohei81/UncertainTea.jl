mutable struct RuntimeContext
    rng::AbstractRNG
    constraints::ChoiceMap
    overrides::ChoiceMap
    choices::ChoiceMap
    log_weight::Float64
    prefix::Address
    require_all_choices_scored::Bool
end

struct TeaTrace
    model::TeaModel
    args::Tuple
    choices::ChoiceMap
    retval::Any
    log_weight::Float64
end

RuntimeContext(
    rng::AbstractRNG,
    constraints::ChoiceMap;
    overrides::ChoiceMap=ChoiceMap(),
    require_all_choices_scored::Bool=false,
) = RuntimeContext(rng, constraints, overrides, ChoiceMap(), 0.0, (), require_all_choices_scored)

function _join_address(prefix::Address, address)
    return (prefix..., normalize_address(address)...)
end

function _execute(model::TeaModel, ctx::RuntimeContext, args...)
    return model.impl(ctx, args...)
end

function choice(ctx::RuntimeContext, address, dist::AbstractTeaDistribution)
    full_address = _join_address(ctx.prefix, address)
    # In-model duplicate recording must error: a ChoiceMap-style update here
    # would silently drop the earlier random choice (and leave the recorded
    # log-weight inconsistent with the trace). User-side ChoiceMap updates
    # outside a model execution are unaffected.
    isnothing(_choice_index_normalized(ctx.choices, full_address)) || throw(
        ArgumentError(
            "duplicate choice address $(full_address) in model execution: every random choice " *
            "needs a unique address, but this address was already recorded earlier in the same " *
            "execution (the second choice would silently overwrite the first)",
        ),
    )
    if haskey(ctx.overrides, full_address)
        value = ctx.overrides[full_address]
        ctx.log_weight += Float64(logpdf(dist, value))
    elseif haskey(ctx.constraints, full_address)
        value = ctx.constraints[full_address]
        ctx.log_weight += Float64(logpdf(dist, value))
    elseif ctx.require_all_choices_scored
        throw(ArgumentError("logjoint requires a provided value for choice $(full_address)"))
    else
        value = rand(ctx.rng, dist)
    end
    _pushchoice!(ctx.choices, full_address, value)
    return value
end

function choice(ctx::RuntimeContext, address, call::TeaCall)
    previous_prefix = ctx.prefix
    ctx.prefix = _join_address(ctx.prefix, address)
    value = _execute(call.model, ctx, call.args...)
    ctx.prefix = previous_prefix
    return value
end

function choice(::RuntimeContext, address, rhs)
    throw(ArgumentError("unsupported right-hand side for choice at $(normalize_address(address)): $(typeof(rhs))"))
end

function _run_model(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    rng::AbstractRNG=Random.default_rng(),
    overrides::ChoiceMap=choicemap(),
    require_all_choices_scored::Bool=false,
)
    ctx = RuntimeContext(
        rng,
        constraints;
        overrides=overrides,
        require_all_choices_scored=require_all_choices_scored,
    )
    retval = _execute(model, ctx, args...)
    trace = TeaTrace(model, args, ctx.choices, retval, ctx.log_weight)
    return trace, ctx.log_weight
end

"""
    generate(model::TeaModel, args=(), constraints=choicemap(); rng) -> (trace, log_weight)

Run `model` with the given argument tuple, sampling every random choice whose
address is absent from `constraints` and fixing (and scoring) every choice
whose address is present. Returns the resulting `TeaTrace` and the log-weight:
the sum of the log-densities of the constrained choices.
"""
function generate(model::TeaModel, args::Tuple=(), constraints::ChoiceMap=choicemap(); rng::AbstractRNG=Random.default_rng())
    return _run_model(model, args, constraints; rng=rng)
end

"""
    assess(model::TeaModel, args=(), constraints=choicemap(); rng) -> Float64

Score a complete execution: every random choice the model makes must have a
value in `constraints` (a missing address throws). Returns the log joint
density of the supplied choices.
"""
function assess(model::TeaModel, args::Tuple=(), constraints::ChoiceMap=choicemap(); rng::AbstractRNG=Random.default_rng())
    _, log_weight = _run_model(model, args, constraints; rng=rng, require_all_choices_scored=true)
    return log_weight
end

Base.getindex(trace::TeaTrace, address) = trace.choices[address]
Base.length(trace::TeaTrace) = length(trace.choices)

function Base.show(io::IO, trace::TeaTrace)
    print(io, "TeaTrace(", trace.model.name, ", ", length(trace), " choices)")
end
