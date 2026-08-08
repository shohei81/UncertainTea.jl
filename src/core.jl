abstract type TeaMode end

struct StaticMode <: TeaMode end
struct DynamicMode <: TeaMode end

const STATIC_MODE = StaticMode()
const DYNAMIC_MODE = DynamicMode()

struct TeaModel{M<:TeaMode,F,S}
    mode::M
    name::Symbol
    impl::F
    spec::S
    # Completes a partial trailing-argument tuple using the model's default
    # argument expressions (Julia's own default-argument semantics), so the
    # compiled scoring APIs agree with `generate`. `nothing` when the model
    # was constructed without a filler (defaults then require the full tuple).
    argument_filler::Any
    # True when the (dynamic-mode) model body contains branchful control flow
    # that the linear execution plan cannot represent; compiled scoring must
    # reject such models instead of silently scoring both branches.
    branchful::Bool
    evaluator_cache::Base.RefValue{Any}
    backend_cache::Base.RefValue{Any}
    # Compiled scoring plans keyed by `(signature, dims)`: the conditioning
    # signature (the set of constrained observation addresses) plus the
    # runtime-dims tuple resolved from the model arguments (issue #289; always
    # `()` for dims-free models). See docs/constraint-driven-conditioning.md
    # and `_resolve_signature_plan`. Lazily a
    # `Dict{Tuple{Set{Address},Tuple{Vararg{Int}}},ResolvedSignaturePlan}`, so a
    # model re-run with new data at the same observed addresses reuses its
    # cached plan/layout.
    signature_cache::Base.RefValue{Any}
end

function TeaModel(
    mode::M,
    name::Symbol,
    impl::F,
    spec::S;
    argument_filler=nothing,
    branchful::Bool=false,
) where {M<:TeaMode,F,S}
    return TeaModel{M,F,S}(
        mode,
        name,
        impl,
        spec,
        argument_filler,
        branchful,
        Ref{Any}(nothing),
        Ref{Any}(nothing),
        Ref{Any}(nothing),
    )
end

# Complete a partial model-argument tuple for the compiled scoring APIs.
# Missing TRAILING arguments are filled from the model's default argument
# expressions (evaluated with Julia's own default-argument semantics, so
# defaults may reference earlier arguments); a tuple that cannot be completed
# throws a DimensionMismatch instead of silently diverging from `generate`.
function _complete_model_args(model::TeaModel, args::Tuple)
    expected = length(modelspec(model).arguments)
    length(args) == expected && return args
    filler = model.argument_filler
    if length(args) > expected || isnothing(filler)
        throw(DimensionMismatch("expected $expected model arguments, got $(length(args))"))
    end
    try
        return filler(args...)::Tuple
    catch err
        if err isa MethodError && err.f === filler
            throw(
                DimensionMismatch(
                    "expected $expected model arguments, got $(length(args)); the model's default " *
                    "argument values do not cover the missing trailing arguments",
                ),
            )
        end
        rethrow()
    end
end

struct TeaCall{M<:TeaModel,A<:Tuple}
    model::M
    args::A
end

(model::TeaModel)(args...) = TeaCall(model, args)

function Base.show(io::IO, model::TeaModel)
    print(io, "TeaModel(", model.name, ", mode=", nameof(typeof(model.mode)))
    model.spec === nothing || print(io, ", choices=", length(model.spec.choices))
    print(io, ")")
end
