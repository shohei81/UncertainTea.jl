# --- conditioning-signature resolution (issue #95) ---------------------------
#
# The compiled scoring path classifies latents/observations from the CONDITIONING
# SIGNATURE (docs/constraint-driven-conditioning.md): the set of constrained
# addresses that name a static, unscoped choice. The signature-specific execution
# plan (and its parameter layout) is memoized per `(model, (signature, dims))` in
# the model's `signature_cache` -- `dims` is the runtime-dims tuple resolved from
# the model arguments (issue #289; always `()` for dims-free models) -- so
# re-running inference with new data at the same observed addresses reuses the
# compiled plan.

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

# A representative single argument tuple for signature resolution. The batched
# and device paths accept either a shared tuple or a per-column vector of
# tuples; the resolved plan is per-signature, so the first entry stands in for
# the batch. Per-column args of a runtime-dims model must additionally agree on
# ONE dims tuple before the representative column may stand in for the batch;
# the batched entry points resolve through
# `_resolve_signature_plan_representative`, which validates exactly that.
_representative_args(args::Tuple) = args
function _representative_args(args::AbstractVector)
    isempty(args) && return ()
    representative = first(args)
    # Malformed per-column args (entries that are not tuples) cannot stand in
    # for the batch; return `()` so `_validate_batched_args` raises its
    # informative ArgumentError instead of resolution hitting a MethodError.
    return representative isa Tuple ? representative : ()
end

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

# The signature-cache key (issue #289): the conditioning signature plus the
# runtime-dims tuple resolved from the model arguments. Dims-free models (the
# overwhelming majority) always resolve `()` for the dims component, so one
# signature keeps exactly one cache entry; runtime-dim models re-specialize
# everything memoized on the ResolvedSignaturePlan per dims for free.
const _SignatureCacheKey = Tuple{Set{Address},Tuple{Vararg{Int}}}

# Families whose runtime-length LATENT support has landed (issue #289, PR-2):
# the diagonal and dense multivariate normals, both unconstraining through a
# plain `VectorIdentityTransform(n)` constructed at signature-resolution time.
# The remaining candidate families (mvstudentt/mvstudenttdense/dirichlet) keep
# the early pending error until PR-3.
const _RUNTIME_DIM_SUPPORTED_FAMILIES = (:mvnormal, :mvnormaldense)

# Runtime dims of `model` under `signature` given `args` (issue #289). The fast
# path is one boolean test: a model without runtime-dim candidates resolves to
# the empty tuple without touching `args`. A candidate that is OBSERVED under
# the signature contributes nothing (a constrained mv choice scores with a
# dynamic size and needs no slot); a LATENT candidate of a supported family
# resolves its length from the model arguments by the dims walk
# (src/evaluator/runtime_dims.jl), one Int per latent candidate in plan order.
function _resolve_runtime_dims(
    @nospecialize(model::TeaModel),
    signature::Set{Address},
    @nospecialize(args::Tuple),
)
    candidates = executionplan(model).runtime_dim_candidates
    isempty(candidates) && return ()
    latents = RuntimeDimCandidate[]
    for candidate in candidates
        candidate.address in signature && continue
        candidate.family in _RUNTIME_DIM_SUPPORTED_FAMILIES || throw(
            ArgumentError(
                "latent `$(candidate.family)` at address `$(candidate.address)` with a " *
                "runtime-length argument is not supported yet (issue #289): runtime dimensions " *
                "currently cover `mvnormal`/`mvnormaldense` latents; the length must be a " *
                "literal vector/tuple, or the address must be constrained as an observation",
            ),
        )
        push!(latents, candidate)
    end
    isempty(latents) && return ()
    return _runtime_dims_walk(model, latents, args)
end

# Batched-path resolution: the batched/device entry points accept a shared
# constraints/args pair or per-column vectors. The resolved plan is
# per-signature, so the representative pair stands in for the batch -- but for
# a runtime-dims model the representative column must not silently stand in
# for columns of a DIFFERENT latent length, so per-column args are first
# validated to agree on one dims tuple (issue #289).
function _resolve_signature_plan_representative(
    @nospecialize(model::TeaModel),
    @nospecialize(constraints),
    @nospecialize(args),
)
    _validate_shared_runtime_dims(model, constraints, args)
    return _resolve_signature_plan(
        model, _representative_constraints(constraints), _representative_args(args),
    )
end

function _validate_shared_runtime_dims(
    @nospecialize(model::TeaModel),
    @nospecialize(constraints),
    @nospecialize(args),
)
    args isa AbstractVector || return nothing
    isempty(executionplan(model).runtime_dim_candidates) && return nothing
    representative = _representative_constraints(constraints)
    representative isa ChoiceMap || return nothing
    signature = _conditioning_signature(model, representative)
    reference = nothing
    for column_args in args
        column_args isa Tuple || continue
        dims = _resolve_runtime_dims(model, signature, column_args)
        if isnothing(reference)
            reference = dims
        elseif dims != reference
            throw(
                DimensionMismatch(
                    "batched per-column model arguments resolve conflicting runtime latent " *
                    "dimensions $(reference) vs $(dims); every column of a batch must resolve " *
                    "the same dims tuple (issue #289) -- run mixed-length batches as separate calls",
                ),
            )
        end
    end
    return nothing
end

function _resolve_signature_plan(
    @nospecialize(model::TeaModel),
    signature::Set{Address},
    @nospecialize(args::Tuple),
)
    dims = _resolve_runtime_dims(model, signature, args)
    return _resolve_signature_plan_keyed(model, (signature, dims))
end

function _resolve_signature_plan_keyed(@nospecialize(model::TeaModel), key::_SignatureCacheKey)
    _reject_branchful_compiled_scoring(model)
    return lock(_PLAN_MEMO_LOCK) do
        cache = model.signature_cache[]
        if isnothing(cache)
            cache = Dict{_SignatureCacheKey,ResolvedSignaturePlan}()
            model.signature_cache[] = cache
        end
        store = cache::Dict{_SignatureCacheKey,ResolvedSignaturePlan}
        existing = get(store, key, nothing)
        isnothing(existing) && begin
            plan = _signature_execution_plan(executionplan(model), key[1], key[2])
            existing = ResolvedSignaturePlan(plan, _compile_execution_plan(model, plan))
            store[key] = existing
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

# --- NaN observation guard (issue #346) ---------------------------------------
#
# A NaN inside a constrained observation value makes `logjoint` (and every
# gradient) silently NaN, flowing unnoticed into waic/psis_loo-style scoring,
# while sampler init failures blame the parameters instead of the data. NaN in
# observed data is ALWAYS a data bug (unlike Inf, which can legitimately score
# -Inf where the support allows it), so throw at constraint-resolution time and
# name the offending address. The scan is value-only (no model execution) and
# is stamped on the ChoiceMap per `mutation_count`
# (`nan_checked_mutation_count`), so the hot path pays one Int comparison per
# evaluation — mirroring how the issue-#310 misconditioning warning stays off
# the hot path.

_constraint_value_has_nan(@nospecialize(value)) = false
_constraint_value_has_nan(value::Real) = isnan(value)
_constraint_value_has_nan(value::AbstractArray) = any(_constraint_value_has_nan, value)
_constraint_value_has_nan(value::Tuple) = any(_constraint_value_has_nan, value)

function _validate_constraint_values_not_nan(constraints::ChoiceMap)
    constraints.nan_checked_mutation_count == constraints.mutation_count && return nothing
    for entry in constraints.entries
        if _constraint_value_has_nan(last(entry))
            detail = last(entry) isa Union{AbstractArray,Tuple} ? "contains NaN" : "is NaN"
            throw(
                ArgumentError(
                    "constraint value for observed choice `$(first(entry))` $detail — " *
                    "check the data passed to choicemap",
                ),
            )
        end
    end
    constraints.nan_checked_mutation_count = constraints.mutation_count
    return nothing
end

function _resolve_signature_plan(model::TeaModel, constraints::ChoiceMap, args::Tuple)
    _validate_constraint_values_not_nan(constraints)
    signature = _conditioning_signature(model, constraints)
    # the dims resolution stays outside the locked region (the walk is pure and
    # cheap; dims-free models pay one boolean test)
    dims = _resolve_runtime_dims(model, signature, args)
    key = (signature, dims)
    # first-encounter validation only: peek the memo unlocked (a benign race
    # merely repeats the cheap static check)
    cache = model.signature_cache[]
    if isnothing(cache) || !haskey(cache::Dict{_SignatureCacheKey,ResolvedSignaturePlan}, key)
        _warn_unmatched_constraint_addresses(model, constraints)
    end
    return _resolve_signature_plan_keyed(model, key)
end

# The signature-specific parameter layout for one conditioning. The CPU
# inference entry points size, initialize, and reconstruct against THIS layout
# (issue #95, PR-6), not the syntactic default `parameterlayout(model)`: the
# latent set is a function of which addresses are constrained, so a constrained
# bound choice drops its slot and an unconstrained unbound choice gains one.
_conditioned_parameter_layout(model::TeaModel, constraints::ChoiceMap, args) =
    _resolve_signature_plan(model, constraints, _representative_args(args)).plan.parameter_layout

# Per-column conditioning (a chain/result may carry one ChoiceMap per column):
# every column shares the same signature, so the representative constraints fix
# the (static) layout, matching how `_batched_signature_layout` resolves it.
_conditioned_parameter_layout(model::TeaModel, constraints::AbstractVector, args) =
    _conditioned_parameter_layout(model, _representative_constraints(constraints), args)

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
