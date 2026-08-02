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

function _resolve_signature_plan(model::TeaModel, constraints::ChoiceMap)
    _validate_constraint_values_not_nan(constraints)
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
