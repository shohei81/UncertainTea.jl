function _batched_backend_gradient_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints;
    reject_invalid_parameters::Bool=false,
)
    element_type = float(eltype(params))
    workspace = BatchedLogjointWorkspace(
        model, batch_constraints; reject_invalid_parameters=reject_invalid_parameters,
    )
    backend_plan = workspace.backend_plan
    isnothing(backend_plan) && return nothing
    _backend_gradient_supported(backend_plan) || return nothing

    cache = BatchedBackendGradientCache(
        workspace,
        zeros(element_type, size(params, 1), length(workspace.environment.layout.symbols), size(params, 2)),
        Matrix{element_type}[],
        batch_args,
        batch_constraints,
        _backend_gradient_seed_rows(workspace.layout),
        IdDict{Any,Any}(),
        IdDict{Any,Any}(),
        IdDict{Any,Any}(),
    )
    totals = _batched_totals_buffer!(workspace, size(params, 2), element_type)
    try
        _batched_backend_logjoint_and_gradient_unconstrained!(totals, gradient_buffer, model, cache, params)
    catch err
        err isa BatchedBackendFallback || rethrow()
        return nothing
    end
    return cache
end

function _batched_gradient_column!(
    model::TeaModel,
    workspace::BatchedLogjointWorkspace,
    destination::AbstractVector,
    seed::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    objective = theta -> _logjoint_unconstrained_with_workspace!(model, workspace, theta, args, constraints)
    config = ForwardDiff.GradientConfig(objective, seed)
    ForwardDiff.gradient!(destination, objective, seed, config)
    return destination
end

function _batched_gradient_column_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
    batch_index::Int;
    reject_invalid_parameters::Bool=false,
)
    column_constraints = _batched_constraints(batch_constraints, batch_index)
    workspace = BatchedLogjointWorkspace(
        model, column_constraints; reject_invalid_parameters=reject_invalid_parameters,
    )
    objective = BatchedGradientObjective(
        model,
        workspace,
        _batched_args(batch_args, batch_index),
        column_constraints,
    )
    seed = collect(view(params, :, batch_index))
    config = ForwardDiff.GradientConfig(objective, seed)
    buffer = view(gradient_buffer, :, batch_index)
    return BatchedGradientColumnCache(objective, config, buffer)
end

function _batched_flat_gradient_cache(
    model::TeaModel,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints;
    reject_invalid_parameters::Bool=false,
)
    workspace = BatchedLogjointWorkspace(
        model, batch_constraints; reject_invalid_parameters=reject_invalid_parameters,
    )
    objective = BatchedFlatGradientObjective(
        model,
        workspace,
        batch_args,
        batch_constraints,
        size(params, 1),
        size(params, 2),
    )
    seed = collect(vec(params))
    probe = _batched_totals_buffer!(workspace, size(params, 2), eltype(params))
    try
        _batched_logjoint_unconstrained_with_workspace!(probe, model, workspace, params, batch_args, batch_constraints)
    catch err
        err isa BatchedBackendFallback || rethrow()
        return nothing
    end
    config = ForwardDiff.GradientConfig(objective, seed)
    return BatchedFlatGradientCache(objective, config, vec(gradient_buffer))
end

# --- analytic-gradient chain-block threading (issue #143) -------------------
#
# The HOST batched analytic gradient (`_batched_backend_logjoint_and_gradient_unconstrained!`)
# is per-column independent: the sufficient-statistics fusion and the observed-
# loop reduction run per chain, so partitioning the batch columns into
# contiguous blocks and writing disjoint slices of the totals/gradient buffers
# is BITWISE identical to the serial pass. That makes threading over blocks a
# free multi-core win for logistic-shaped models (per-observation covariate work
# that cannot be sufficient-statistics-fused). The workspace scratch is NOT
# thread-safe, so each block owns its own `BatchedBackendGradientCache`.
#
# Work-size gate: threading only helps when the per-call work stays >= ~ms;
# task-spawn overhead dominates sub-ms gradients (a sufficient-statistics-fused
# gauss gradient is ~0.27 ms at 512 chains and MUST stay serial). We gate on a
# deterministic, machine-independent estimate of the NON-FUSED per-observation
# gradient work:
#
#     work = batch_size * parameter_count * (sum of observation counts over
#            observed-loop steps that did NOT take the sufficient-statistics
#            fused tier)
#
# A sufficient-statistics-fused loop collapses to a few cached numbers, so it
# contributes 0 -- gauss (fully fused) estimates 0 and always stays serial. The
# fused/non-fused split is read once from the construction-time probe's staging
# on the backend cache. The threshold below (50_000 work units) was tuned on the
# crossppl logistic benchmark (n=500 observations, 9 parameters): it thresholds
# logistic IN at batch_size >= 64 (64*9*500 = 288_000, a ~4 ms serial gradient)
# while keeping genuinely small non-fused loops (and all fused models) serial.
const _BATCHED_GRADIENT_THREAD_WORK_THRESHOLD = Ref(50_000)

# Enumerate contiguous, disjoint column ranges partitioning 1:n into k blocks as
# evenly as possible (leading blocks take the remainder).
function _batched_gradient_block_ranges(n::Int, k::Int)
    base, extra = divrem(n, k)
    ranges = Vector{UnitRange{Int}}(undef, k)
    start = 1
    for b = 1:k
        width = base + (b <= extra ? 1 : 0)
        ranges[b] = start:(start+width-1)
        start += width
    end
    return ranges
end

_batched_args_block(args::Tuple, ::UnitRange{Int}) = args
_batched_args_block(args::AbstractVector, range::UnitRange{Int}) = args[range]

_batched_constraints_block(constraints::ChoiceMap, ::UnitRange{Int}) = constraints
_batched_constraints_block(constraints::AbstractVector, range::UnitRange{Int}) = constraints[range]

# Non-fused per-observation gradient work estimate (see the gate note above).
# Reads the construction-time probe staging: `observed_loop_values` holds the
# gathered observation vector per loop step; `observed_loop_stats` holds a
# non-`nothing` sufficient-statistics summary only for the fused loops. A loop
# with a fused stats entry contributes 0; every other staged loop contributes
# its observation count.
function _batched_gradient_nonfused_obs(backend_cache::BatchedBackendGradientCache)
    nonfused = 0
    for (step, entry) in backend_cache.observed_loop_values
        stats_entry = get(backend_cache.observed_loop_stats, step, nothing)
        if stats_entry !== nothing && stats_entry[2] !== nothing
            continue  # sufficient-statistics-fused loop: ~O(1) per chain
        end
        nonfused += length(entry[2])
    end
    return nonfused
end

# Build the chain-block threading plan, or return `nothing` when the work-size
# gate keeps the problem serial (small/fused models, single-threaded runtimes,
# or too few columns to block). This is construction-time work, done once.
function _build_backend_gradient_thread_plan(
    model::TeaModel,
    backend_cache::BatchedBackendGradientCache,
    gradient_buffer::AbstractMatrix,
    params::AbstractMatrix,
    batch_args,
    batch_constraints,
    batch_size::Int,
    parameter_count::Int;
    reject_invalid_parameters::Bool=false,
)
    Threads.nthreads() > 1 || return nothing
    nonfused_obs = _batched_gradient_nonfused_obs(backend_cache)
    work = batch_size * parameter_count * nonfused_obs
    work >= _BATCHED_GRADIENT_THREAD_WORK_THRESHOLD[] || return nothing

    # each block must own >= 2 columns so per-block scratch pays for the spawn
    nblocks = min(Threads.nthreads(), fld(batch_size, 2))
    nblocks >= 2 || return nothing

    ranges = _batched_gradient_block_ranges(batch_size, nblocks)
    caches = Vector{typeof(backend_cache)}(undef, length(ranges))
    for (index, range) in enumerate(ranges)
        block_cache = _batched_backend_gradient_cache(
            model,
            view(gradient_buffer, :, range),
            view(params, :, range),
            _batched_args_block(batch_args, range),
            _batched_constraints_block(batch_constraints, range);
            reject_invalid_parameters=reject_invalid_parameters,
        )
        # a per-block capability gap the full-batch probe did not hit degrades
        # the whole cache to serial (correctness over parallelism)
        isnothing(block_cache) && return nothing
        caches[index] = block_cache
    end
    return BatchedGradientThreadPlan(ranges, caches, reject_invalid_parameters)
end

# Should a caught block error degrade the whole call to the serial path (a
# `BatchedBackendFallback`, or -- in reject mode -- a per-lane parameter
# validation throw), or is it a genuine bug to rethrow? Mirrors the serial
# `_batched_backend_gradient_or_columns!` degradation predicate.
function _batched_gradient_block_error_recoverable(err, reject_invalid_parameters::Bool)
    return err isa BatchedBackendFallback ||
           (reject_invalid_parameters && (err isa ArgumentError || err isa DomainError))
end

# Run each block's analytic gradient under `Threads.@threads`, writing disjoint
# slices of `totals`/`gradient_buffer`. Returns `true` on success; returns
# `false` if any block hit a recoverable fallback/reject error (the caller then
# redoes the whole call serially -- partial block writes are overwritten). A
# non-recoverable block error is rethrown.
function _threaded_backend_gradient!(
    totals::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
    plan::BatchedGradientThreadPlan,
)
    ranges = plan.ranges
    caches = plan.caches
    nblocks = length(ranges)
    errors = Vector{Any}(nothing, nblocks)
    Threads.@threads for block = 1:nblocks
        range = ranges[block]
        try
            _batched_backend_logjoint_and_gradient_unconstrained!(
                view(totals, range),
                view(cache.gradient_buffer, :, range),
                cache.model,
                caches[block],
                view(params, :, range),
            )
        catch err
            errors[block] = err
        end
    end
    recoverable = false
    for err in errors
        isnothing(err) && continue
        if _batched_gradient_block_error_recoverable(err, plan.reject_invalid_parameters)
            recoverable = true
        else
            throw(err)
        end
    end
    return !recoverable
end

# Threaded logjoint-only pass (no gradient): each block scores its column slice
# through its own workspace. The per-workspace scorer self-recovers to its
# per-column fallback, so blocks do not raise recoverable errors here; any
# escape degrades the whole call to serial.
function _threaded_backend_logjoint!(
    destination::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
    plan::BatchedGradientThreadPlan,
)
    ranges = plan.ranges
    caches = plan.caches
    nblocks = length(ranges)
    errors = Vector{Any}(nothing, nblocks)
    Threads.@threads for block = 1:nblocks
        range = ranges[block]
        block_cache = caches[block]
        try
            _batched_logjoint_unconstrained_with_workspace!(
                view(destination, range),
                cache.model,
                block_cache.workspace,
                view(params, :, range),
                block_cache.args,
                block_cache.constraints,
            )
        catch err
            errors[block] = err
        end
    end
    for err in errors
        isnothing(err) || return false
    end
    return true
end

# `reject_invalid_parameters=true` selects Stan-style reject semantics for the
# compiled-plan evaluation tiers of this cache (issue #157); samplers construct
# their caches with it on, the public gradient APIs keep the throwing default.
function BatchedLogjointGradientCache(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap();
    reject_invalid_parameters::Bool=false,
    adtype::Symbol=:auto,
)
    adtype in (:auto, :forward, :reverse) ||
        throw(ArgumentError("adtype must be :auto, :forward, or :reverse, got $(adtype)"))
    batch_size = _validate_batched_unconstrained_params(model, params, constraints)
    batch_args = _validate_batched_args(model, args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    parameter_count = size(params, 1)
    gradient_buffer = Matrix{float(eltype(params))}(undef, parameter_count, batch_size)
    # Lane-compaction scratch (issue #160), pre-sized to the full batch width so the
    # masked-NUTS gather/scatter never allocates per leapfrog step.
    compact_params = Matrix{float(eltype(params))}(undef, parameter_count, batch_size)
    compact_gradient = Matrix{float(eltype(params))}(undef, parameter_count, batch_size)
    compact_logjoint = Vector{Float64}(undef, batch_size)
    compact_index = Vector{Int}(undef, batch_size)
    if batch_size == 0
        return BatchedLogjointGradientCache(
            model, Any[], nothing, nothing, gradient_buffer, parameter_count, batch_size, nothing,
            compact_params, compact_gradient, compact_logjoint, compact_index, nothing,
        )
    end

    # Analytic backend tier is always best when available -- reverse mode never
    # preempts it (even under adtype=:reverse, which only prefers reverse over the
    # forward-mode tiers).
    backend_cache = _batched_backend_gradient_cache(
        model, gradient_buffer, params, batch_args, batch_constraints;
        reject_invalid_parameters=reject_invalid_parameters,
    )
    if !isnothing(backend_cache)
        thread_plan = _build_backend_gradient_thread_plan(
            model, backend_cache, gradient_buffer, params, batch_args, batch_constraints,
            batch_size, parameter_count; reject_invalid_parameters=reject_invalid_parameters,
        )
        return BatchedLogjointGradientCache(
            model, Any[], backend_cache, nothing, gradient_buffer, parameter_count, batch_size, thread_plan,
            compact_params, compact_gradient, compact_logjoint, compact_index, nothing,
        )
    end

    # Reverse-mode tier (issue #268, A2): preempts the forward-mode tiers below
    # when eligible + guarded. `nothing` when it does not apply, so the existing
    # flat/column fallbacks stay in charge.
    reverse_cache = _maybe_batched_reverse_gradient_cache(
        model, params, args, constraints, parameter_count, adtype,
    )
    if !isnothing(reverse_cache)
        return BatchedLogjointGradientCache(
            model, Any[], nothing, nothing, gradient_buffer, parameter_count, batch_size, nothing,
            compact_params, compact_gradient, compact_logjoint, compact_index, reverse_cache,
        )
    end

    flat_cache =
        isnothing(_backend_execution_plan(model)) ? nothing :
        _batched_flat_gradient_cache(
            model, gradient_buffer, params, batch_args, batch_constraints;
            reject_invalid_parameters=reject_invalid_parameters,
        )
    if !isnothing(flat_cache)
        return BatchedLogjointGradientCache(
            model,
            Any[],
            nothing,
            flat_cache,
            gradient_buffer,
            parameter_count,
            batch_size,
            nothing,
            compact_params,
            compact_gradient,
            compact_logjoint,
            compact_index,
            nothing,
        )
    end

    first_cache = _batched_gradient_column_cache(
        model, gradient_buffer, params, batch_args, batch_constraints, 1;
        reject_invalid_parameters=reject_invalid_parameters,
    )
    column_caches = Vector{typeof(first_cache)}(undef, batch_size)
    column_caches[1] = first_cache
    for batch_index = 2:batch_size
        column_caches[batch_index] = _batched_gradient_column_cache(
            model,
            gradient_buffer,
            params,
            batch_args,
            batch_constraints,
            batch_index;
            reject_invalid_parameters=reject_invalid_parameters,
        )
    end

    return BatchedLogjointGradientCache(
        model,
        column_caches,
        nothing,
        nothing,
        gradient_buffer,
        parameter_count,
        batch_size,
        nothing,
        compact_params,
        compact_gradient,
        compact_logjoint,
        compact_index,
        nothing,
    )
end

# Build the batched per-column reverse-mode tier (issue #268, A2), or `nothing` to
# leave the forward/analytic tiers in charge. The guard is deliberately
# conservative -- reverse mode is used only when every precondition holds and a
# trial gradient actually compiles -- so it can never make the gradient path fail
# where forward mode would have worked (the user asked for automatic + stable):
#
#   * adtype selects it (:reverse forces it; :auto uses it above a size threshold;
#     :forward never does);
#   * the batch shares ONE `args`/`constraints` (multi-chain on the same posterior),
#     so a single generated objective serves every column;
#   * the model is on the type-stable generated-scorer path (else no
#     Enzyme-differentiable objective exists);
#   * Enzyme is loaded AND a trial value+gradient on column 1 compiles and is
#     finite -- this both activates the extension method and rejects any model
#     Enzyme cannot handle, BEFORE any sampling runs.
#
# Any failure returns `nothing`, and the caller falls back to the forward tiers.
#
# Threshold (issue #277). Measured `batched_nuts` (4 chains, 300+300) on non-analytic
# coupled models: once warm, reverse mode beats forward from ~P=8 (1.3x) and the
# speedup grows monotonically (P=16 ~4x, P=24 ~5x, P=32 ~9.5x). Including the
# one-time per-objective Enzyme compile (~2s), a SINGLE short run breaks even around
# P=24-28. 24 captures the strong medium-model wins (aligning with the ~20-parameter
# rule of thumb PPLs like Turing use), while the sub-threshold band stays on forward
# mode where the absolute run cost is small and the compile would not amortize; any
# repeated or longer run past the threshold amortizes the compile many times over.
const _REVERSE_MODE_AUTO_MIN_PARAMS = 24

function _maybe_batched_reverse_gradient_cache(model, params, args, constraints, parameter_count, adtype)
    adtype === :forward && return nothing
    # cheap gate: the extension method exists only while Enzyme is loaded, so
    # without it (the common case) we skip straight to forward mode without
    # building any objective or attempting a gradient.
    isempty(methods(reverse_mode_value_and_gradient)) && return nothing
    # only the shared-posterior case (a single args tuple + a single ChoiceMap)
    # collapses to one reusable objective; per-column vectors fall back to forward.
    (args isa Tuple && constraints isa ChoiceMap) || return nothing
    if adtype === :auto && parameter_count < _REVERSE_MODE_AUTO_MIN_PARAMS
        return nothing
    end
    seed = collect(view(params, :, 1))
    objective = _generated_gradient_objective_or_nothing(model, seed, args, constraints)
    isnothing(objective) && return nothing
    # compile + finiteness guard: run one real value+gradient. A MethodError here
    # means Enzyme is not loaded; any other error means Enzyme cannot compile this
    # objective. Either way, fall back to forward mode rather than fail later.
    try
        value, gradient = Base.invokelatest(reverse_mode_value_and_gradient, objective, seed)
        (isfinite(value) && all(isfinite, gradient)) || return nothing
    catch
        return nothing
    end
    return BatchedReverseGradientCache(objective, seed)
end

# A cached analytic backend gradient can hit a runtime capability gap the
# construction-time probe could not see (a marginalize branch whose suffix
# becomes unevaluable for an ignored column at the CURRENT parameters):
# recompute that call per column, which evaluates only what each column needs
# (the scalar workspace path itself retries on the compiled plan). Later calls
# keep the analytic tier.
function _batched_backend_gradient_or_columns!(
    totals::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    backend_cache = cache.backend_cache
    # threaded chain-block path (issue #143): run per-block analytic gradients in
    # parallel over disjoint column slices. A recoverable block error degrades to
    # the serial body below (it recomputes the whole call, overwriting any
    # partial block writes).
    plan = cache.thread_plan
    if !isnothing(plan) && _threaded_backend_gradient!(totals, cache, params, plan)
        return totals, cache.gradient_buffer
    end
    try
        _batched_backend_logjoint_and_gradient_unconstrained!(
            totals,
            cache.gradient_buffer,
            cache.model,
            backend_cache,
            params,
        )
        return totals, cache.gradient_buffer
    catch err
        # reject mode (issue #157): an analytic-tier parameter-validation throw
        # for one lane degrades this call to the per-column path, where only the
        # offending column scores -Inf
        if !(
            err isa BatchedBackendFallback ||
            (
                backend_cache.workspace.environment.reject_invalid_parameters &&
                (err isa ArgumentError || err isa DomainError)
            )
        )
            rethrow()
        end
    end
    for batch_index = 1:cache.batch_size
        column_args = _batched_args(backend_cache.args, batch_index)
        column_constraints = _batched_constraints(backend_cache.constraints, batch_index)
        seed = collect(view(params, :, batch_index))
        _batched_gradient_column!(
            cache.model,
            backend_cache.workspace,
            view(cache.gradient_buffer, :, batch_index),
            seed,
            column_args,
            column_constraints,
        )
        totals[batch_index] = _logjoint_unconstrained_with_workspace!(
            cache.model,
            backend_cache.workspace,
            seed,
            column_args,
            column_constraints,
        )
    end
    return totals, cache.gradient_buffer
end

function batched_logjoint_gradient_unconstrained!(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == cache.parameter_count ||
        throw(DimensionMismatch("expected $(cache.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batch elements, got $(size(params, 2))"))

    if !isnothing(cache.reverse_cache)
        _batched_reverse_fill_value_and_gradient!(nothing, cache, params)
        return cache.gradient_buffer
    end

    if !isnothing(cache.backend_cache)
        totals = _batched_totals_buffer!(cache.backend_cache.workspace, cache.batch_size, eltype(cache.gradient_buffer))
        _batched_backend_gradient_or_columns!(totals, cache, params)
        return cache.gradient_buffer
    end

    if !isnothing(cache.flat_cache)
        ForwardDiff.gradient!(cache.flat_cache.flat_buffer, cache.flat_cache.objective, vec(params), cache.flat_cache.config)
        return cache.gradient_buffer
    end

    # per-column ForwardDiff fallback (issue #143): each column cache owns an
    # independent objective/workspace/config and writes a disjoint gradient-buffer
    # slice, so the loop threads with no shared mutable state and no RNG -- the
    # result is bitwise identical to the serial loop. This tier is inherently
    # heavy per column, so thread whenever there is more than one column per
    # thread; no work-size gate is needed.
    if Threads.nthreads() > 1 && cache.batch_size >= 2 * Threads.nthreads()
        Threads.@threads for batch_index = 1:cache.batch_size
            column_cache = cache.column_caches[batch_index]
            ForwardDiff.gradient!(column_cache.buffer, column_cache.objective, view(params, :, batch_index), column_cache.config)
        end
        return cache.gradient_buffer
    end
    for batch_index = 1:cache.batch_size
        column_cache = cache.column_caches[batch_index]
        ForwardDiff.gradient!(column_cache.buffer, column_cache.objective, view(params, :, batch_index), column_cache.config)
    end
    return cache.gradient_buffer
end

function _batched_logjoint_unconstrained_from_gradient_cache!(
    destination::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    length(destination) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batched values, got $(length(destination))"))

    if !isnothing(cache.reverse_cache)
        return _batched_reverse_fill_value!(destination, cache, params)
    end

    if !isnothing(cache.backend_cache)
        plan = cache.thread_plan
        if !isnothing(plan) && _threaded_backend_logjoint!(destination, cache, params, plan)
            return destination
        end
        return _batched_logjoint_unconstrained_with_workspace!(
            destination,
            cache.model,
            cache.backend_cache.workspace,
            params,
            cache.backend_cache.args,
            cache.backend_cache.constraints,
        )
    end

    if !isnothing(cache.flat_cache)
        objective = cache.flat_cache.objective
        return _batched_logjoint_unconstrained_with_workspace!(
            destination,
            cache.model,
            objective.workspace,
            params,
            objective.args,
            objective.constraints,
        )
    end

    for batch_index = 1:cache.batch_size
        column_cache = cache.column_caches[batch_index]
        objective = column_cache.objective
        destination[batch_index] = _logjoint_unconstrained_with_workspace!(
            cache.model,
            objective.workspace,
            view(params, :, batch_index),
            objective.args,
            objective.constraints,
        )
    end
    return destination
end

function _batched_logjoint_and_gradient_unconstrained!(
    destination::AbstractVector,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    size(params, 1) == cache.parameter_count ||
        throw(DimensionMismatch("expected $(cache.parameter_count) parameters, got $(size(params, 1))"))
    size(params, 2) == cache.batch_size ||
        throw(DimensionMismatch("expected $(cache.batch_size) batch elements, got $(size(params, 2))"))

    if !isnothing(cache.reverse_cache)
        # one ReverseWithPrimal pass per column yields BOTH value and gradient
        return _batched_reverse_fill_value_and_gradient!(destination, cache, params)
    end

    if !isnothing(cache.backend_cache)
        _batched_backend_gradient_or_columns!(destination, cache, params)
        return destination, cache.gradient_buffer
    end

    batched_logjoint_gradient_unconstrained!(cache, params)
    _batched_logjoint_unconstrained_from_gradient_cache!(destination, cache, params)
    return destination, cache.gradient_buffer
end

# --- batched per-column reverse-mode fill (issue #268, A2) -------------------
#
# All three reduce to looping the shared generated objective over columns with an
# Enzyme reverse pass. Each column is independent (multi-chain, same posterior),
# so this mirrors the forward column tier -- just O(1) in the parameter count
# instead of O(P). A per-column try/catch maps an invalid-region Enzyme failure to
# a non-finite logjoint / NaN gradient, exactly what the samplers already gate on,
# so a chain wandering off never crashes the run.

# value + gradient (destination === nothing writes only the gradient buffer).
function _batched_reverse_fill_value_and_gradient!(destination, cache::BatchedLogjointGradientCache, params)
    rc = cache.reverse_cache
    objective = rc.objective
    for batch_index = 1:cache.batch_size
        copyto!(rc.theta, view(params, :, batch_index))
        try
            value, gradient = Base.invokelatest(reverse_mode_value_and_gradient, objective, rc.theta)
            copyto!(view(cache.gradient_buffer, :, batch_index), gradient)
            destination === nothing || (destination[batch_index] = value)
        catch
            fill!(view(cache.gradient_buffer, :, batch_index), NaN)
            destination === nothing || (destination[batch_index] = -Inf)
        end
    end
    return destination, cache.gradient_buffer
end

# value only (the sampler asks for the logjoint without the gradient).
function _batched_reverse_fill_value!(destination, cache::BatchedLogjointGradientCache, params)
    rc = cache.reverse_cache
    objective = rc.objective
    for batch_index = 1:cache.batch_size
        copyto!(rc.theta, view(params, :, batch_index))
        destination[batch_index] = try
            Base.invokelatest(objective, rc.theta)
        catch
            -Inf
        end
    end
    return destination
end

# --- lane compaction (issue #160) -------------------------------------------
#
# Masked batched NUTS runs a FULL-WIDTH batched gradient at every leapfrog leaf,
# even once most chains have finished/diverged (measured waste 60.8% at 64
# chains, 68.3% at 256). The batched gradient of column c depends ONLY on
# `params[:, c]` -- columns are fully independent -- so once the active fraction
# drops below a threshold the sampler gathers the active columns, evaluates the
# gradient over just those k columns, and scatters the results back to the active
# lanes. Because gather -> gradient -> scatter is a pure permutation over
# independent columns, the active lanes receive BITWISE-identical logjoint/
# gradient values; the inactive lanes are downstream don't-cares (gated out by
# `valid`), so leaving their destination entries stale is equivalent to the
# full-width path overwriting them with never-read values. No RNG lives in the
# gradient, so the masked-doubling draw order is untouched.

# Compact once the active fraction falls below this. Above it, the gather/scatter
# + narrower-batch bookkeeping overhead would eat the saved lane work, so the
# full-width path stays in charge (and the all-active leapfrog step -- the common
# early-round case -- is completely unchanged, including its zero-allocation
# guarantee).
const _BATCHED_LANE_COMPACTION_ACTIVE_FRACTION = 0.5

# Only the analytic backend tier is compacted: its per-column arithmetic is
# provably batch-width-independent (the observed-loop reduction runs per chain and
# the sufficient-statistics fusion is per column), so a k-column evaluation is
# bitwise identical to the same columns inside the full C-column batch. The
# ForwardDiff flat/column tiers differentiate a batch-shaped objective, so they
# stay full width. A chain-block thread plan (issue #143) also keeps the full
# width -- compacting would drop it to a single analytic block; threaded-block
# compaction is a follow-up.
#
# The backend cache's args/constraints must be SHARED (a single tuple / ChoiceMap
# broadcast to every column), not PER-COLUMN vectors: the compact call reuses the
# cache's own fixed args/constraints, so a k-column params slice with a length-C
# per-column constraints vector would mis-length the observed-value gather. Shared
# args/constraints apply identically to any column subset, so the gathered lanes
# score exactly as they would full width. Per-column args/constraints keep the
# full-width path (compacting them would need a rebuilt cache).
function _batched_lane_compaction_beneficial(cache::BatchedLogjointGradientCache, active_count::Int)
    backend_cache = cache.backend_cache
    backend_cache === nothing && return false
    cache.thread_plan === nothing || return false
    backend_cache.args isa Tuple || return false
    backend_cache.constraints isa ChoiceMap || return false
    active_count >= 1 || return false
    active_count < cache.batch_size || return false
    return active_count < _BATCHED_LANE_COMPACTION_ACTIVE_FRACTION * cache.batch_size
end

# Gather the `active_count` active columns of `params` to the front of the compact
# scratch, evaluate the analytic backend logjoint+gradient over just those
# columns, and scatter back into `values_destination` / `gradient_destination` at
# the active lanes (inactive lanes left untouched -- don't-cares). Returns `true`
# on success; returns `false` WITHOUT completing if the analytic tier hits a
# recoverable capability gap or (reject mode, issue #157) a lane parameter-
# validation throw, so the caller redoes the call full width -- where the
# per-column degradation in `_batched_backend_gradient_or_columns!` handles it and
# overwrites any partial scatter.
function _batched_compact_logjoint_and_gradient!(
    values_destination::AbstractVector,
    gradient_destination::AbstractMatrix,
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
    active::AbstractVector{Bool},
    active_count::Int,
)
    backend_cache = cache.backend_cache
    parameter_count = cache.parameter_count
    compact_params = cache.compact_params
    index = cache.compact_index
    slot = 0
    @inbounds for chain_index in eachindex(active)
        active[chain_index] || continue
        slot += 1
        index[slot] = chain_index
        for row = 1:parameter_count
            compact_params[row, slot] = params[row, chain_index]
        end
    end

    totals = view(cache.compact_logjoint, 1:active_count)
    gradients = view(cache.compact_gradient, :, 1:active_count)
    try
        _batched_backend_logjoint_and_gradient_unconstrained!(
            totals,
            gradients,
            cache.model,
            backend_cache,
            view(compact_params, :, 1:active_count),
        )
    catch err
        reject = backend_cache.workspace.environment.reject_invalid_parameters
        if err isa BatchedBackendFallback || (reject && (err isa ArgumentError || err isa DomainError))
            return false
        end
        rethrow()
    end

    @inbounds for slot = 1:active_count
        chain_index = index[slot]
        values_destination[chain_index] = cache.compact_logjoint[slot]
        for row = 1:parameter_count
            gradient_destination[row, chain_index] = cache.compact_gradient[row, slot]
        end
    end
    return true
end

function batched_logjoint_gradient_unconstrained(
    cache::BatchedLogjointGradientCache,
    params::AbstractMatrix,
)
    return copy(batched_logjoint_gradient_unconstrained!(cache, params))
end

function batched_logjoint(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_constrained_params(model, params, constraints)
    batch_args = _validate_batched_args(model, args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    batch_size == 0 && return float(eltype(params))[]

    # The backend totals and environment buffers inherit the params element type
    # (issue #92): an integer-typed constrained matrix would make the totals
    # integer and hit InexactError when a float log-density is accumulated.
    # Promote silently so integer observation/parameter matrices just work.
    eltype(params) <: AbstractFloat || (params = float.(params))

    workspace = BatchedLogjointWorkspace(model, batch_constraints)
    # the backend plan for a dependent-transform model scores in z space, so
    # the CONSTRAINED entry point must use the per-column reference instead
    if !isnothing(workspace.backend_plan) && !_has_dependent_transforms(workspace.layout)
        try
            return _logjoint_with_batched_backend!(workspace, params, batch_args, batch_constraints)
        catch err
            if !(err isa BatchedBackendFallback)
                rethrow()
            end
        end
    end
    return _fallback_batched_logjoint!(workspace, params, batch_args, batch_constraints)
end

function batched_logjoint_unconstrained(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    batch_size = _validate_batched_unconstrained_params(model, params, constraints)
    batch_args = _validate_batched_args(model, args, batch_size)
    batch_constraints = _validate_batched_constraints(constraints, batch_size)
    batch_size == 0 && return float(eltype(params))[]

    workspace = BatchedLogjointWorkspace(model, batch_constraints)
    values = Vector{float(eltype(params))}(undef, batch_size)
    return _batched_logjoint_unconstrained_with_workspace!(values, model, workspace, params, batch_args, batch_constraints)
end

function batched_logjoint_gradient_unconstrained(
    model::TeaModel,
    params::AbstractMatrix,
    args=(),
    constraints=choicemap(),
)
    cache = BatchedLogjointGradientCache(model, params, args, constraints)
    return batched_logjoint_gradient_unconstrained(cache, params)
end
