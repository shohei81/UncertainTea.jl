# Static nested sampling (Skilling 2006) -- a CPU-only, gradient-free evidence
# estimator with a quantified (information-based) uncertainty and natural
# multimodal coverage. This is the package's third, independent log-evidence
# estimator, complementing importance sampling and tempered SMC
# (`log_evidence`, src/inference/smc/core.jl); see issue #234.
#
# WHY IT IS CORRECT (the crux is the likelihood-vs-prior split and the
# constrained-prior replacement):
#
#   The evidence is Z = ∫ L(θ) dπ(θ), the likelihood integrated against the
#   prior measure. Nested sampling turns this 1-D-in-prior-volume integral into
#   a sum over ordered prior draws. Everything runs in UNCONSTRAINED space so
#   every existing transform is reused; the prior measure used throughout is the
#   pushforward of the model prior through the transform. The evidence integral
#   is invariant to the parameterization, so working unconstrained gives the same
#   Z as working constrained -- as long as the live points are prior draws in
#   that space and the replacement draws from the SAME prior restricted to the
#   likelihood constraint.
#
#   Likelihood vs prior split: for an unconstrained position u mapping to the
#   constrained θ with log-abs-det J,
#       logjoint_unconstrained(u) = logprior_constrained(θ) + logJ + logL(θ),
#   and the pointwise-likelihood walk (src/evaluator_pointwise.jl) records
#   exactly the observation log-densities, whose sum IS logL(θ). So
#       logL           = Σ observation log-densities (parameterization-free),
#       logprior_u(u)  = [logjoint_constrained(θ) - logL] + logJ.
#   The Jacobian cancels out of logL (a likelihood is not a density in θ and does
#   not transform), which is why logL is computed the same way in either space.
#
# CONSTRAINED-PRIOR REPLACEMENT (`replacement`):
#   * `:rwmh` (default) -- seed a short random-walk Metropolis chain at a random
#     SURVIVING live point (which already satisfies logL > L*), propose Gaussian
#     steps whose per-coordinate scale is the live ensemble's spread, and accept
#     iff logL' > L* AND log(rand) < logprior_u(u') - logprior_u(u). That target
#     is the prior restricted to {logL > L*}; symmetric proposals need only the
#     prior-density ratio, so NO gradient is used -- the genuinely new capability
#     (works on non-differentiable likelihoods). Its correctness is only as good
#     as the walk's mixing: reliable in LOW dimension; long walks / many live
#     points are needed as dimension grows. See scope note below.
#   * `:rejection` -- draw from the prior until logL > L*. Exact i.i.d. draws (the
#     gold standard) but the acceptance rate decays like the prior volume, so it
#     is only practical for very low-D problems / early termination. Useful as a
#     correctness oracle.
#
# SCOPE / NON-GOALS (honest):
#   * Static nested sampling only -- no dynamic NS, no MultiNest-style ellipsoidal
#     decomposition, no device story (CPU-only, gradient-free by design).
#   * Continuous latents only (needs the unconstrained transform, as HMC does);
#     `marginalize=:enumerate` models are unsupported (the pointwise-likelihood
#     walk they rely on rejects entangled enumeration structures).
#   * Validated for correctness on LOW-dimensional problems (evidence matched to
#     analytic within uncertainty; both modes of a bimodal 1-D posterior
#     recovered). The `:rwmh` replacement's mixing -- hence evidence quality --
#     degrades in higher dimensions and needs a longer `num_walk`; that regime is
#     a follow-up, not a validated claim here.

"""
    NestedSamplingResult

Result of [`nested_sampling`](@ref). Holds the dead + remaining-live points as
importance-weighted posterior draws plus the log-evidence and its uncertainty.

Fields of interest (accessors below): `log_evidence_estimate`,
`log_evidence_error` (the information-based `sqrt(H/N)` one-sigma), `information`
(`H`), `normalized_weights`, and the `constrained_samples` / `unconstrained_samples`
matrices whose columns are the draws in draw order.
"""
struct NestedSamplingResult
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    unconstrained_samples::Matrix{Float64}
    constrained_samples::Matrix{Float64}
    loglikelihoods::Vector{Float64}
    logweights::Vector{Float64}
    normalized_weights::Vector{Float64}
    log_evidence_estimate::Float64
    log_evidence_error::Float64
    information::Float64
    effective_sample_size::Float64
    num_live_points::Int
    num_iterations::Int
    replacement::Symbol
    evaluation_backend::Symbol
end

function log_evidence(result::NestedSamplingResult)
    return result.log_evidence_estimate
end

"""
    log_evidence_error(result::NestedSamplingResult) -> Float64

The one-standard-deviation uncertainty on `log_evidence` from nested sampling,
estimated as `sqrt(H / N)` with information `H` and `N` live points (Skilling
2006).
"""
function log_evidence_error(result::NestedSamplingResult)
    return result.log_evidence_error
end

"""
    information(result::NestedSamplingResult) -> Float64

The information `H` (the negative relative entropy of the prior with respect to
the posterior, in nats) accumulated by nested sampling; drives the evidence
uncertainty `sqrt(H / N)`.
"""
function information(result::NestedSamplingResult)
    return result.information
end

function ess(result::NestedSamplingResult)
    return result.effective_sample_size
end

function numsamples(result::NestedSamplingResult)
    return size(result.unconstrained_samples, 2)
end

function Base.show(io::IO, result::NestedSamplingResult)
    print(
        io,
        "NestedSamplingResult(model=",
        result.model.name,
        ", live=",
        result.num_live_points,
        ", iterations=",
        result.num_iterations,
        ", replacement=:",
        result.replacement,
        ", log_evidence=",
        round(result.log_evidence_estimate; digits=4),
        " ± ",
        round(result.log_evidence_error; digits=4),
        ")",
    )
end

# Posterior predictive from nested-sampling draws: resample by the normalized
# importance weights (systematic), then run the shared predictive kernel. Mirrors
# the importance-sampling `predict` path.
function predict(
    model::TeaModel,
    args::Tuple,
    result::NestedSamplingResult;
    num_draws::Int=size(result.constrained_samples, 2),
    rng::AbstractRNG=Random.default_rng(),
)
    num_draws > 0 || throw(ArgumentError("predict requires num_draws > 0"))
    ancestors = _systematic_resample_indices(result.normalized_weights, num_draws, rng)
    columns = (view(result.constrained_samples, :, ancestor) for ancestor in ancestors)
    return _predictive_from_param_columns(model, args, columns, rng)
end

# log(exp(a) + exp(b)) without overflow; -Inf-safe.
function _ns_logaddexp(a::Float64, b::Float64)
    a == -Inf && return b
    b == -Inf && return a
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

# Sum of the per-observation log-densities at a constrained parameter vector --
# i.e. logL(θ) -- reusing the pointwise-likelihood walk (src/evaluator_pointwise.jl)
# against a pre-resolved signature plan. `records` is a reusable scratch buffer.
function _ns_loglikelihood!(
    records::Vector{Pair{Any,Float64}},
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    constrained::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    plan = resolved.plan
    env = PlanEnvironment(plan.environment_layout)
    for (slot, value) in zip(plan.environment_layout.argument_slots, args)
        _environment_set!(env, slot, value)
    end
    empty!(records)
    _record_compiled_steps(resolved.compiled.steps, env, constrained, constraints, records)
    total = 0.0
    for entry in records
        total += last(entry)
    end
    return total
end

# Evaluate (logL, logprior_unconstrained, constrained) at an unconstrained
# position under Stan-style reject semantics: an invalid parameter scores
# (-Inf, -Inf) instead of throwing (a rejected proposal), matching how the
# samplers evaluate. Returns the constrained vector so the caller can store it.
#
# logjoint_constrained(θ) = logprior_constrained(θ) + logL(θ); the pointwise walk
# gives logL, so logprior_u = (logjoint_constrained - logL) + logJ.
function _ns_density!(
    records::Vector{Pair{Any,Float64}},
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    u::AbstractVector,
    completed_args::Tuple,
    constraints::ChoiceMap,
)
    constrained, logabsdet = _transform_to_constrained_with_logabsdet(
        model, resolved, u, completed_args, constraints; reject_invalid_parameters=true,
    )
    logjoint_c = _logjoint(
        model, resolved, constrained, completed_args, constraints, nothing;
        reject_invalid_parameters=true,
    )
    if !isfinite(logjoint_c)
        return -Inf, -Inf, constrained
    end
    logL = _ns_loglikelihood!(records, model, resolved, constrained, completed_args, constraints)
    logprior_u = (logjoint_c - logL) + logabsdet
    return logL, logprior_u, constrained
end

# One prior draw in unconstrained space, with its (logL, logprior_u, constrained).
function _ns_prior_draw!(
    records::Vector{Pair{Any,Float64}},
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    args::Tuple,
    completed_args::Tuple,
    constraints::ChoiceMap,
    rng::AbstractRNG,
)
    u = _initial_hmc_position(model, resolved, args, constraints, nothing, rng)
    logL, logprior_u, constrained = _ns_density!(records, model, resolved, u, completed_args, constraints)
    return u, logL, logprior_u, constrained
end

# Rejection replacement: draw from the prior until logL > logL_star. Returns
# `nothing` if `max_attempts` is exhausted (the caller falls back to the seed).
function _ns_replace_rejection!(
    records::Vector{Pair{Any,Float64}},
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    args::Tuple,
    completed_args::Tuple,
    constraints::ChoiceMap,
    logL_star::Float64,
    max_attempts::Int,
    rng::AbstractRNG,
)
    for _ = 1:max_attempts
        u, logL, logprior_u, constrained =
            _ns_prior_draw!(records, model, resolved, args, completed_args, constraints, rng)
        if logL > logL_star
            return u, logL, constrained
        end
    end
    return nothing
end

# Random-walk Metropolis replacement targeting the prior restricted to
# {logL > logL_star}. `seed_u` is a surviving live point (already inside the
# constraint); `step` is the per-coordinate Gaussian proposal scale. Symmetric
# proposal => acceptance uses only the prior-density ratio (gradient-free).
function _ns_replace_rwmh!(
    records::Vector{Pair{Any,Float64}},
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    completed_args::Tuple,
    constraints::ChoiceMap,
    seed_u::AbstractVector,
    seed_logL::Float64,
    seed_logprior_u::Float64,
    logL_star::Float64,
    step::AbstractVector,
    num_walk::Int,
    rng::AbstractRNG,
)
    dim = length(seed_u)
    u = collect(Float64, seed_u)
    proposal = Vector{Float64}(undef, dim)
    logL_u = seed_logL
    logprior_u = seed_logprior_u
    accepted = 0
    for _ = 1:num_walk
        for d = 1:dim
            proposal[d] = u[d] + step[d] * randn(rng)
        end
        logL_p, logprior_p, _ = _ns_density!(records, model, resolved, proposal, completed_args, constraints)
        if logL_p > logL_star && log(rand(rng)) < (logprior_p - logprior_u)
            copyto!(u, proposal)
            logL_u = logL_p
            logprior_u = logprior_p
            accepted += 1
        end
    end
    # Recompute the constrained vector for the final accepted position.
    _, _, constrained = _ns_density!(records, model, resolved, u, completed_args, constraints)
    return u, logL_u, constrained, accepted
end

# Per-coordinate proposal scale for the random walk: the standard deviation of
# the live ensemble in each unconstrained coordinate, times `step_scale`. This
# makes the walk scale-aware (a fixed step would mis-fit the constrained region
# as it shrinks). A degenerate (near-zero) spread falls back to `step_scale`.
function _ns_walk_step!(
    step::Vector{Float64},
    live_positions::Matrix{Float64},
    step_scale::Float64,
)
    dim, n = size(live_positions)
    for d = 1:dim
        mean_d = 0.0
        for i = 1:n
            mean_d += live_positions[d, i]
        end
        mean_d /= n
        var_d = 0.0
        for i = 1:n
            delta = live_positions[d, i] - mean_d
            var_d += delta * delta
        end
        var_d /= max(n - 1, 1)
        sd = sqrt(var_d)
        step[d] = step_scale * (sd > 0 ? sd : 1.0)
    end
    return step
end

"""
    nested_sampling(model, args=(), constraints=choicemap(); num_live_points=100, kwargs...)

Static nested sampling (Skilling 2006): a CPU-only, gradient-free estimator of the
log-evidence `log Z = log ∫ L(θ) dπ(θ)` with an information-based uncertainty, plus
importance-weighted posterior draws as a by-product. Multimodal posteriors are
handled naturally because live points populate every surviving mode.

`N = num_live_points` points are drawn from the prior (in unconstrained space,
reusing the transform machinery). Each iteration removes the lowest-likelihood
live point (recording it as a dead point with its prior-volume weight) and
replaces it by a draw from the prior restricted to `logL > L*`.

Keywords:
- `num_live_points::Int=100` — the live set size `N`.
- `replacement::Symbol=:rwmh` — the constrained-prior draw. `:rwmh` runs a short
  random-walk Metropolis chain seeded at a surviving live point (default, works on
  non-differentiable models, best in low dimension); `:rejection` draws from the
  prior until the constraint holds (exact but only practical in very low dimension).
- `num_walk::Int=25` — random-walk steps per replacement (`:rwmh`).
- `step_scale::Float64=0.5` — random-walk proposal scale as a fraction of the live
  ensemble's per-coordinate spread (`:rwmh`).
- `max_rejection_attempts::Int=100000` — attempt cap per replacement (`:rejection`).
- `dlogz::Float64=0.01` — termination tolerance on the remaining log-evidence.
- `max_iterations::Int=10000` — hard iteration cap.
- `rng::AbstractRNG=Random.default_rng()` — seed this for determinism.

Returns a [`NestedSamplingResult`](@ref); use `log_evidence`, `log_evidence_error`,
and the normalized-weight `constrained_samples` for posterior moments.
"""
function nested_sampling(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_live_points::Int=100,
    replacement::Symbol=:rwmh,
    num_walk::Int=25,
    step_scale::Float64=0.5,
    max_rejection_attempts::Int=100_000,
    dlogz::Float64=0.01,
    max_iterations::Int=10_000,
    rng::AbstractRNG=Random.default_rng(),
)
    num_live_points >= 2 ||
        throw(ArgumentError("nested_sampling requires num_live_points >= 2, got $num_live_points"))
    (replacement === :rwmh || replacement === :rejection) ||
        throw(ArgumentError("nested_sampling replacement must be :rwmh or :rejection, got :$replacement"))
    num_walk >= 1 || throw(ArgumentError("nested_sampling requires num_walk >= 1, got $num_walk"))
    max_iterations >= 1 ||
        throw(ArgumentError("nested_sampling requires max_iterations >= 1, got $max_iterations"))

    resolved = _resolve_signature_plan(model, constraints)
    completed_args = _complete_model_args(model, args)
    layout = resolved.plan.parameter_layout
    dim = parametercount(layout)                     # unconstrained latent count
    constrained_dim = parametervaluecount(layout)    # constrained latent count
    dim >= 1 ||
        throw(ArgumentError("nested_sampling requires at least one latent parameter (model has none under this conditioning)"))
    N = num_live_points
    records = Pair{Any,Float64}[]

    # --- initial live set from the prior ---
    live_positions = Matrix{Float64}(undef, dim, N)
    live_constrained = Matrix{Float64}(undef, constrained_dim, N)
    live_logL = Vector{Float64}(undef, N)
    live_logprior = Vector{Float64}(undef, N)
    for i = 1:N
        u, logL, logprior_u, constrained =
            _ns_prior_draw!(records, model, resolved, args, completed_args, constraints, rng)
        isfinite(logL) ||
            throw(ArgumentError("nested_sampling drew a prior point with non-finite likelihood; check the model/constraints"))
        live_positions[:, i] = u
        live_constrained[:, i] = constrained
        live_logL[i] = logL
        live_logprior[i] = logprior_u
    end

    # --- shrinkage loop ---
    dead_positions = Vector{Vector{Float64}}()
    dead_constrained = Vector{Vector{Float64}}()
    dead_logL = Float64[]
    dead_logw = Float64[]

    logX_prev = 0.0            # log X_0 = log 1
    logZ = -Inf
    step = Vector{Float64}(undef, dim)
    iteration = 0
    while iteration < max_iterations
        iteration += 1

        kmin = argmin(live_logL)
        logL_star = live_logL[kmin]
        logX = -iteration / N                        # deterministic volume estimate
        # weight of the shell removed this step: L* * (X_{i-1} - X_i)
        dX = exp(logX_prev) - exp(logX)
        logw = logL_star + log(dX)

        push!(dead_positions, copy(view(live_positions, :, kmin)))
        push!(dead_constrained, copy(view(live_constrained, :, kmin)))
        push!(dead_logL, logL_star)
        push!(dead_logw, logw)
        logZ = _ns_logaddexp(logZ, logw)
        logX_prev = logX

        # termination: remaining live evidence estimate is negligible vs logZ.
        logL_max_live = maximum(live_logL)
        logZ_remaining = logL_max_live + logX
        if _ns_logaddexp(logZ, logZ_remaining) - logZ < dlogz
            break
        end

        # replace the removed point with a draw from the prior | logL > logL_star.
        if replacement === :rejection
            drawn = _ns_replace_rejection!(
                records, model, resolved, args, completed_args, constraints,
                logL_star, max_rejection_attempts, rng,
            )
            if drawn === nothing
                # Could not satisfy the constraint within the attempt budget:
                # stop rather than bias the run with a stale copy.
                break
            end
            u, logL_new, constrained_new = drawn
            logprior_new = _ns_density!(records, model, resolved, u, completed_args, constraints)[2]
        else
            _ns_walk_step!(step, live_positions, step_scale)
            # seed from a random SURVIVING live point (not the one just removed).
            seed_index = _ns_random_survivor(rng, N, kmin)
            u, logL_new, constrained_new, _ = _ns_replace_rwmh!(
                records, model, resolved, completed_args, constraints,
                view(live_positions, :, seed_index),
                live_logL[seed_index], live_logprior[seed_index],
                logL_star, step, num_walk, rng,
            )
            logprior_new = _ns_density!(records, model, resolved, u, completed_args, constraints)[2]
        end

        live_positions[:, kmin] = u
        live_constrained[:, kmin] = constrained_new
        live_logL[kmin] = logL_new
        live_logprior[kmin] = logprior_new
    end

    # --- remaining live points: split the leftover volume X_m equally ---
    logX_final = logX_prev
    log_over_N = log(N)
    for i = 1:N
        logw = live_logL[i] + logX_final - log_over_N
        push!(dead_positions, copy(view(live_positions, :, i)))
        push!(dead_constrained, copy(view(live_constrained, :, i)))
        push!(dead_logL, live_logL[i])
        push!(dead_logw, logw)
        logZ = _ns_logaddexp(logZ, logw)
    end

    # --- assemble result: normalized weights, information H, uncertainty ---
    total = length(dead_logw)
    unconstrained_samples = Matrix{Float64}(undef, dim, total)
    constrained_samples = Matrix{Float64}(undef, length(dead_constrained[1]), total)
    for j = 1:total
        unconstrained_samples[:, j] = dead_positions[j]
        constrained_samples[:, j] = dead_constrained[j]
    end

    normalized = Vector{Float64}(undef, total)
    H = 0.0
    for j = 1:total
        w = exp(dead_logw[j] - logZ)
        normalized[j] = w
        if w > 0 && isfinite(dead_logL[j])
            H += w * (dead_logL[j] - logZ)
        end
    end
    H = max(H, 0.0)
    logZ_error = sqrt(H / N)
    ess_value = _effective_sample_size(normalized)

    return NestedSamplingResult(
        model,
        args,
        constraints,
        unconstrained_samples,
        constrained_samples,
        copy(dead_logL),
        copy(dead_logw),
        normalized,
        logZ,
        logZ_error,
        H,
        ess_value,
        N,
        iteration,
        replacement,
        _batched_evaluation_backend(model),
    )
end

# A random surviving live-point index (any index other than the one removed).
function _ns_random_survivor(rng::AbstractRNG, N::Int, removed::Int)
    N == 1 && return 1
    idx = rand(rng, 1:(N-1))
    return idx < removed ? idx : idx + 1
end
