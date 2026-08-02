# Pathfinder variational initialization (Zhang, Carpenter, Gelman, Vehtari
# 2022): run L-BFGS toward the mode of the unconstrained log joint, build a
# Gaussian approximation N(mu_l, Sigma_l) at every iterate from the L-BFGS
# inverse-Hessian estimate (mu_l = x_l + Sigma_l g_l, one quasi-Newton step),
# score each by a Monte Carlo ELBO, and keep the best. Multi-path mode pools
# draws from several independent paths by importance resampling. Sigma_l is
# materialized densely from the two-loop recursion -- O(d^2 m) per iterate,
# fine for the CPU reference scale; the paper's O(d m) factorized form is a
# possible future optimization.

struct PathfinderResult
    model::TeaModel
    args::Tuple
    constraints::ChoiceMap
    location::Vector{Float64}
    covariance::Matrix{Float64}
    cholesky_lower::Matrix{Float64}
    # d x num_draws unconstrained draws: from the best Gaussian for a single
    # path, importance-resampled across paths for num_paths > 1.
    draws::Matrix{Float64}
    elbo::Float64
    elbo_history::Vector{Float64}
    num_paths::Int
    best_path::Int
    converged::Bool
end

function Base.show(io::IO, result::PathfinderResult)
    print(
        io,
        "PathfinderResult(dim=$(length(result.location)), paths=$(result.num_paths), ",
        "elbo=$(round(result.elbo; sigdigits=4)), draws=$(size(result.draws, 2)), ",
        "converged=$(result.converged))",
    )
end

function _lbfgs_dense_inverse_hessian(
    dim::Int,
    s_history::Vector{Vector{Float64}},
    y_history::Vector{Vector{Float64}},
    rho_history::Vector{Float64},
)
    covariance = Matrix{Float64}(undef, dim, dim)
    basis = zeros(Float64, dim)
    column = Vector{Float64}(undef, dim)
    for j = 1:dim
        basis[j] = 1.0
        _lbfgs_apply_inverse_hessian!(column, basis, s_history, y_history, rho_history)
        covariance[:, j] = column
        basis[j] = 0.0
    end
    # two-loop output is symmetric up to floating-point noise
    for j = 1:dim, i = 1:(j-1)
        value = 0.5 * (covariance[i, j] + covariance[j, i])
        covariance[i, j] = value
        covariance[j, i] = value
    end
    return covariance
end

# One candidate Gaussian per accepted L-BFGS iterate. `elbo` is -Inf whenever
# any ELBO draw hit a non-finite log joint (the exact ELBO of a Gaussian with
# mass outside the target support); `finite_elbo`/`finite_fraction` keep the
# average over the finite draws as a tie-breaker when every candidate leaks.
struct _PathfinderCandidate
    location::Vector{Float64}
    covariance::Matrix{Float64}
    cholesky_lower::Matrix{Float64}
    logdet_covariance::Float64
    elbo::Float64
    finite_elbo::Float64
    finite_fraction::Float64
end

function _pathfinder_gaussian_logdensity(candidate::_PathfinderCandidate, point::AbstractVector)
    dim = length(candidate.location)
    standardized = candidate.location .- point
    standardized .*= -1.0
    ldiv!(LowerTriangular(candidate.cholesky_lower), standardized)
    return -0.5 *
           (candidate.logdet_covariance + dot(standardized, standardized) + dim * log(2.0 * pi))
end

function _pathfinder_candidate(
    objective,
    x::Vector{Float64},
    gradient::Vector{Float64},
    s_history::Vector{Vector{Float64}},
    y_history::Vector{Vector{Float64}},
    rho_history::Vector{Float64},
    num_elbo_draws::Int,
    rng::AbstractRNG,
)
    dim = length(x)
    covariance = _lbfgs_dense_inverse_hessian(dim, s_history, y_history, rho_history)
    factor = cholesky(Symmetric(covariance); check=false)
    issuccess(factor) || return nothing
    cholesky_lower = Matrix(factor.L)
    logdet_covariance = 2.0 * sum(log, diag(factor.L))

    location = copy(x)
    mul!(location, covariance, gradient, 1.0, 1.0)

    log_two_pi = log(2.0 * pi)
    elbo_total = 0.0
    elbo_count = 0
    noise = Vector{Float64}(undef, dim)
    draw = Vector{Float64}(undef, dim)
    for _ = 1:num_elbo_draws
        randn!(rng, noise)
        copyto!(draw, location)
        mul!(draw, cholesky_lower, noise, 1.0, 1.0)
        logp = objective(draw)
        isfinite(logp) || continue
        logq = -0.5 * (logdet_covariance + dot(noise, noise) + dim * log_two_pi)
        elbo_total += logp - logq
        elbo_count += 1
    end
    elbo_count > 0 || return nothing
    finite_elbo = elbo_total / elbo_count
    # a draw outside the target support makes the exact ELBO -Inf
    elbo = elbo_count == num_elbo_draws ? finite_elbo : -Inf
    return _PathfinderCandidate(
        location,
        covariance,
        cholesky_lower,
        logdet_covariance,
        elbo,
        finite_elbo,
        elbo_count / num_elbo_draws,
    )
end

function _pathfinder_single_path(
    objective,
    gradient!,
    x0::Vector{Float64};
    history::Int,
    max_iters::Int,
    g_tol::Float64,
    num_elbo_draws::Int,
    rng::AbstractRNG,
)
    candidates = _PathfinderCandidate[]
    elbo_history = Float64[]
    callback =
        (x, fx, g, s_history, y_history, rho_history) -> begin
            candidate = _pathfinder_candidate(
                objective,
                x,
                g,
                s_history,
                y_history,
                rho_history,
                num_elbo_draws,
                rng,
            )
            isnothing(candidate) || push!(candidates, candidate)
            push!(elbo_history, isnothing(candidate) ? -Inf : candidate.elbo)
            return nothing
        end
    _, _, _, _, _, converged = _lbfgs_maximize(
        objective,
        gradient!,
        x0;
        history=history,
        max_iters=max_iters,
        g_tol=g_tol,
        iteration_callback=callback,
    )
    if isempty(candidates)
        # no accepted step (already at the mode, or immediate line-search
        # failure): fall back to a unit-covariance candidate at the start point
        gradient = similar(x0)
        gradient!(gradient, x0)
        candidate = _pathfinder_candidate(
            objective,
            x0,
            gradient,
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}(),
            Float64[],
            num_elbo_draws,
            rng,
        )
        isnothing(candidate) && return nothing, elbo_history, converged
        push!(candidates, candidate)
        push!(elbo_history, candidate.elbo)
    end
    best = argmax([
        (isfinite(candidate.elbo), candidate.finite_fraction, candidate.finite_elbo) for
        candidate in candidates
    ])
    return candidates[best], elbo_history, converged
end

function _pathfinder_gaussian_draws!(
    destination::AbstractMatrix,
    log_density::AbstractVector,
    candidate::_PathfinderCandidate,
    rng::AbstractRNG,
)
    dim = size(destination, 1)
    log_two_pi = log(2.0 * pi)
    noise = Vector{Float64}(undef, dim)
    for j in axes(destination, 2)
        randn!(rng, noise)
        column = view(destination, :, j)
        copyto!(column, candidate.location)
        mul!(column, candidate.cholesky_lower, noise, 1.0, 1.0)
        log_density[j] =
            -0.5 * (candidate.logdet_covariance + dot(noise, noise) + dim * log_two_pi)
    end
    return destination
end

"""
    pathfinder(model, args=(), constraints=choicemap();
               num_paths=1, num_draws=100, num_elbo_draws=25, history=6,
               max_iters=200, g_tol=1e-6, initial_params=nothing,
               rng=Random.default_rng()) -> PathfinderResult

Pathfinder variational initialization (Zhang et al. 2022): L-BFGS runs toward
the posterior mode in unconstrained space, every accepted iterate yields a
Gaussian approximation built from the L-BFGS inverse-Hessian estimate, and a
Monte Carlo ELBO selects the best one. With `num_paths > 1`, the paths start
from independent prior initializations and the returned `draws` pool the
per-path draws by importance resampling against the equal mixture of the
path Gaussians (Pareto smoothing of the weights is a possible refinement).
Every returned draw has a finite log joint. Gradients always use ForwardDiff;
there is no `adtype` selection here (only the batched samplers accept `adtype`).

The result can be passed directly as `initial_params` to `hmc`/`nuts`/
`nuts_chains`/`batched_hmc`/`batched_nuts`, which then start each chain from
a Pathfinder draw instead of a prior draw.
"""
function pathfinder(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    num_paths::Int=1,
    num_draws::Int=100,
    num_elbo_draws::Int=25,
    history::Int=6,
    max_iters::Int=200,
    g_tol::Float64=1e-6,
    initial_params=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    num_paths > 0 || throw(ArgumentError("pathfinder requires num_paths > 0"))
    num_draws > 0 || throw(ArgumentError("pathfinder requires num_draws > 0"))
    num_elbo_draws > 0 || throw(ArgumentError("pathfinder requires num_elbo_draws > 0"))
    parametercount(_conditioned_parameter_layout(model, constraints)) > 0 ||
        throw(ArgumentError("pathfinder requires at least one parameterized latent choice"))

    objective = theta -> logjoint_unconstrained(model, theta, args, constraints)
    seed = _initial_hmc_position(model, args, constraints, initial_params, rng)
    cache = _logjoint_gradient_cache(model, seed, args, constraints)
    gradient! = (g, x) -> copyto!(g, _logjoint_gradient!(cache, x))

    path_candidates = _PathfinderCandidate[]
    path_elbo_histories = Vector{Float64}[]
    path_converged = Bool[]
    for path_index = 1:num_paths
        x0 =
            path_index == 1 ? seed :
            _initial_hmc_position(model, args, constraints, nothing, rng)
        candidate, elbo_history, converged = _pathfinder_single_path(
            objective,
            gradient!,
            x0;
            history=history,
            max_iters=max_iters,
            g_tol=g_tol,
            num_elbo_draws=num_elbo_draws,
            rng=rng,
        )
        isnothing(candidate) && continue
        push!(path_candidates, candidate)
        push!(path_elbo_histories, elbo_history)
        push!(path_converged, converged)
    end
    isempty(path_candidates) && throw(
        ErrorException(
            "pathfinder found no usable Gaussian approximation on any path; " *
            "the log joint may be non-finite around the initialization",
        ),
    )

    best_path = argmax([candidate.elbo for candidate in path_candidates])
    best = path_candidates[best_path]
    dim = length(best.location)

    draws = Matrix{Float64}(undef, dim, num_draws)
    if length(path_candidates) == 1
        # rejection-filter so every returned draw has a finite log joint (the
        # best Gaussian may still carry a little mass outside the support)
        noise = Vector{Float64}(undef, dim)
        filled = 0
        attempts = 0
        max_attempts = 200 * num_draws
        while filled < num_draws
            attempts += 1
            attempts <= max_attempts || throw(
                ErrorException(
                    "pathfinder could not draw $num_draws points with a finite log joint; " *
                    "the selected Gaussian lies mostly outside the target support",
                ),
            )
            randn!(rng, noise)
            column = view(draws, :, filled + 1)
            copyto!(column, best.location)
            mul!(column, best.cholesky_lower, noise, 1.0, 1.0)
            isfinite(objective(column)) || continue
            filled += 1
        end
    else
        # importance-resample pooled per-path draws toward the target; the
        # proposal is the equal mixture of the path Gaussians, so the weight
        # denominator is the mixture density, not the generating component's
        per_path = cld(num_draws, length(path_candidates))
        total = per_path * length(path_candidates)
        pooled = Matrix{Float64}(undef, dim, total)
        log_weights = Vector{Float64}(undef, total)
        log_density = Vector{Float64}(undef, per_path)
        component_log_density = Vector{Float64}(undef, length(path_candidates))
        for (path_index, candidate) in enumerate(path_candidates)
            offset = (path_index - 1) * per_path
            block = view(pooled, :, (offset+1):(offset+per_path))
            _pathfinder_gaussian_draws!(block, log_density, candidate, rng)
        end
        for j = 1:total
            point = view(pooled, :, j)
            logp = objective(point)
            if !isfinite(logp)
                log_weights[j] = -Inf
                continue
            end
            for (candidate_index, candidate) in enumerate(path_candidates)
                component_log_density[candidate_index] =
                    _pathfinder_gaussian_logdensity(candidate, point)
            end
            max_component = maximum(component_log_density)
            log_mixture =
                max_component +
                log(sum(exp(value - max_component) for value in component_log_density)) -
                log(length(path_candidates))
            log_weights[j] = logp - log_mixture
        end
        max_log_weight = maximum(log_weights)
        isfinite(max_log_weight) ||
            throw(ErrorException("pathfinder importance weights are all non-finite"))
        weights = exp.(log_weights .- max_log_weight)
        weights ./= sum(weights)
        cumulative = cumsum(weights)
        for j = 1:num_draws
            u = rand(rng)
            index = searchsortedfirst(cumulative, u)
            index = min(index, total)
            draws[:, j] = view(pooled, :, index)
        end
    end

    return PathfinderResult(
        model,
        args,
        constraints,
        best.location,
        best.covariance,
        best.cholesky_lower,
        draws,
        best.elbo,
        path_elbo_histories[best_path],
        num_paths,
        best_path,
        path_converged[best_path],
    )
end

# Seed the warmup INITIAL diagonal inverse-mass matrix from a Pathfinder fit
# (Zhang, Carpenter, Gelman, Vehtari, JMLR 2022): in HMC/NUTS the inverse-mass
# matrix is the position-covariance estimate, and Pathfinder already estimated a
# posterior covariance, so its diagonal starts warmup from a metric near the
# target. Only the diagonal is used -- the device metric is diagonal and the host
# default is diagonal; a dense seed is out of scope. Each entry is made safe the
# way `_inverse_mass_matrix` (src/inference/nuts/adaptation.jl) regularizes an
# adapted variance: clamp to at least `regularization`, and replace a non-finite
# diagonal entry with 1.0 (the unit-metric default).
function _pathfinder_inverse_mass_seed(pf::PathfinderResult, model::TeaModel, num_params::Int, regularization::Real)
    pf.model === model || throw(
        ArgumentError("the PathfinderResult passed as initial_params was fit on a different model"),
    )
    length(pf.location) == num_params || throw(
        ArgumentError(
            "the PathfinderResult passed as initial_params has dimension $(length(pf.location)), expected $num_params",
        ),
    )
    reg = Float64(regularization)
    seed = Vector{Float64}(undef, num_params)
    @inbounds for i = 1:num_params
        diag_i = pf.covariance[i, i]
        seed[i] = isfinite(diag_i) ? max(diag_i, reg) : 1.0
    end
    return seed
end

# Sampler initialization from a Pathfinder fit: chains start from Pathfinder
# draws (sampled with replacement, so multi-path pooling carries over).
function _initial_hmc_position(
    model::TeaModel,
    args::Tuple,
    constraints::ChoiceMap,
    initial_params::PathfinderResult,
    rng::AbstractRNG,
)
    initial_params.model === model || throw(
        ArgumentError("the PathfinderResult passed as initial_params was fit on a different model"),
    )
    return initial_params.draws[:, rand(rng, 1:size(initial_params.draws, 2))]
end

# Resolved-plan variant (the batched initial-position loop threads the resolved
# signature plan, issue #156): a Pathfinder init never draws from the prior, so
# the plan is unused and the position comes from the fit's draws unchanged.
function _initial_hmc_position(
    model::TeaModel,
    resolved::ResolvedSignaturePlan,
    args::Tuple,
    constraints::ChoiceMap,
    initial_params::PathfinderResult,
    rng::AbstractRNG,
)
    return _initial_hmc_position(model, args, constraints, initial_params, rng)
end
