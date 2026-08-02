struct MAPResult
    unconstrained_mode::Vector{Float64}
    constrained_mode::Vector{Float64}
    logjoint::Float64
    converged::Bool
    iterations::Int
    gradient_norm::Float64
end

function Base.show(io::IO, result::MAPResult)
    print(
        io,
        "MAPResult(dim=$(length(result.unconstrained_mode)), logjoint=$(result.logjoint), ",
        "converged=$(result.converged), iterations=$(result.iterations), ",
        "gradient_norm=$(result.gradient_norm))",
    )
end

struct LaplaceResult
    map::MAPResult
    covariance::Matrix{Float64}
    cholesky_lower::Matrix{Float64}
end

function Base.show(io::IO, result::LaplaceResult)
    print(
        io,
        "LaplaceResult(dim=$(length(result.map.unconstrained_mode)), ",
        "logjoint=$(result.map.logjoint), converged=$(result.map.converged))",
    )
end

# Two-loop recursion: overwrite `out` with H*v, where H is the L-BFGS
# inverse-Hessian estimate defined by the (s, y, rho) history and the standard
# gamma = s'y / y'y initial scaling. Shared by the optimizer's direction
# computation and Pathfinder's per-iterate Gaussian approximations.
function _lbfgs_apply_inverse_hessian!(
    out::AbstractVector,
    v::AbstractVector,
    s_history::Vector{Vector{Float64}},
    y_history::Vector{Vector{Float64}},
    rho_history::Vector{Float64},
    alpha_cache::Vector{Float64}=Vector{Float64}(undef, length(s_history)),
)
    copyto!(out, v)
    m = length(s_history)
    resize!(alpha_cache, m)
    @inbounds for i = m:-1:1
        ai = rho_history[i] * dot(s_history[i], out)
        alpha_cache[i] = ai
        axpy!(-ai, y_history[i], out)
    end

    gamma = 1.0
    if m > 0
        sy = dot(s_history[end], y_history[end])
        yy = dot(y_history[end], y_history[end])
        if yy > 0
            gamma = sy / yy
        end
    end
    out .*= gamma

    @inbounds for i = 1:m
        bi = rho_history[i] * dot(y_history[i], out)
        axpy!(alpha_cache[i] - bi, s_history[i], out)
    end
    return out
end

# Self-contained L-BFGS maximizer of an objective `f` with gradient supplied by
# `gradient!(g, x)`. Uses two-loop recursion with limited history and a
# backtracking Armijo line search. Returns (x, fx, g, gnorm, iterations, converged).
# `iteration_callback(x, fx, g, s_history, y_history, rho_history)` fires after
# every accepted step with the histories BORROWED (the callee must copy).
function _lbfgs_maximize(
    f,
    gradient!,
    x0::AbstractVector;
    history::Int=10,
    max_iters::Int=500,
    g_tol::Float64=1e-8,
    iteration_callback=nothing,
)
    x = collect(float.(x0))
    n = length(x)
    g = similar(x)
    gradient!(g, x)
    fx = f(x)

    gnorm = norm(g, Inf)
    gnorm <= g_tol && return (x, fx, g, gnorm, 0, true)

    s_history = Vector{Vector{Float64}}()
    y_history = Vector{Vector{Float64}}()
    rho_history = Vector{Float64}()

    q = similar(x)
    alpha_cache = Vector{Float64}()
    converged = false
    iteration = 0

    c1 = 1e-4
    max_backtracks = 40

    while iteration < max_iters
        iteration += 1

        # Two-loop recursion to compute the search direction (ascent on f, so we
        # work with the maximization gradient directly).
        _lbfgs_apply_inverse_hessian!(q, g, s_history, y_history, rho_history, alpha_cache)
        direction = q  # ascent direction (approx H * g)

        gd = dot(g, direction)
        # Guard against a non-ascent direction (e.g. numerical issues); fall back
        # to steepest ascent.
        if !(gd > 0)
            copyto!(direction, g)
            gd = dot(g, direction)
            if !(gd > 0)
                converged = false
                break
            end
        end

        # Backtracking Armijo line search (maximization: seek sufficient increase).
        step = 1.0
        x_new = similar(x)
        f_new = fx
        line_ok = false
        for _ = 0:max_backtracks
            @. x_new = x + step * direction
            f_new = f(x_new)
            if isfinite(f_new) && f_new >= fx + c1 * step * gd
                line_ok = true
                break
            end
            step *= 0.5
        end

        if !line_ok
            converged = false
            break
        end

        g_new = similar(x)
        gradient!(g_new, x_new)

        s = x_new .- x
        y = g .- g_new  # gradient decrease matches minimization convention for -f
        sy = dot(s, y)
        if sy > 1e-12
            push!(s_history, s)
            push!(y_history, y)
            push!(rho_history, 1.0 / sy)
            if length(s_history) > history
                popfirst!(s_history)
                popfirst!(y_history)
                popfirst!(rho_history)
            end
        end

        x = x_new
        g = g_new
        fx = f_new
        gnorm = norm(g, Inf)

        isnothing(iteration_callback) ||
            iteration_callback(x, fx, g, s_history, y_history, rho_history)

        if gnorm <= g_tol
            converged = true
            break
        end
    end

    return (x, fx, g, gnorm, iteration, converged)
end

"""
    map_estimate(model, args=(), constraints=choicemap(); init=nothing, max_iters=500, g_tol=1e-8, history=10, rng) -> MAPResult

Maximum a posteriori estimate: maximize the unconstrained-space log joint with
L-BFGS (memory `history`, at most `max_iters` iterations, gradient-norm
tolerance `g_tol`). `init` seeds the optimizer (a random draw when `nothing`).
Returns a `MAPResult` carrying the unconstrained and constrained modes, the
objective value, and convergence information.
"""
function map_estimate(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    init=nothing,
    max_iters::Int=500,
    g_tol::Float64=1e-8,
    history::Int=10,
    rng::AbstractRNG=Random.default_rng(),
)
    start = _resolve_unconstrained_point(model, args, constraints, init, rng, "init")
    seed = collect(float.(start))

    objective = theta -> logjoint_unconstrained(model, theta, args, constraints)
    cache = _logjoint_gradient_cache(model, seed, args, constraints)
    gradient! = (g, x) -> copyto!(g, _logjoint_gradient!(cache, x))

    mode, fx, _, gnorm, iterations, converged =
        _lbfgs_maximize(objective, gradient!, seed; history=history, max_iters=max_iters, g_tol=g_tol)

    constrained = transform_to_constrained(model, mode, args, constraints)
    return MAPResult(mode, constrained, fx, converged, iterations, gnorm)
end

"""
    laplace_approximation(model, args=(), constraints=choicemap(); kwargs...) -> LaplaceResult

Laplace (Gaussian) approximation of the posterior in unconstrained space: find
the mode with `map_estimate` (keyword arguments are forwarded), then set the
covariance to the inverse of the negative Hessian at the mode. Returns a
`LaplaceResult`; draw samples from it with `rand(rng, result, n)`.
"""
function laplace_approximation(
    model::TeaModel,
    args::Tuple=(),
    constraints::ChoiceMap=choicemap();
    kwargs...,
)
    map_result = map_estimate(model, args, constraints; kwargs...)
    mode = map_result.unconstrained_mode

    objective = theta -> logjoint_unconstrained(model, theta, args, constraints)
    H = ForwardDiff.hessian(objective, mode)
    negH = Symmetric(-H)

    n = length(mode)
    factor = nothing
    jitter = 0.0
    let scale = 1e-8
        while true
            candidate = jitter == 0.0 ? negH : Symmetric(-H + jitter * I(n))
            attempt = cholesky(candidate; check=false)
            if issuccess(attempt)
                factor = attempt
                break
            end
            if jitter == 0.0
                jitter = scale
            else
                jitter *= 10
            end
            if jitter > 1e-2
                throw(
                    ErrorException(
                        "laplace_approximation: negative Hessian is not positive definite at the mode; " *
                        "the objective may be unbounded or the mode may not be a local maximum",
                    ),
                )
            end
        end
    end

    covariance = Matrix(Symmetric(inv(factor)))
    # Lower Cholesky factor of the covariance so that draws mode .+ L*z with
    # z ~ N(0, I) have covariance = inv(-H).
    L = Matrix(cholesky(Symmetric(covariance)).L)
    return LaplaceResult(map_result, covariance, L)
end

function Base.rand(rng::AbstractRNG, result::LaplaceResult, n::Int)
    mode = result.map.unconstrained_mode
    L = result.cholesky_lower
    d = length(mode)
    samples = Matrix{Float64}(undef, d, n)
    z = Vector{Float64}(undef, d)
    for j = 1:n
        randn!(rng, z)
        @views samples[:, j] .= mode .+ L * z
    end
    return samples
end
