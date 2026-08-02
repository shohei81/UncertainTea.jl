# Reverse-mode AD prototype for issue #268 (follow-up to the RFC #263).
#
# The RFC measured a QUADRATIC forward-mode (ForwardDiff) gradient ceiling on
# non-analytic high-`P` models — exactly the models the analytic/fused path
# (GLM, iid suffstats) does not cover. This script prototypes the recommended
# direction (Option 1: host reverse-mode AD via Enzyme.jl) and quantifies the
# win, keeping Enzyme in this ISOLATED benchmark environment so it never enters
# the core UncertainTea dependency graph (the "clean boundary" the RFC asked for).
#
# It reports three things:
#   (1) reverse-mode vs forward-mode CORRECTNESS + SCALING on a pure high-`P`
#       logjoint mirroring the RFC's coupled model — reverse-mode should match to
#       ~1e-15 and flatten the quadratic curve to linear.
#   (2) the GP hyperparameter gradient (the RFC's stated first target): a pure
#       `UncertainTea.logpdf(gp, y)` through the dense Cholesky.
#   (3) the INTEGRATION BLOCKER: Enzyme cannot yet differentiate UncertainTea's
#       real batched per-column objective, because the compiled-plan/workspace
#       evaluator is type-unstable (Any-typed scratch) and mutates captured
#       state. This is the concrete architectural finding that scopes the real
#       implementation — a type-stable, non-mutating per-column evaluator path.
#
# Setup + run (from the repository root):
#   julia --project=bench/reverse_mode -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=bench/reverse_mode bench/reverse_mode/enzyme_scaling.jl

using UncertainTea, ForwardDiff, Enzyme, Random, LinearAlgebra, BenchmarkTools, Printf
using UncertainTea.Inference
const UT = UncertainTea

# --- (1) pure high-P coupled logjoint --------------------------------------
# normal(0,1) prior on each x[i] + a per-element nonlinear neighbour coupling
# y[i] ~ normal(tanh(x[i]) + 0.5 x[i+1], 0.3). Nothing sufficient-statistics
# fuses, so this is the class the analytic path does NOT cover. Fully type
# stable and non-mutating — the shape reverse-mode AD wants.
function coupled_logjoint(x, y)
    T = eltype(x)
    c = T(0.9189385332046727)                     # 0.5 log(2pi)
    lp = zero(T)
    @inbounds for i in eachindex(x)
        lp += -x[i]^2 / 2 - c                      # normal(0,1) prior
    end
    @inbounds for i = 1:(length(x)-1)
        z = (y[i] - (tanh(x[i]) + T(0.5) * x[i+1])) / T(0.3)
        lp += -z^2 / 2 - log(T(0.3)) - c           # normal(mu, 0.3) obs
    end
    return lp
end

function run_scaling()
    println("== (1) pure high-P coupled logjoint: Enzyme reverse vs ForwardDiff ==")
    println("P     FD (ms)    Enz-rev (ms)   speedup   max|Δ|")
    for P in (10, 25, 50, 100, 200, 400, 800)
        rng = MersenneTwister(P)
        x = randn(rng, P)
        y = [tanh(x[i]) + 0.5 * x[i+1] + 0.3 * randn(rng) for i = 1:(P-1)]
        fdg(u) = ForwardDiff.gradient(v -> coupled_logjoint(v, y), u)
        eng(u) = Enzyme.gradient(Reverse, coupled_logjoint, u, Const(y))[1]
        d = maximum(abs.(fdg(x) .- eng(x)))
        t_fd = @belapsed $fdg($x)
        t_en = @belapsed $eng($x)
        @printf("%-5d %8.4f   %10.4f   %6.1fx   %.1e\n", P, 1e3 * t_fd, 1e3 * t_en, t_fd / t_en, d)
    end
end

# --- (2) GP hyperparameter gradient (RFC first target) ----------------------
function run_gp()
    println("\n== (2) GP hyperparameter gradient: Enzyme reverse vs ForwardDiff ==")
    rng = MersenneTwister(262)
    n = 30
    X = reshape(sort(rand(rng, n) .* 5), 1, n)
    Ktrue = exp.(-0.5 .* [(X[1, i] - X[1, j])^2 for i = 1:n, j = 1:n]) + 0.04 .* Matrix(I, n, n)
    ygp = cholesky(Symmetric(Ktrue)).L * randn(rng, n)
    gp_obj(h) = UT.logpdf(UT.gaussianprocess(X, exp(h[1]), exp(h[2]), exp(h[3])), ygp)
    h0 = [0.0, 0.0, -1.0]
    fd = ForwardDiff.gradient(gp_obj, h0)
    # gp_obj closes over the constant inputs/observations (no derivative data), so
    # it is passed as Const — otherwise Enzyme cannot prove the closure readonly.
    en = only(Enzyme.gradient(set_runtime_activity(Reverse), Const(gp_obj), h0))
    @printf("FD  = %s\nEnz = %s\nmax|Δ| = %.2e  %s\n", string(round.(fd; digits=6)),
        string(round.(en; digits=6)), maximum(abs.(fd .- en)),
        maximum(abs.(fd .- en)) < 1e-6 ? "MATCH" : "MISMATCH")
end

# --- (3) integration blocker: the real batched objective is not yet
#         Enzyme-differentiable ---------------------------------------------
@tea static function _blocker_model()
    x ~ iid(normal(0.0, 1.0), 8)
    for i = 1:7
        {:y => i} ~ normal(tanh(x[i]) + 0.5 * x[i+1], 0.3)
    end
    return x
end

function run_blocker()
    println("\n== (3) integration blocker: Enzyme on UncertainTea's real objective ==")
    rng = MersenneTwister(1)
    xt = randn(rng, 8)
    cm = UT.choicemap([(:y => i, tanh(xt[i]) + 0.5 * xt[i+1] + 0.3 * randn(rng)) for i = 1:7])
    params = reshape(randn(rng, 8), 8, 1)
    cache = UT.BatchedLogjointGradientCache(_blocker_model, params, (), cm)
    obj = cache.column_caches[1].objective
    theta = vec(copy(params))
    fd = ForwardDiff.gradient(obj, theta)
    @printf("ForwardDiff on the real objective: OK (grad[1:3] = %s)\n", string(round.(fd[1:3]; digits=4)))
    try
        Enzyme.gradient(set_runtime_activity(Reverse), Const(obj), theta)
        println("Enzyme on the real objective: UNEXPECTEDLY SUCCEEDED")
    catch e
        msg = sprint(showerror, e)
        println("Enzyme on the real objective: BLOCKED — ", first(msg, 140))
        println("  -> the compiled-plan/workspace evaluator is not Enzyme-differentiable;")
        println("     a type-stable, non-mutating per-column evaluator is the next step.")
    end
end

run_scaling()
run_gp()
run_blocker()
