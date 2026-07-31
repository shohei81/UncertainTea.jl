# Regenerates the synthetic shared datasets (logistic.json, linreg.json).
# The JSON files are checked in; this script only documents their provenance.
# Run from bench/crossppl: julia --project=julia generate_data.jl
using Random
using JSON3

const SEED = 20260723

function write_json(path, obj)
    open(path, "w") do io
        JSON3.pretty(io, obj)
        println(io)
    end
    println("wrote ", path)
end

function logistic_data(rng)
    n, d = 500, 8
    X = randn(rng, d, n)
    alpha = 0.3
    beta = [0.8, -1.2, 0.5, 0.0, 1.5, -0.4, 0.9, -0.7]
    eta = alpha .+ X' * beta
    y = Int.(rand(rng, n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
    return Dict(
        "model" => "logistic",
        "seed" => SEED,
        "n" => n,
        "d" => d,
        "true_alpha" => alpha,
        "true_beta" => beta,
        "X" => [X[:, i] for i = 1:n],
        "y" => y,
    )
end

# Large logistic regression (D=16 coefficients, N=8000 observations): the same
# GLM shape as `logistic_data`, scaled up so the per-observation D-dim dot
# product over N observations makes the gradient genuinely heavy — the model
# that exercises the device analytic path and many-chains scaling (issue #121).
# D is 16, not the audit's nominal 64, because the device analytic path caps
# the coefficient dimension at 16 (DEVICE_MAX_VECTOR_DIMENSION, a kernel
# compile-time budget) and this model must lower to BOTH the host and device
# paths; N is raised to 8000 so D*N (= 128000) matches the intended
# D=64 * N=2000 per-gradient work budget.  Coefficients are kept moderate
# (|beta| ~ 0.25) with independent standard normal covariates and N >> D, so
# the posterior is well-conditioned (no separation) and every framework clears
# the correctness gate.
function logistic_large_data(rng)
    n, d = 8000, 16
    X = randn(rng, d, n)
    alpha = 0.5
    beta = 0.25 .* randn(rng, d)
    eta = alpha .+ X' * beta
    y = Int.(rand(rng, n) .< 1.0 ./ (1.0 .+ exp.(-eta)))
    return Dict(
        "model" => "logistic_large",
        "seed" => SEED,
        "n" => n,
        "d" => d,
        "true_alpha" => alpha,
        "true_beta" => beta,
        "X" => [X[:, i] for i = 1:n],
        "y" => y,
    )
end

function gauss_data(rng)
    n = 1000
    mu, s = 0.5, 1.2
    ys = mu .+ s .* randn(rng, n)
    return Dict(
        "model" => "gauss",
        "seed" => SEED,
        "n" => n,
        "true_mu" => mu,
        "true_s" => s,
        "y" => ys,
    )
end

# issue #224: 2-component Gaussian mixture with ORDERED means (mu1 < mu2) drawn
# from a fixed-weight (0.4/0.6) mixture, shared scale. The identifiable
# parameterization (mu2 = mu1 + exp(log_gap)) matches the Julia/Stan/NumPyro
# models exactly so the correctness gate compares well-defined posteriors.
function mixture_data(rng)
    n = 500
    mu1, mu2, s, w1 = -1.5, 2.0, 0.7, 0.4
    ys = Vector{Float64}(undef, n)
    for i = 1:n
        ys[i] = (rand(rng) < w1 ? mu1 : mu2) + s * randn(rng)
    end
    return Dict(
        "model" => "mixture",
        "seed" => SEED,
        "n" => n,
        "true_mu1" => mu1,
        "true_mu2" => mu2,
        "true_s" => s,
        "true_w1" => w1,
        "y" => ys,
    )
end

rng = MersenneTwister(SEED)
dir = joinpath(@__DIR__, "data")
write_json(joinpath(dir, "logistic.json"), logistic_data(rng))
write_json(joinpath(dir, "gauss.json"), gauss_data(rng))
# Independent RNG so adding these datasets leaves logistic.json / gauss.json
# byte-identical (they share `rng`, whose draw sequence must not shift).
write_json(joinpath(dir, "logistic_large.json"), logistic_large_data(MersenneTwister(SEED + 1)))
write_json(joinpath(dir, "mixture.json"), mixture_data(MersenneTwister(SEED + 2)))
