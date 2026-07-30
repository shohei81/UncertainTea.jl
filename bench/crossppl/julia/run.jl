# UncertainTea runner for the cross-PPL benchmark (issue #121).
#
# Usage (from bench/crossppl):
#   julia --project=julia julia/run.jl --model eight_schools_noncentered \
#       --variant cpu --chains 4 --samples 1000 --warmup 1000 --seed 1 --reps 3
#
# Variants:
#   cpu            nuts_chains (sequential chains, per-chain adaptation)
#   batched-cpu    batched_nuts tree_strategy=:hybrid (vectorized chains, CPU,
#                  single-threaded host path)
#   batched-cpu-ka batched_nuts tree_strategy=:masked on the
#                  KernelAbstractions CPU() backend — parallelizes across
#                  Julia threads, so launch with `julia -t auto`
#   batched-metal  batched_nuts tree_strategy=:masked, Metal backend, Float32
#   chees          batched_chees (ChEES-HMC, issue #161): fixed-length jittered
#                  HMC with cross-chain trajectory-length adaptation. Reported as
#                  its own `uncertaintea-chees` label, not replacing NUTS.
#
# Emits, per repetition:
#   results/raw/<model>__uncertaintea-<variant>__chains<K>__seed<S>__rep<R>.npz
#     "draws": Float64 (chains, draws, params)
#   ... plus a .json sidecar with parameter names, timings and settings.
#
# Timing protocol: one cold full run first (JIT compile; reported as ttfx_s),
# then per warm repetition two runs from an identical RNG seed — (warmup, 1
# draw) and (warmup, S draws).  The warmup trajectories are identical, so the
# difference isolates pure sampling time (sampling_s); draws come from the
# second run.

using UncertainTea
using Random
using JSON3
using NPZ

# Metal is macOS-only; load it lazily so the CPU variants run inside the
# Linux benchmark container.
if any(i -> ARGS[i] == "--variant" && ARGS[i+1] == "batched-metal", 1:(length(ARGS)-1))
    using Metal
end
if any(i -> ARGS[i] == "--variant" && ARGS[i+1] == "batched-cpu-ka", 1:(length(ARGS)-1))
    using KernelAbstractions
end

include(joinpath(@__DIR__, "models.jl"))

function parse_args(argv)
    opts = Dict{String,String}(
        "variant" => "cpu",
        "chains" => "4",
        "samples" => "1000",
        "warmup" => "1000",
        "seed" => "1",
        "reps" => "3",
        "out" => joinpath(@__DIR__, "..", "results", "raw"),
        "target-accept" => "0.8",
        "max-tree-depth" => "10",
        # ChEES-HMC options (variant chees, issue #161). `num-leapfrog-steps` seeds
        # the initial trajectory length T = L*eps; ChEES then adapts T from the
        # cross-chain ensemble during warmup. `jitter` is the Halton jitter fraction.
        "num-leapfrog-steps" => "10",
        "jitter" => "1.0",
    )
    i = 1
    while i <= length(argv)
        startswith(argv[i], "--") || error("unexpected argument $(argv[i])")
        key = argv[i][3:end]
        opts[key] = argv[i+1]
        i += 2
    end
    return opts
end

function load_json(name)
    return JSON3.read(read(joinpath(@__DIR__, "..", "data", name), String))
end

# Returns (model, args, constraints, canonical parameter names).
function setup_model(model_name)
    if startswith(model_name, "eight_schools")
        data = load_json("eight_schools.json")
        sigma = Float64.(data.sigma)
        cons = choicemap(((:y => i, Float64(data.y[i])) for i = 1:data.J)...)
        model = model_name == "eight_schools_centered" ? bench_eight_schools_centered :
                bench_eight_schools_noncentered
        return (model, (sigma,), cons)
    elseif model_name == "logistic"
        data = load_json("logistic.json")
        n, d = data.n, data.d
        X = reduce(hcat, [Float64.(col) for col in data.X])  # d x n
        @assert size(X) == (d, n)
        cons = choicemap(((:y => i, Float64(data.y[i])) for i = 1:n)...)
        return (bench_logistic, (X, n), cons)
    elseif model_name == "logistic_large"
        data = load_json("logistic_large.json")
        n, d = data.n, data.d
        X = reduce(hcat, [Float64.(col) for col in data.X])  # d x n
        @assert size(X) == (d, n)
        cons = choicemap(((:y => i, Float64(data.y[i])) for i = 1:n)...)
        return (bench_logistic_large, (X, n), cons)
    elseif model_name == "gauss"
        data = load_json("gauss.json")
        n = data.n
        cons = choicemap(((:y => i, Float64(data.y[i])) for i = 1:n)...)
        return (bench_gauss, (n,), cons)
    else
        error("unknown model $(model_name)")
    end
end

function run_once(model, args, cons, opts; num_samples, seed)
    variant = opts["variant"]
    num_chains = parse(Int, opts["chains"])
    num_warmup = parse(Int, opts["warmup"])
    target_accept = parse(Float64, opts["target-accept"])
    max_tree_depth = parse(Int, opts["max-tree-depth"])
    # --init "v1,v2,...": pinned unconstrained initial position for every
    # chain — a diagnostic workaround for issue #137 (prior-draw init strands
    # chains under shared adaptation).  Runs carry a "-pinned-init" label so
    # they are never mistaken for default-configuration results.
    initial_params = haskey(opts, "init") ?
                     [parse(Float64, x) for x in split(opts["init"], ",")] : nothing
    rng = MersenneTwister(seed)
    if variant == "cpu"
        t = @elapsed chains = nuts_chains(model, args, cons;
            num_chains, num_samples, num_warmup, target_accept, max_tree_depth,
            initial_params, rng)
        return (chains, t)
    elseif variant == "batched-cpu"
        t = @elapsed chains = batched_nuts(model, args, cons;
            num_chains, num_samples, num_warmup, target_accept, max_tree_depth,
            tree_strategy=:hybrid, initial_params, rng)
        return (chains, t)
    elseif variant == "batched-cpu-ka"
        backend = KernelAbstractions.CPU()
        t = @elapsed chains = batched_nuts(model, args, cons;
            num_chains, num_samples, num_warmup, target_accept, max_tree_depth,
            tree_strategy=:masked, backend, precision=Float64, initial_params, rng)
        return (chains, t)
    elseif variant == "batched-metal"
        backend = Metal.MetalBackend()
        t = @elapsed chains = batched_nuts(model, args, cons;
            num_chains, num_samples, num_warmup, target_accept, max_tree_depth,
            tree_strategy=:masked, backend, precision=Float32, initial_params, rng)
        return (chains, t)
    elseif variant == "chees"
        # ChEES-HMC (issue #161): fixed-length jittered HMC whose trajectory length
        # is adapted from the cross-chain ensemble during warmup. No tree, so no
        # max_tree_depth; `num-leapfrog-steps` seeds the initial trajectory length.
        num_leapfrog_steps = parse(Int, opts["num-leapfrog-steps"])
        jitter_amount = parse(Float64, opts["jitter"])
        t = @elapsed chains = batched_chees(model, args, cons;
            num_chains, num_samples, num_warmup, target_accept,
            num_leapfrog_steps, jitter_amount, initial_params, rng)
        return (chains, t)
    else
        error("unknown variant $(variant)")
    end
end

function main(argv)
    opts = parse_args(argv)
    model_name = opts["model"]
    variant = opts["variant"]
    (model, args, cons) = setup_model(model_name)
    outdir = abspath(opts["out"])
    mkpath(outdir)
    num_chains = parse(Int, opts["chains"])
    num_samples = parse(Int, opts["samples"])
    base_seed = parse(Int, opts["seed"])
    reps = parse(Int, opts["reps"])

    # Cold run: includes JIT compilation for this (model, variant, shape).
    (_, ttfx_s) = run_once(model, args, cons, opts; num_samples, seed=base_seed)
    println("cold run (ttfx): ", round(ttfx_s; digits=2), " s")

    for rep = 1:reps
        seed = base_seed + rep
        (_, t_warmup_only) = run_once(model, args, cons, opts; num_samples=1, seed)
        (chains, t_total) = run_once(model, args, cons, opts; num_samples, seed)
        sampling_s = max(t_total - t_warmup_only, 0.0)
        draws = permutedims(posterior_array(chains), (2, 1, 3))  # (chains, draws, params)
        names = parameter_names(chains)
        variant_label = haskey(opts, "init") ? string(variant, "-pinned-init") : variant
        stem = string(model_name, "__uncertaintea-", variant_label,
            "__chains", num_chains, "__seed", seed, "__rep", rep)
        npzwrite(joinpath(outdir, stem * ".npz"), Dict("draws" => draws))
        meta = Dict(
            "model" => model_name,
            "framework" => "uncertaintea",
            "variant" => variant_label,
            "params" => names,
            "num_chains" => num_chains,
            "num_samples" => num_samples,
            "num_warmup" => parse(Int, opts["warmup"]),
            "seed" => seed,
            "rep" => rep,
            "timings" => Dict(
                "ttfx_s" => ttfx_s,
                "warmup_s" => t_warmup_only,
                "total_s" => t_total,
                "sampling_s" => sampling_s,
            ),
            "sampler" =>
                variant == "chees" ?
                Dict(
                    "kind" => "chees",
                    "init" => haskey(opts, "init") ? opts["init"] : "prior-draw",
                    "target_accept" => parse(Float64, opts["target-accept"]),
                    "num_leapfrog_steps" => parse(Int, opts["num-leapfrog-steps"]),
                    "jitter" => parse(Float64, opts["jitter"]),
                    "adapt_trajectory_length" => true,
                    "metric" => "diag",
                    "chain_parallelism" => "vectorized",
                    "precision" => "Float64",
                ) :
                Dict(
                    "kind" => "nuts",
                    "init" => haskey(opts, "init") ? opts["init"] : "prior-draw",
                    "target_accept" => parse(Float64, opts["target-accept"]),
                    "max_tree_depth" => parse(Int, opts["max-tree-depth"]),
                    "metric" => "diag",
                    "chain_parallelism" =>
                        variant == "cpu" ? "sequential" :
                        variant == "batched-cpu-ka" ?
                        "vectorized-threads-$(Threads.nthreads())" : "vectorized",
                    "precision" => variant == "batched-metal" ? "Float32" : "Float64",
                ),
            "diagnostics" => Dict("divergence_rate" => divergencerate(chains)),
            "env" => Dict(
                "julia" => string(VERSION),
                "uncertaintea_commit" =>
                    strip(read(`git -C $(joinpath(@__DIR__, "..", "..", "..")) rev-parse --short HEAD`, String)),
            ),
        )
        open(joinpath(outdir, stem * ".json"), "w") do io
            JSON3.pretty(io, meta)
        end
        println(stem, "  sampling=", round(sampling_s; digits=3), "s",
            "  warmup=", round(t_warmup_only; digits=3), "s",
            "  div=", round(divergencerate(chains); digits=4))
    end
end

main(ARGS)
