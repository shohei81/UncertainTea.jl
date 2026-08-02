# Go/no-go benchmark for the mask-based iterative-doubling batched NUTS path.
#
# Times batched_nuts with tree_strategy=:hybrid vs :masked on CPU for
# (num_params, num_chains) in [(2, 8), (2, 64), (16, 64)] with a small budget
# (200 warmup + 200 samples), one warmup run per cell. The numbers inform the
# go/no-go decision for further NUTS-on-GPU investment; they gate nothing.
#
# Run from the repository root:
#   julia --project=. bench/nuts_masked_bench.jl

using UncertainTea
using UncertainTea.Inference
using Random

@tea static function nuts_bench_two_param()
    mu ~ normal(0.0, 1.0)
    log_sigma ~ normal(0.0, 0.5)
    {:y} ~ normal(mu, exp(log_sigma))
    return mu
end

@tea static function nuts_bench_sixteen_param()
    state ~ mvnormal(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75],
    )
    return state
end

bench_mean(x) = sum(x) / length(x)

function bench_run(model, constraints, num_chains, strategy, seed)
    return batched_nuts(
        model,
        (),
        constraints;
        num_chains=num_chains,
        num_samples=200,
        num_warmup=200,
        tree_strategy=strategy,
        rng=MersenneTwister(seed),
    )
end

# Full-width gradient-call estimate for the masked path: one call for the
# initial trajectory step plus at most 2^d - 2 leaf calls for a transition
# that reaches tree depth d (rounds 1..d-1 of 2^r leaves each). Every one of
# those calls evaluates all num_chains lanes -- the shape a device gradient
# consumes directly.
function bench_masked_gradient_calls(chains)
    depth_matrix = reduce(hcat, treedepths(chains))
    total = 0.0
    for sample_index in axes(depth_matrix, 1)
        max_depth = maximum(view(depth_matrix, sample_index, :))
        total += 1 + max(0, (1 << max_depth) - 2)
    end
    return total / size(depth_matrix, 1)
end

function main()
    configs = [
        ("2 params", nuts_bench_two_param, choicemap((:y, 0.7)), 2, 8),
        ("2 params", nuts_bench_two_param, choicemap((:y, 0.7)), 2, 64),
        ("16 params", nuts_bench_sixteen_param, choicemap(), 16, 64),
    ]

    println("batched_nuts tree strategy benchmark (CPU, 200 warmup + 200 samples)")
    println()
    header =
        rpad("config", 24) * rpad("chains", 8) *
        rpad("hybrid [s]", 12) * rpad("masked [s]", 12) *
        rpad("masked/hybrid", 15) * "masked grad calls/iter (est)"
    println(header)
    println(repeat("-", length(header)))

    for (label, model, constraints, num_params, num_chains) in configs
        bench_run(model, constraints, num_chains, :hybrid, 1)
        hybrid_time = @elapsed bench_run(model, constraints, num_chains, :hybrid, 2)
        bench_run(model, constraints, num_chains, :masked, 1)
        masked_time = @elapsed masked_chains = bench_run(model, constraints, num_chains, :masked, 2)
        gradient_calls = bench_masked_gradient_calls(masked_chains)
        println(
            rpad("($num_params, $num_chains) $label", 24) *
            rpad(string(num_chains), 8) *
            rpad(string(round(hybrid_time; digits=3)), 12) *
            rpad(string(round(masked_time; digits=3)), 12) *
            rpad(string(round(masked_time / hybrid_time; digits=2)), 15) *
            string(round(gradient_calls; digits=1)),
        )
    end

    println()
    println("masked grad calls/iter: estimated full-width batched gradient calls per")
    println("sampling iteration (each call covers all chains; the device-gradient")
    println("workload if the masked inner loop were moved on-device).")
end

main()
