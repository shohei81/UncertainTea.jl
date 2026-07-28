# Issue #137 device stranding gate.
#
# With SHARED step-size adaptation, prior-draw initialization strands the chains
# whose initial curvature the single shared step size never fits: on the gauss
# shape (mu ~ normal, s ~ gamma, many iid observations) the low-initial-s chains
# diverge (~5-7% aggregate divergence, a handful of fully-stranded chains). The
# fix is PER-CHAIN step-size adaptation on the device masked NUTS path, backed by a
# pooled/shared diagonal mass (the device leapfrog kernels consume one shared
# inverse-mass vector, so a per-chain mass is not device-representable, but the
# per-chain step is the operative fix). This test is the gate: at the device
# default (per-chain, prior-draw init, NO pinned init) the aggregate divergence
# collapses to well under 1% with no chain stranded, and the posterior recovers s.

using KernelAbstractions: CPU

# Local mean helper (Statistics is not imported by the test harness; keep this file
# standalone-runnable).
dnuts_stranding_mean(x) = sum(x) / length(x)

# Gauss shape: the device-lowerable form (mirrors bench_gauss / gpu_gauss_model).
@tea static function dnuts_stranding_gauss(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

@testset "dnuts_device_perchain_stranding" begin
    n = 400
    mu_true = 0.3
    s_true = 1.2
    data_rng = MersenneTwister(137)
    ys = [mu_true + s_true * randn(data_rng) for _ = 1:n]
    constraints = choicemap((:y => i, ys[i]) for i = 1:n)

    # Device default: per-chain step + pooled shared mass, prior-draw init (no
    # pinned/fixed initial_params), on the CPU() reference backend.
    chains = batched_nuts(
        dnuts_stranding_gauss,
        (n,),
        constraints;
        num_chains=64,
        num_samples=500,
        num_warmup=200,
        tree_strategy=:masked,
        backend=CPU(),
        rng=MersenneTwister(2024),
    )

    # The gate: aggregate divergence < 1% and no chain more than 50% divergent.
    @test divergencerate(chains) < 0.01
    per_chain_div = [divergencerate(chain) for chain in chains.chains]
    @test maximum(per_chain_div) < 0.5

    # Posterior recovers the scale s (second constrained parameter).
    names = parameter_names(chains)
    s_index = findfirst(==("s"), names)
    @test s_index !== nothing
    draws = posterior_array(chains)
    s_mean = dnuts_stranding_mean(vec(draws[:, :, s_index]))
    @test isapprox(s_mean, s_true; atol=0.25)

    # Sanity: draws finite and chains mixed.
    @test all(isfinite, draws)
    @test all(<(1.2), rhat(chains))
end
