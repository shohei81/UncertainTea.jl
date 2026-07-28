# Issue #158: the HOST batched default is now pooled mass + per-chain step, the
# SAME algorithm the device masked path already runs (a single diagonal mass
# pooled across all chains, one dual-averaged step per chain). This test gates that
# the two implementations agree: host masked NUTS (backend === nothing) vs the
# device masked NUTS on the CPU() reference backend, both at the per-chain default.
#
# The two paths use different proposal machinery (host hybrid/masked doubling vs
# device CPU() kernels) and host-side RNG is consumed in a different order, so the
# runs are statistically -- not bitwise -- equivalent; we assert posterior-mean and
# mass agreement within tolerance, matched divergence behavior, and mixing.

using KernelAbstractions: CPU

hdp_mean(x) = sum(x) / length(x)

# Gauss shape (device-lowerable): mu ~ normal, s ~ gamma, iid observations.
@tea static function hdp_gauss(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    return mu
end

@testset "host_device_pooled_agreement" begin
    n = 120
    mu_true = 0.3
    s_true = 1.2
    data_rng = MersenneTwister(158)
    ys = [mu_true + s_true * randn(data_rng) for _ = 1:n]
    constraints = choicemap((:y => i, ys[i]) for i = 1:n)

    run_kwargs = (
        num_chains=32,
        num_samples=400,
        num_warmup=200,
        tree_strategy=:masked,
    )

    # Host pooled path (backend === nothing): pooled mass + per-chain step.
    host = batched_nuts(
        hdp_gauss, (n,), constraints;
        run_kwargs...,
        backend=nothing,
        rng=MersenneTwister(9158),
    )
    # Device pooled path (CPU() reference backend): same algorithm.
    device = batched_nuts(
        hdp_gauss, (n,), constraints;
        run_kwargs...,
        backend=CPU(),
        rng=MersenneTwister(9158),
    )

    host_names = parameter_names(host)
    device_names = parameter_names(device)
    @test host_names == device_names
    s_index = findfirst(==("s"), host_names)
    mu_index = findfirst(==("mu"), host_names)
    @test s_index !== nothing && mu_index !== nothing

    host_draws = posterior_array(host)
    device_draws = posterior_array(device)

    host_s = hdp_mean(vec(host_draws[:, :, s_index]))
    device_s = hdp_mean(vec(device_draws[:, :, s_index]))
    host_mu = hdp_mean(vec(host_draws[:, :, mu_index]))
    device_mu = hdp_mean(vec(device_draws[:, :, mu_index]))

    # Both recover the truth, and agree with each other (two independent samplers
    # of the SAME pooled-mass / per-chain-step algorithm).
    @test isapprox(host_s, s_true; atol=0.2)
    @test isapprox(device_s, s_true; atol=0.2)
    @test isapprox(host_s, device_s; atol=0.15)
    @test isapprox(host_mu, device_mu; atol=0.15)

    # Both mix and neither strands chains.
    @test all(<(1.05), rhat(host))
    @test all(<(1.05), rhat(device))
    @test divergencerate(host) < 0.02
    @test divergencerate(device) < 0.02

    # Pooled mass is shared across chains on BOTH paths (all chains carry the same
    # diagonal mass), and the pooled mass entries agree between host and device.
    host_mass = host.chains[1].mass_matrix
    device_mass = device.chains[1].mass_matrix
    for chain in host.chains
        @test chain.mass_matrix == host_mass
    end
    for chain in device.chains
        @test chain.mass_matrix == device_mass
    end
    @test length(host_mass) == length(device_mass)
    for p = 1:length(host_mass)
        @test isapprox(host_mass[p], device_mass[p]; rtol=0.5)
    end
end
