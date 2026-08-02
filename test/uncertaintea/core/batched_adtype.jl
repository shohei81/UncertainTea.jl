# adtype (gradient-backend selection) argument handling on the batched path
# (issue #268, part A2). These run WITHOUT Enzyme loaded, so they cover the
# validation and the safe fall-through: without the extension, every adtype
# resolves to the forward-mode tiers, so behavior is unchanged.

using KernelAbstractions: CPU

@tea static function adtype_coupled()
    x ~ iid(normal(0.0, 1.0), 40)
    for i = 1:39
        {:y => i} ~ normal(tanh(x[i]) + 0.5 * x[i+1], 0.3)
    end
    return x
end

@testset "batched_adtype" begin
    adtype_cm = choicemap([(:y => i, 0.05 * i) for i = 1:39])
    adtype_params = reshape(collect(range(-1.0, 1.0; length=40 * 4)), 40, 4)

    @testset "adtype is validated" begin
        @test_throws ArgumentError UncertainTea.BatchedLogjointGradientCache(
            adtype_coupled, adtype_params, (), adtype_cm; adtype=:bogus,
        )
        @test_throws ArgumentError batched_nuts(
            adtype_coupled, (), adtype_cm; num_chains=2, num_samples=2, adtype=:bogus,
        )
    end

    @testset "reverse is host-only: rejected with a device backend" begin
        @test_throws ArgumentError batched_nuts(
            adtype_coupled, (), adtype_cm;
            num_chains=2, num_samples=2, tree_strategy=:masked, backend=CPU(), adtype=:reverse,
        )
    end

    @testset "without Enzyme every adtype falls back to forward mode" begin
        # the extension is not loaded here, so the reverse tier never engages and
        # the gradient is the ordinary forward-mode result.
        forward = copy(UncertainTea.batched_logjoint_gradient_unconstrained(
            adtype_coupled, adtype_params, (), adtype_cm,
        ))
        for adtype in (:auto, :forward, :reverse)
            cache = UncertainTea.BatchedLogjointGradientCache(
                adtype_coupled, adtype_params, (), adtype_cm; adtype=adtype,
            )
            @test cache.reverse_cache === nothing
            @test UncertainTea.batched_logjoint_gradient_unconstrained!(cache, adtype_params) ≈ forward rtol = 1e-10
        end
    end

    @testset "advi/svgd/smc accept and validate adtype (issue #275)" begin
        # a small model, no Enzyme -> every adtype falls back to forward and runs.
        @tea static function adtype_small()
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu, 1.0)
            return mu
        end
        small_cm = choicemap((:y, 0.5))

        for adtype in (:auto, :forward, :reverse)
            r = batched_advi(adtype_small, (), small_cm; num_particles=8, num_iterations=20, adtype=adtype, rng=MersenneTwister(1))
            @test all(isfinite, r.location)
            s = batched_svgd(adtype_small, (), small_cm; num_particles=8, num_iterations=15, adtype=adtype, rng=MersenneTwister(2))
            @test all(isfinite, s.constrained_particles)
        end

        @test_throws ArgumentError batched_advi(adtype_small, (), small_cm; num_particles=4, num_iterations=2, adtype=:bogus)
        @test_throws ArgumentError batched_svgd(adtype_small, (), small_cm; num_particles=4, num_iterations=2, adtype=:bogus)
        @test_throws ArgumentError batched_smc(adtype_small, (), small_cm; num_particles=4, adtype=:bogus)

        # reverse is host-only: rejected with a device backend
        @test_throws ArgumentError batched_advi(
            adtype_small, (), small_cm; num_particles=4, num_iterations=2, adtype=:reverse, backend=CPU(),
        )
        @test_throws ArgumentError batched_svgd(
            adtype_small, (), small_cm; num_particles=4, num_iterations=2, adtype=:reverse, backend=CPU(),
        )
    end
end
