# issue #227: observation-parallel tiled gradient with an observation-consuming
# step AFTER the tiled loop.
#
# The tiled path (issue #153, PR #195) splits the plan into a prelude (everything
# except the loop) + a per-tile body kernel. The observation cursor is threaded
# sequentially through the plan walk, so a post-loop observed choice must resume at
# the cursor advanced past the loop's ENTIRE consumption (`base_cursor + n_obs*stride`)
# -- otherwise it reads the loop's FIRST observation row instead of its own, a silent
# wrong logjoint/gradient that only engages at `n_obs >= DEVICE_GRADIENT_TILE_MIN_OBS`.
# The fix splits the prelude into pre-loop and post-loop step tuples and launches the
# post-loop walk from that advanced cursor. These parity checks pin device-vs-host
# agreement for a trailing-observation model both above and below the tile threshold;
# a masked-NUTS recovery confirms the sampler now targets the correct posterior.

using KernelAbstractions: CPU

# Local mean helper (Statistics is not imported by the test harness).
tpo_mean(x) = sum(x) / length(x)

# A large iid observation loop followed by a trailing observed scalar. `z` is a
# separate observation row AFTER the loop, so a correct walk must skip the loop's
# `n_obs` rows before scoring it.
@tea static function tpo_trailing_model(n)
    mu ~ normal(0.0, 1.0)
    s ~ gamma(2.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, s)
    end
    z ~ normal(mu, s)
    return mu
end

# Post-loop observation whose row lies several strides past the loop end (the loop
# body itself consumes one row/iteration): a trailing SMALL dataset after the loop.
@tea static function tpo_trailing_pair_model(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0)
    end
    {:z => 1} ~ normal(mu, 1.0)
    {:z => 2} ~ normal(mu, 1.0)
    return mu
end

@testset "tpo_parity_above_and_below_threshold" begin
    tile_min = UncertainTea.DEVICE_GRADIENT_TILE_MIN_OBS

    # Above the threshold: tiling engages AND a post-loop observation follows, so
    # this is exactly the issue #227 shape. The distinctive trailing z (far from the
    # loop data) makes a misread of the loop's first row show up as a large error.
    for n in (tile_min, 1000)
        rng = MersenneTwister(227)
        ys = randn(rng, n)
        params = randn(rng, 2, 3)
        cm = choicemap(((:y => i, ys[i]) for i = 1:n)..., (:z, 100.0))
        ws = DeviceBatchedWorkspace(tpo_trailing_model, 3; precision=Float64, args=(n,), constraints=cm)
        @test !isnothing(ws.tiled_gradient)        # tiled path active
        @test ws.tiled_gradient.n_obs == n
        v, g = device_batched_logjoint_gradient!(ws, params)
        vref = batched_logjoint_unconstrained(tpo_trailing_model, params, (n,), cm)
        gref = batched_logjoint_gradient_unconstrained(tpo_trailing_model, params, (n,), cm)
        @test v ≈ vref rtol = 1e-10
        @test g ≈ gref rtol = 1e-10
    end

    # Just below the threshold: the serial scan is kept (no tiled descriptor); the
    # identical model must still agree with the host (the bitwise-identical path).
    let n = tile_min - 1
        rng = MersenneTwister(228)
        ys = randn(rng, n)
        params = randn(rng, 2, 3)
        cm = choicemap(((:y => i, ys[i]) for i = 1:n)..., (:z, 100.0))
        ws = DeviceBatchedWorkspace(tpo_trailing_model, 3; precision=Float64, args=(n,), constraints=cm)
        @test isnothing(ws.tiled_gradient)
        v, g = device_batched_logjoint_gradient!(ws, params)
        @test v ≈ batched_logjoint_unconstrained(tpo_trailing_model, params, (n,), cm) rtol = 1e-10
        @test g ≈ batched_logjoint_gradient_unconstrained(tpo_trailing_model, params, (n,), cm) rtol = 1e-10
    end

    # Two trailing observation rows after the loop: the post-loop cursor must land
    # exactly at `base_cursor + n_obs*stride + 1`, then advance normally.
    let n = 512
        rng = MersenneTwister(229)
        ys = randn(rng, n)
        params = randn(rng, 1, 3)
        cm = choicemap(((:y => i, ys[i]) for i = 1:n)..., (:z => 1, 50.0), (:z => 2, -50.0))
        ws = DeviceBatchedWorkspace(tpo_trailing_pair_model, 3; precision=Float64, args=(n,), constraints=cm)
        @test !isnothing(ws.tiled_gradient)
        v, g = device_batched_logjoint_gradient!(ws, params)
        @test v ≈ batched_logjoint_unconstrained(tpo_trailing_pair_model, params, (n,), cm) rtol = 1e-10
        @test g ≈ batched_logjoint_gradient_unconstrained(tpo_trailing_pair_model, params, (n,), cm) rtol = 1e-10
    end
end

# Weakly-informative bulk loop + a tight, informative trailing observation: the
# posterior is dominated by `z`, so reading the loop's first row instead would move
# the recovered mean by many posterior std. A conjugate gaussian gives an analytic
# target to check the device NUTS lands on the RIGHT posterior.
@tea static function tpo_recovery_model(n)
    mu ~ normal(0.0, 3.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 10.0)
    end
    z ~ normal(mu, 0.3)
    return mu
end

@testset "tpo_masked_nuts_recovers_posterior" begin
    n = 300
    rng = MersenneTwister(2270)
    ys = randn(rng, n)
    zval = 2.0
    cm = choicemap(((:y => i, ys[i]) for i = 1:n)..., (:z, zval))

    # analytic gaussian posterior for mu (precision-weighted).
    prec = 1 / 9 + n / 100 + 1 / 0.09
    post_mean = (sum(ys) / 100 + zval / 0.09) / prec

    device = batched_nuts(
        tpo_recovery_model, (n,), cm;
        num_chains=4, num_samples=200, num_warmup=200,
        tree_strategy=:masked, backend=CPU(), rng=MersenneTwister(2271),
    )
    draws = posterior_array(device)
    @test all(isfinite, draws)
    @test isapprox(tpo_mean(draws), post_mean; atol=0.1)
    @test all(<(1.1), rhat(device))
    @test divergencerate(device) < 0.05
end
