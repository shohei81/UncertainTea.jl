# Issue #143: the HOST batched analytic gradient threads over chain-blocks. The
# analytic path is per-column independent (the sufficient-statistics fusion and
# the observed-loop reduction run per chain), so partitioning the batch columns
# into contiguous blocks and writing disjoint totals/gradient slices is BITWISE
# identical to the serial pass -- no shared accumulation, no RNG. These checks
# assert that identity directly (same-process serial-vs-threaded `==` via the
# raw bit pattern, never a golden snapshot) and confirm the work-size gate keeps
# the sufficient-statistics-fused gauss model serial. They pass at both -t 1 (no
# thread plan is built, so serial == serial trivially) and -t 8 (real threading
# vs serial).

# Compare two Float64 arrays by raw bit pattern (a reassociated reduction would
# be only tolerance-equal; disjoint-slice threading must be EXACTLY equal).
function _tbg_bits_equal(a, b)
    size(a) == size(b) || return false
    for index in eachindex(a, b)
        reinterpret(UInt64, a[index]) == reinterpret(UInt64, b[index]) || return false
    end
    return true
end

# Build a gradient cache with the work-size gate forced on/off, restoring the
# threshold afterwards so the toggle never leaks into other tests.
function _tbg_cache(model, params, args, cons; threaded::Bool)
    ref = UncertainTea._BATCHED_GRADIENT_THREAD_WORK_THRESHOLD
    saved = ref[]
    ref[] = threaded ? 1 : typemax(Int)
    try
        return BatchedLogjointGradientCache(model, params, args, cons)
    finally
        ref[] = saved
    end
end

@testset "threaded_batched_gradient" begin
    @tea static function tbg_logistic(X, n)
        alpha ~ normal(0.0, 2.5)
        beta ~ mvnormal(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5),
        )
        for i = 1:n
            {:y => i} ~ bernoullilogit(alpha + sum(beta .* X[:, i]))
        end
        return alpha
    end

    @tea static function tbg_gauss(n)
        mu ~ normal(0.0, 1.0)
        s ~ gamma(2.0, 1.0)
        for i = 1:n
            {:y => i} ~ normal(mu, s)
        end
        return mu
    end

    rng = MersenneTwister(20260728)
    D = 8
    n = 120
    X = randn(rng, D, n)
    yobs = Float64.(rand(rng, Bool, n))
    logistic_cons = choicemap(((:y => i, yobs[i]) for i = 1:n)...)
    C = 64
    params = randn(rng, D + 1, C)

    @testset "logistic gradient/logjoint bitwise serial == threaded" begin
        threaded = _tbg_cache(tbg_logistic, params, (X, n), logistic_cons; threaded=true)
        serial = _tbg_cache(tbg_logistic, params, (X, n), logistic_cons; threaded=false)

        # the analytic backend tier is actually taken, not the flat/column tiers
        @test !isnothing(threaded.backend_cache)
        @test isnothing(threaded.flat_cache)
        @test isempty(threaded.column_caches)
        # the work-size gate builds a real block partition when threads exist
        if Threads.nthreads() > 1
            @test !isnothing(threaded.thread_plan)
            @test length(threaded.thread_plan.ranges) >= 2
            # blocks are contiguous, disjoint, and cover the whole batch
            covered = reduce(vcat, collect.(threaded.thread_plan.ranges))
            @test covered == collect(1:C)
        end
        @test isnothing(serial.thread_plan)

        gt = copy(batched_logjoint_gradient_unconstrained!(threaded, params))
        gs = copy(batched_logjoint_gradient_unconstrained!(serial, params))
        @test _tbg_bits_equal(gt, gs)

        lt = UncertainTea._batched_logjoint_unconstrained_from_gradient_cache!(
            Vector{Float64}(undef, C), threaded, params,
        )
        ls = UncertainTea._batched_logjoint_unconstrained_from_gradient_cache!(
            Vector{Float64}(undef, C), serial, params,
        )
        @test _tbg_bits_equal(lt, ls)

        # combined value + gradient (the sampler entry point)
        vt = Vector{Float64}(undef, C)
        vs = Vector{Float64}(undef, C)
        UncertainTea._batched_logjoint_and_gradient_unconstrained!(vt, threaded, params)
        UncertainTea._batched_logjoint_and_gradient_unconstrained!(vs, serial, params)
        @test _tbg_bits_equal(vt, vs)
        @test _tbg_bits_equal(threaded.gradient_buffer, serial.gradient_buffer)
    end

    @testset "gauss stays serial (work-size gate)" begin
        gn = 400
        gy = randn(rng, gn)
        gcons = choicemap(((:y => i, gy[i]) for i = 1:gn)...)
        gparams = randn(rng, 2, 256)

        # a fully sufficient-statistics-fused loop has zero non-fused observation
        # work, so it never crosses the gate -- serial at any thread count
        gcache = BatchedLogjointGradientCache(tbg_gauss, gparams, (gn,), gcons)
        @test !isnothing(gcache.backend_cache)
        @test UncertainTea._batched_gradient_nonfused_obs(gcache.backend_cache) == 0
        @test isnothing(gcache.thread_plan)

        # even with the gate threshold forced to its minimum the plan stays empty
        # (work == 0), and the gradient still equals a forced-serial reference
        threaded = _tbg_cache(tbg_gauss, gparams, (gn,), gcons; threaded=true)
        @test isnothing(threaded.thread_plan)
        gt = copy(batched_logjoint_gradient_unconstrained!(threaded, gparams))
        gs = copy(batched_logjoint_gradient_unconstrained!(gcache, gparams))
        @test _tbg_bits_equal(gt, gs)
    end

    @testset "batched_nuts draws bitwise serial == threaded" begin
        # the gradient carries no RNG and blocks write disjoint columns, so a
        # fixed-seed batched_nuts run is bitwise identical whether the gradient
        # threads or not. Force the gate on for one run, off for the other.
        function run_nuts(threaded::Bool)
            ref = UncertainTea._BATCHED_GRADIENT_THREAD_WORK_THRESHOLD
            saved = ref[]
            ref[] = threaded ? 1 : typemax(Int)
            try
                return batched_nuts(
                    tbg_logistic,
                    (X, n),
                    logistic_cons;
                    num_chains=C,
                    num_samples=40,
                    num_warmup=30,
                    tree_strategy=:masked,
                    rng=MersenneTwister(4321),
                )
            finally
                ref[] = saved
            end
        end

        threaded_draws = posterior_array(run_nuts(true))
        serial_draws = posterior_array(run_nuts(false))
        @test _tbg_bits_equal(threaded_draws, serial_draws)
    end
end
