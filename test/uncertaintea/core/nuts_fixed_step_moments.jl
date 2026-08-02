# Fixed-step closed-form posterior moment checks for every NUTS path.
#
# NUTS is exact at ANY fixed step size (the multinomial trajectory selection
# leaves the target invariant), so second moments must match the closed-form
# posterior. Before the issue #93 fix -- U-turned/divergent subtrees merged
# into proposal selection -- variances were biased by 10-70% depending on step
# size (over-dispersed at small steps, under-dispersed at large ones), while
# means of symmetric targets stayed correct. These tests pin the variance so
# that regression cannot pass unnoticed again; all tolerances leave several
# sigma of seeded sampling noise but exclude the pre-fix values unambiguously.

function fsm_std(values)
    center = sum(values) / length(values)
    return sqrt(sum(abs2, values .- center) / (length(values) - 1))
end

fsm_var(values) = fsm_std(values)^2

@testset "nuts fixed-step closed-form moments" begin
    @tea static function fsm_prior_normal()
        x ~ normal(0.0, 1.0)
        return x
    end

    # batched_nuts hybrid strategy, standard normal prior: truth sd = 1.0.
    # Pre-fix: sd = 1.53 at step 0.3.
    fsm_hybrid = batched_nuts(
        fsm_prior_normal,
        (),
        choicemap();
        num_chains=64,
        num_samples=1500,
        num_warmup=100,
        step_size=0.3,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        max_tree_depth=6,
        rng=MersenneTwister(7),
    )
    fsm_hybrid_sd = fsm_std(vec(posterior_array(fsm_hybrid)[:, :, 1]))
    @test 0.94 <= fsm_hybrid_sd <= 1.06

    # batched_nuts masked strategy: truth sd = 1.0. Pre-fix: sd = 1.63 at
    # step 0.2.
    fsm_masked = batched_nuts(
        fsm_prior_normal,
        (),
        choicemap();
        num_chains=64,
        num_samples=1500,
        num_warmup=100,
        step_size=0.2,
        adapt_step_size=false,
        adapt_mass_matrix=false,
        max_tree_depth=6,
        tree_strategy=:masked,
        rng=MersenneTwister(7),
    )
    fsm_masked_sd = fsm_std(vec(posterior_array(fsm_masked)[:, :, 1]))
    @test 0.94 <= fsm_masked_sd <= 1.06

    # Scalar nuts on a conjugate model: posterior exactly N(0.7, 0.5).
    # Pre-fix variances: ~1.10 at step 0.25 (too high) and ~0.37 at step 1.2
    # (too low), so the band below fails on both sides of the pre-fix bias.
    @tea static function fsm_conjugate()
        mu ~ normal(0.0, 1.0)
        {:y} ~ normal(mu, 1.0)
        return mu
    end
    fsm_constraints = choicemap((:y, 1.4))
    for (fsm_step, fsm_seed) in ((0.25, 1), (1.2, 2))
        fsm_chain = first(
            nuts(
                fsm_conjugate,
                (),
                fsm_constraints;
                num_samples=6000,
                num_warmup=300,
                step_size=fsm_step,
                adapt_step_size=false,
                adapt_mass_matrix=false,
                max_tree_depth=8,
                rng=MersenneTwister(fsm_seed),
            ),
        )
        fsm_draws = vec(fsm_chain.constrained_samples)
        @test 0.41 <= fsm_var(fsm_draws) <= 0.60
        @test abs(sum(fsm_draws) / length(fsm_draws) - 0.7) < 0.08
    end

    # Tempered-SMC NUTS move at beta = 1.0 on the prior normal: repeated moves
    # must leave N(0, 1) invariant at a fixed move step size. Pre-fix the
    # merged invalid subtrees inflated the stationary sd here as well.
    fsm_move_rng = MersenneTwister(11)
    fsm_particles = randn(fsm_move_rng, 1, 256)
    fsm_logjoint = zeros(256)
    fsm_logproposal = zeros(256)
    fsm_logratio = zeros(256)
    fsm_move_draws = Float64[]
    for fsm_move_index = 1:120
        UncertainTea._batched_nuts_move!(
            fsm_particles,
            fsm_logjoint,
            fsm_logproposal,
            fsm_logratio,
            fsm_prior_normal,
            (),
            choicemap(),
            [0.0],
            [0.0],
            1.0,
            0.4,
            5,
            1000.0,
            [1.0],
            fsm_move_rng,
        )
        fsm_move_index > 20 && append!(fsm_move_draws, vec(fsm_particles))
    end
    fsm_move_sd = fsm_std(fsm_move_draws)
    @test 0.94 <= fsm_move_sd <= 1.06
end
