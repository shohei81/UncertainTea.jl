# MH-within-Gibbs (docs/mh-within-gibbs.md, issue #13 track 2): symmetric
# single-site MH on discrete latents alternating with NUTS on the continuous
# block. Track 1's enumeration provides the mixed-model oracle; pure-discrete
# posteriors are checked against direct truncated enumeration.

# Local mean helper (Statistics is not imported by the test harness).
gibbs_mean(x) = sum(x) / length(x)

# pure discrete: no continuous slots at all
@tea static function gibbs_pure_poisson_model()
    z ~ poisson(3.0)
    {:y} ~ normal(1.0 * z, 0.5)
    return z
end

# the indicator mixture in its unmarginalized (Gibbs) spelling ...
@tea static function gibbs_indicator_model()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    z ~ bernoulli(0.3)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
    return m1
end

# ... and its track-1 enumerated spelling: the acceptance oracle
@tea static function gibbs_indicator_enum_model()
    m1 ~ normal(-2.0, 1.0)
    m2 ~ normal(2.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
    return m1
end

# loop-scoped bernoulli sites sharing one template spec
@tea static function gibbs_loop_model(n)
    p ~ beta(2.0, 2.0)
    for i = 1:n
        {:z => i} ~ bernoulli(p)
    end
    {:y} ~ normal(10.0 * p, 1.0)
    return p
end

# a binomial site whose walk proposals can leave the dynamic support: the
# suffix indexes ps[z + 1], so z = 3 throws and must reject, not crash
@tea static function gibbs_dynamic_bound_model(ps)
    q ~ beta(2.0, 2.0)
    z ~ binomial(2, q)
    {:y} ~ normal(ps[z+1], 1.0)
    return q
end

# a marginalized site must stay OUT of the Gibbs site set
@tea static function gibbs_mixed_marginalized_model()
    mu ~ normal(0.0, 1.0)
    m ~ bernoulli(0.4; marginalize=:enumerate)
    z ~ poisson(2.0)
    {:y} ~ normal(mu + m + z, 0.8)
    return mu
end

# a slotless CONTINUOUS latent has no sampler and must error clearly
@tea static function gibbs_continuous_slotless_model(n)
    for i = 1:n
        {:x => i} ~ normal(0.0, 1.0)
    end
    {:y} ~ normal(1.0, 1.0)
    return n
end

# a discrete site inside a SUBMODEL: the trace address is prefixed
# ((:sub, :z)) and only the inlined execution plan carries its spec
@tea static function gibbs_count_child()
    z ~ poisson(2.0)
    return z
end

@tea static function gibbs_nested_model()
    mu ~ normal(0.0, 1.0)
    z = ({:sub} ~ gibbs_count_child())
    {:y} ~ normal(mu + z, 0.8)
    return mu
end

# a marginalized site with a DYNAMIC (non-loop) address: the exclusion must
# match by template, not by forcing a static address
@tea static function gibbs_dynamic_marginalized_model(k)
    mu ~ normal(0.0, 1.0)
    {:m => k} ~ bernoulli(0.4; marginalize=:enumerate)
    z ~ poisson(2.0)
    {:y} ~ normal(mu + z, 0.8)
    return mu
end

# a wildcard (dynamic-address) template coexisting with a literal site of a
# DIFFERENT family: the exact literal match must win over the wildcard
@tea static function gibbs_wildcard_model(addr)
    {addr} ~ poisson(2.0)
    z ~ bernoulli(0.5)
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

# the vararg categorical form carries K as the argument count
@tea static function gibbs_vararg_categorical_model()
    mu ~ normal(0.0, 1.0)
    z ~ categorical(0.2, 0.3, 0.5)
    {:y} ~ normal(mu + z, 1.0)
    return mu
end

# a fixed-shape loop whose iterator SHADOWS a sampled symbol: the site set is
# fixed, so this valid model must run
@tea static function gibbs_shadowing_iterator_model(n)
    z ~ poisson(2.0)
    for z = 1:n
        {:w => z} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

# a singleton categorical support can never move and must self-transition
@tea static function gibbs_singleton_model()
    mu ~ normal(0.0, 1.0)
    z ~ categorical([1.0])
    {:y} ~ normal(mu + z, 1.0)
    return mu
end

# observations rule out most prior draws of z (binomial needs z >= y): the
# initializer must retry, or accept an explicit initial_discrete seed
@tea static function gibbs_constrained_init_model()
    z ~ poisson(3.0)
    {:y} ~ binomial(z, 0.6)
    return z
end

# a prior draw can make `generate` ITSELF throw (ps[z + 1] out of bounds for
# z >= 3): the initializer must retry through that too
@tea static function gibbs_throwing_init_model(ps)
    z ~ poisson(1.0)
    {:y} ~ normal(ps[z+1], 1.0)
    return z
end

# a loop bound depending on a Gibbs site changes the latent-choice SET
# (trans-dimensional): rejected at construction, not silently biased
@tea static function gibbs_transdim_loop_model()
    z ~ poisson(2.0)
    for i = 1:z
        {:w => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

# same for a dynamic address depending on the site's binding
@tea static function gibbs_transdim_address_model()
    z ~ poisson(2.0)
    {:w => z} ~ bernoulli(0.5)
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

# a CONTINUOUS parameter shaping the site set is trans-dimensional too: NUTS
# moves would change which {:z => i} exist
@tea static function gibbs_continuous_shape_model()
    x ~ lognormal(1.0, 0.5)
    n = floor(Int, x) + 1
    for i = 1:n
        {:z => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * n, 1.0)
    return x
end

# a categorical with a dynamic probability vector hides the support size and
# a +-1 walk could strand in a connected component: rejected honestly
@tea static function gibbs_dynamic_categorical_model(ps)
    mu ~ normal(0.0, 1.0)
    z ~ categorical(ps)
    {:y} ~ normal(mu + z, 1.0)
    return mu
end

# the continuous-shaped loop can draw ZERO controlled sites at the seed; the
# rejection must fire regardless of the drawn shape
@tea static function gibbs_zero_site_seed_model()
    x ~ lognormal(0.0, 0.3)
    for i = 1:(floor(Int, x)-1)
        {:z => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * x, 1.0)
    return x
end

# a latent-dependent loop WITHOUT sites followed by a fixed-shape site loop
# reusing the iterator name: the iterator taint is scoped, so this valid
# static-shape model must run
@tea static function gibbs_iterator_reuse_model(n)
    z ~ poisson(2.0)
    total = 0.0
    for i = 1:z
        total = total + 1.0
    end
    for i = 1:n
        {:w => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * z + total, 1.0)
    return z
end

# a MARGINALIZED binding shaping the site set is sampled per enumeration
# branch: equally trans-dimensional, equally rejected
@tea static function gibbs_marginalized_shape_model()
    m ~ categorical([0.5, 0.5]; marginalize=:enumerate)
    for i = 1:m
        {:w => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * m, 1.0)
    return m
end

# a CONSTRAINED broadcast observation is fixed data: its binding must not
# taint the site loop sized from it
@tea static function gibbs_broadcast_obs_model(mu_vec)
    z ~ poisson(2.0)
    y = ({:y} ~ normal.(mu_vec, 1.0))
    for i = 1:length(y)
        {:w => i} ~ bernoulli(0.5)
    end
    {:v} ~ normal(1.0 * z, 1.0)
    return z
end

# a clean assignment BEFORE the sample must not launder the binding: the
# choice re-taints z at its own (later) step
@tea static function gibbs_laundered_binding_model()
    z = 0
    z ~ poisson(2.0)
    for i = 1:z
        {:w => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

# a reassignment inside a maybe-empty loop must NOT clear the taint: at
# runtime the loop body never runs and n keeps depending on z
@tea static function gibbs_loop_shadowed_reassign_model()
    z ~ poisson(2.0)
    n = z + 1
    for j = 1:0
        n = 2
    end
    for i = 1:n
        {:w => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

# an OBSERVED marginalized choice is conditioned to a fixed value, so a shape
# depending on it stays static and the model must run
@tea static function gibbs_observed_marginalized_shape_model()
    m ~ categorical([0.5, 0.5]; marginalize=:enumerate)
    for i = 1:m
        {:w => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * m, 1.0)
    return m
end

# a straight-line reassignment from a clean expression clears the taint:
# the loop bound no longer depends on z
@tea static function gibbs_reassigned_bound_model()
    z ~ poisson(2.0)
    n = z + 1
    n = 2
    for i = 1:n
        {:w => i} ~ bernoulli(0.5)
    end
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

# prior continuous draws almost always make the suffix throw (lognormal(5,1)
# indexing a 3-vector); a provided initial_params pins the slots during
# discovery so the seeded start works like plain NUTS
@tea static function gibbs_hostile_prior_model(ps)
    x ~ lognormal(5.0, 1.0)
    z ~ bernoulli(0.5)
    {:y} ~ normal(ps[floor(Int, x)+1] + 1.0 * z, 1.0)
    return x
end

# the shape dependence flows through an assignment inside the tainted loop
# (`w_last` tracks the iteration count), not through the site binding itself
@tea static function gibbs_iterator_taint_model()
    w_last = 1
    z ~ poisson(2.0)
    for i = 1:z
        w_last = i + 1
    end
    {:w => w_last} ~ bernoulli(0.5)
    {:y} ~ normal(1.0 * z, 1.0)
    return z
end

@testset "mh_within_gibbs" begin
    @testset "gibbs_pure_discrete_vs_enumeration" begin
        # posterior over z by direct truncated enumeration of the support
        gibbs_pure_lp = [
            UncertainTea.logpdf(poisson(3.0), k) + UncertainTea.logpdf(normal(1.0 * k, 0.5), 4.2) for
            k = 0:60
        ]
        gibbs_pure_weights = exp.(gibbs_pure_lp .- maximum(gibbs_pure_lp))
        gibbs_pure_weights ./= sum(gibbs_pure_weights)
        gibbs_pure_oracle = sum((0:60) .* gibbs_pure_weights)

        gibbs_pure_draws = Float64[]
        for gibbs_pure_seed = 1:2
            gibbs_pure_chain = gibbs(
                gibbs_pure_poisson_model,
                (),
                choicemap((:y, 4.2));
                num_samples=3000,
                num_warmup=500,
                rng=MersenneTwister(40 + gibbs_pure_seed),
            )
            @test isempty(gibbs_pure_chain.mass_matrix)
            @test size(gibbs_pure_chain.constrained_samples, 1) == 0
            append!(gibbs_pure_draws, gibbs_pure_chain.discrete_samples[1, :])
        end
        @test gibbs_mean(gibbs_pure_draws) ≈ gibbs_pure_oracle atol = 0.15
    end

    @testset "gibbs_vs_enumerated_nuts" begin
        gibbs_ind_constraints = choicemap((:y, 0.8))
        gibbs_ind_m1 = Float64[]
        gibbs_ind_m2 = Float64[]
        gibbs_enum_m1 = Float64[]
        gibbs_enum_m2 = Float64[]
        gibbs_ind_z = Float64[]
        for gibbs_ind_seed = 1:3
            gibbs_ind_chain = gibbs(
                gibbs_indicator_model,
                (),
                gibbs_ind_constraints;
                num_samples=600,
                num_warmup=300,
                rng=MersenneTwister(100 + gibbs_ind_seed),
            )
            gibbs_enum_chain = first(
                nuts(
                    gibbs_indicator_enum_model,
                    (),
                    gibbs_ind_constraints;
                    num_samples=600,
                    num_warmup=300,
                    rng=MersenneTwister(200 + gibbs_ind_seed),
                ),
            )
            @test all(isfinite, gibbs_ind_chain.constrained_samples)
            @test gibbs_ind_chain.discrete_addresses == Any[(:z,)]
            append!(gibbs_ind_m1, gibbs_ind_chain.constrained_samples[1, :])
            append!(gibbs_ind_m2, gibbs_ind_chain.constrained_samples[2, :])
            append!(gibbs_enum_m1, gibbs_enum_chain.constrained_samples[1, :])
            append!(gibbs_enum_m2, gibbs_enum_chain.constrained_samples[2, :])
            append!(gibbs_ind_z, gibbs_ind_chain.discrete_samples[1, :])
        end
        @test gibbs_mean(gibbs_ind_m1) ≈ gibbs_mean(gibbs_enum_m1) atol = 0.2
        @test gibbs_mean(gibbs_ind_m2) ≈ gibbs_mean(gibbs_enum_m2) atol = 0.2
        @test 0.0 < gibbs_mean(gibbs_ind_z) < 1.0
    end

    @testset "gibbs_loop_sites_and_determinism" begin
        gibbs_loop_chain = gibbs(
            gibbs_loop_model,
            (3,),
            choicemap((:y, 7.0));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(5),
        )
        @test gibbs_loop_chain.discrete_addresses == Any[(:z, 1), (:z, 2), (:z, 3)]
        @test all(isfinite, gibbs_loop_chain.constrained_samples)
        @test all(value -> value in (0, 1), gibbs_loop_chain.discrete_samples)

        gibbs_loop_replay = gibbs(
            gibbs_loop_model,
            (3,),
            choicemap((:y, 7.0));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(5),
        )
        @test gibbs_loop_replay.constrained_samples == gibbs_loop_chain.constrained_samples
        @test gibbs_loop_replay.discrete_samples == gibbs_loop_chain.discrete_samples
    end

    @testset "gibbs_dynamic_bound_rejection" begin
        gibbs_bound_chain = gibbs(
            gibbs_dynamic_bound_model,
            ([0.0, 1.0, 2.0],),
            choicemap((:y, 1.4));
            num_samples=800,
            num_warmup=200,
            rng=MersenneTwister(7),
        )
        @test all(isfinite, gibbs_bound_chain.logjoint_values)
        @test minimum(gibbs_bound_chain.discrete_samples) >= 0
        @test maximum(gibbs_bound_chain.discrete_samples) <= 2
    end

    @testset "gibbs_site_discovery" begin
        # marginalized sites stay marginalized; only the poisson site remains
        gibbs_mixed_chain = gibbs(
            gibbs_mixed_marginalized_model,
            (),
            choicemap((:y, 2.4));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(9),
        )
        @test gibbs_mixed_chain.discrete_addresses == Any[(:z,)]
        @test all(isfinite, gibbs_mixed_chain.logjoint_values)

        # no discrete sites: degrades to plain NUTS semantics
        gibbs_none_chain = gibbs(
            gibbs_indicator_enum_model,
            (),
            choicemap((:y, 0.8));
            num_samples=100,
            num_warmup=100,
            rng=MersenneTwister(13),
        )
        @test isempty(gibbs_none_chain.discrete_addresses)
        @test all(isfinite, gibbs_none_chain.constrained_samples)

        # slotless continuous latents error with a clear message
        @test_throws ArgumentError gibbs(
            gibbs_continuous_slotless_model,
            (2,),
            choicemap((:y, 0.5));
            num_samples=10,
            rng=MersenneTwister(3),
        )

        # a submodel latent matches through the inlined plan's prefixed address
        gibbs_nested_chain = gibbs(
            gibbs_nested_model,
            (),
            choicemap((:y, 3.1));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(17),
        )
        @test gibbs_nested_chain.discrete_addresses == Any[(:sub, :z)]
        @test all(isfinite, gibbs_nested_chain.logjoint_values)

        # a marginalized site with a dynamic address stays excluded by
        # template matching (no static-address requirement)
        gibbs_dynamic_marg_chain = gibbs(
            gibbs_dynamic_marginalized_model,
            (7,),
            choicemap((:y, 2.4));
            num_samples=100,
            num_warmup=100,
            rng=MersenneTwister(19),
        )
        @test gibbs_dynamic_marg_chain.discrete_addresses == Any[(:z,)]
        @test all(isfinite, gibbs_dynamic_marg_chain.logjoint_values)

        # exact literal matches beat wildcard templates: the bernoulli site
        # is classified as a flip, not shadowed by the {addr} poisson
        gibbs_wildcard_chain = gibbs(
            gibbs_wildcard_model,
            (:w,),
            choicemap((:y, 0.6));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(21),
        )
        @test gibbs_wildcard_chain.discrete_addresses == Any[(:w,), (:z,)]
        @test all(value -> value in (0, 1), gibbs_wildcard_chain.discrete_samples[2, :])
        @test all(value -> value >= 0, gibbs_wildcard_chain.discrete_samples[1, :])

        # a Pair-valued dynamic address normalizes to multiple parts and
        # cannot be matched: clear error, not silent misclassification
        @test_throws ArgumentError gibbs(
            gibbs_wildcard_model,
            (:a => 1,),
            choicemap((:y, 0.6));
            num_samples=10,
            rng=MersenneTwister(23),
        )

        # the vararg categorical form classifies with K = argument count
        gibbs_vararg_chain = gibbs(
            gibbs_vararg_categorical_model,
            (),
            choicemap((:y, 1.8));
            num_samples=200,
            num_warmup=100,
            rng=MersenneTwister(61),
        )
        @test all(value -> value in 1:3, gibbs_vararg_chain.discrete_samples)
        @test all(isfinite, gibbs_vararg_chain.constrained_samples)

        # a fixed loop shadowing a sampled iterator name stays a valid
        # static-shape model
        gibbs_shadow_chain = gibbs(
            gibbs_shadowing_iterator_model,
            (2,),
            choicemap((:y, 2.0));
            num_samples=100,
            num_warmup=50,
            rng=MersenneTwister(63),
        )
        @test (:z,) in gibbs_shadow_chain.discrete_addresses
        @test (:w, 1) in gibbs_shadow_chain.discrete_addresses
        @test all(isfinite, gibbs_shadow_chain.logjoint_values)

        # a singleton categorical self-transitions instead of erroring
        gibbs_singleton_chain = gibbs(
            gibbs_singleton_model,
            (),
            choicemap((:y, 1.4));
            num_samples=100,
            num_warmup=50,
            rng=MersenneTwister(25),
        )
        @test all(==(1), gibbs_singleton_chain.discrete_samples)
        @test all(isfinite, gibbs_singleton_chain.constrained_samples)
    end

    @testset "gibbs_initialization" begin
        # a moderately constrained observation: the bounded prior retry finds
        # a supported state (P(z >= 5) under poisson(3) is ~0.18)
        gibbs_init_chain = gibbs(
            gibbs_constrained_init_model,
            (),
            choicemap((:y, 5));
            num_samples=300,
            num_warmup=100,
            rng=MersenneTwister(33),
        )
        @test all(>=(5), gibbs_init_chain.discrete_samples)

        # a hopeless prior (P(z >= 12) ~ 5e-5): initial_discrete seeds the
        # supported state deterministically
        gibbs_seeded_chain = gibbs(
            gibbs_constrained_init_model,
            (),
            choicemap((:y, 12));
            num_samples=300,
            num_warmup=100,
            initial_discrete=choicemap((:z, 14)),
            rng=MersenneTwister(35),
        )
        @test all(>=(12), gibbs_seeded_chain.discrete_samples)

        # initial_discrete addresses must be discovered sites
        @test_throws ArgumentError gibbs(
            gibbs_constrained_init_model,
            (),
            choicemap((:y, 5));
            num_samples=10,
            initial_discrete=choicemap((:nope, 1)),
            rng=MersenneTwister(37),
        )

        # proposal-scoring rejection stays scoped to support/domain failures
        @test UncertainTea._gibbs_rejectable_error(BoundsError([1.0], 2))
        @test UncertainTea._gibbs_rejectable_error(DomainError(-1.0))
        @test !UncertainTea._gibbs_rejectable_error(InterruptException())
        @test !UncertainTea._gibbs_rejectable_error(ErrorException("bug"))

        # a prior draw that makes generate itself throw is retried
        gibbs_throwing_chain = gibbs(
            gibbs_throwing_init_model,
            ([0.0, 1.0, 2.0],),
            choicemap((:y, 1.2));
            num_samples=300,
            num_warmup=100,
            rng=MersenneTwister(39),
        )
        @test all(value -> 0 <= value <= 2, gibbs_throwing_chain.discrete_samples)
        @test all(isfinite, gibbs_throwing_chain.logjoint_values)

        # a seed enters the prior draws as a constraint, so a model whose
        # UNSEEDED draws almost always throw (only z = 0 is evaluable)
        # initializes deterministically from initial_discrete
        gibbs_seeded_throwing_chain = gibbs(
            gibbs_throwing_init_model,
            ([0.5],),
            choicemap((:y, 0.4));
            num_samples=100,
            num_warmup=50,
            initial_discrete=choicemap((:z, 0)),
            rng=MersenneTwister(49),
        )
        @test all(==(0), gibbs_seeded_throwing_chain.discrete_samples)
    end

    @testset "gibbs_transdimensional_rejection" begin
        # a Gibbs site shaping the latent-choice set is rejected honestly at
        # construction (up-moves would need missing choices, down-moves would
        # leave stale ones -- a silent bias otherwise)
        @test_throws ArgumentError gibbs(
            gibbs_transdim_loop_model,
            (),
            choicemap((:y, 2.0));
            num_samples=10,
            rng=MersenneTwister(41),
        )
        @test_throws ArgumentError gibbs(
            gibbs_transdim_address_model,
            (),
            choicemap((:y, 2.0));
            num_samples=10,
            rng=MersenneTwister(43),
        )
        # ... and for a continuous parameter shaping the site set
        @test_throws ArgumentError gibbs(
            gibbs_continuous_shape_model,
            (),
            choicemap((:y, 3.0));
            num_samples=10,
            rng=MersenneTwister(45),
        )
        # a constraint naming a SLOTTED latent does not fix it (the logjoint
        # samples it from the position), so the rejection stands
        @test_throws ArgumentError gibbs(
            gibbs_continuous_shape_model,
            (),
            choicemap((:x, 1.2), (:y, 3.0));
            num_samples=10,
            rng=MersenneTwister(46),
        )
        # a dynamic categorical support cannot guarantee a connected proposal
        @test_throws ArgumentError gibbs(
            gibbs_dynamic_categorical_model,
            ([0.3, 0.7],),
            choicemap((:y, 1.4));
            num_samples=10,
            rng=MersenneTwister(47),
        )
        # a PARTIALLY observed dynamic template stays a potential site: one
        # constrained (:w, 1) must not launder the whole loop as observed
        @test_throws ArgumentError gibbs(
            gibbs_transdim_loop_model,
            (),
            choicemap((:w => 1, false), (:y, 2.0));
            num_samples=10,
            rng=MersenneTwister(51),
        )
        # the rejection is shape-independent: seeds whose prior draw has zero
        # controlled sites still reject at construction
        for gibbs_zero_seed = 1:3
            @test_throws ArgumentError gibbs(
                gibbs_zero_site_seed_model,
                (),
                choicemap((:y, 1.0));
                num_samples=10,
                rng=MersenneTwister(52 + gibbs_zero_seed),
            )
        end
        # taint flows through assignments inside the tainted loop (iteration
        # count dependence), not only through the site binding itself
        @test_throws ArgumentError gibbs(
            gibbs_iterator_taint_model,
            (),
            choicemap((:y, 2.0));
            num_samples=10,
            rng=MersenneTwister(57),
        )
        # ... but iterator taint is SCOPED: a fixed-shape site loop reusing
        # the iterator name after a latent-dependent site-free loop is valid
        gibbs_reuse_chain = gibbs(
            gibbs_iterator_reuse_model,
            (2,),
            choicemap((:y, 3.0));
            num_samples=100,
            num_warmup=50,
            rng=MersenneTwister(59),
        )
        @test (:z,) in gibbs_reuse_chain.discrete_addresses
        @test (:w, 1) in gibbs_reuse_chain.discrete_addresses
        @test (:w, 2) in gibbs_reuse_chain.discrete_addresses
        @test all(isfinite, gibbs_reuse_chain.logjoint_values)

        # a marginalized binding shaping the site set is rejected too
        @test_throws ArgumentError gibbs(
            gibbs_marginalized_shape_model,
            (),
            choicemap((:y, 1.5));
            num_samples=10,
            rng=MersenneTwister(65),
        )
        # ... unless the marginalized choice is observed (fixed value):
        # OBSERVING m fixes the shape and the model runs
        gibbs_obs_marg_chain = gibbs(
            gibbs_observed_marginalized_shape_model,
            (),
            choicemap((:m, 2), (:y, 2.0));
            num_samples=100,
            num_warmup=50,
            rng=MersenneTwister(71),
        )
        @test (:w, 1) in gibbs_obs_marg_chain.discrete_addresses
        @test (:w, 2) in gibbs_obs_marg_chain.discrete_addresses
        @test all(isfinite, gibbs_obs_marg_chain.logjoint_values)
        # a reassignment inside a maybe-empty loop keeps the outer taint:
        # this model still depends on z at runtime and rejects
        @test_throws ArgumentError gibbs(
            gibbs_loop_shadowed_reassign_model,
            (),
            choicemap((:y, 2.0));
            num_samples=10,
            rng=MersenneTwister(73),
        )
        # a clean assignment BEFORE the sample does not launder the binding
        @test_throws ArgumentError gibbs(
            gibbs_laundered_binding_model,
            (),
            choicemap((:y, 2.0));
            num_samples=10,
            rng=MersenneTwister(75),
        )
        # a constrained broadcast observation is fixed data: sizing a site
        # loop from its binding keeps the shape static and the model runs
        gibbs_bcast_chain = gibbs(
            gibbs_broadcast_obs_model,
            ([0.0, 0.5, 1.0],),
            choicemap((:y, Float32[0.1, 0.6, 0.9]), (:v, 2.0));
            num_samples=100,
            num_warmup=50,
            rng=MersenneTwister(77),
        )
        @test (:w, 1) in gibbs_bcast_chain.discrete_addresses
        @test (:w, 3) in gibbs_bcast_chain.discrete_addresses
        @test all(isfinite, gibbs_bcast_chain.logjoint_values)

        # a straight-line reassignment from a clean expression clears the
        # taint: this fixed-shape model runs
        gibbs_reassigned_chain = gibbs(
            gibbs_reassigned_bound_model,
            (),
            choicemap((:y, 2.0));
            num_samples=100,
            num_warmup=50,
            rng=MersenneTwister(67),
        )
        @test (:w, 1) in gibbs_reassigned_chain.discrete_addresses
        @test (:w, 2) in gibbs_reassigned_chain.discrete_addresses
        @test all(isfinite, gibbs_reassigned_chain.logjoint_values)
    end

    @testset "gibbs_initial_params_pinning" begin
        # prior continuous draws almost always throw (P(x < 2) ~ 1e-5 under
        # lognormal(5, 1)), so unseeded initialization exhausts its retries;
        # the provided initial_params pins the slots during discovery and the
        # seeded start works. The continuous pass runs with a frozen tiny
        # step so the trajectory cannot wander into the throwing region --
        # this pins the INITIALIZATION behavior, not sampling robustness for
        # a model that is genuinely unevaluable over most of its prior.
        gibbs_hostile_chain = gibbs(
            gibbs_hostile_prior_model,
            ([0.0, 1.0, 2.0],),
            choicemap((:y, 1.4));
            num_samples=20,
            num_warmup=0,
            initial_params=[0.0],
            step_size=1e-10,
            max_tree_depth=1,
            adapt_step_size=false,
            adapt_mass_matrix=false,
            rng=MersenneTwister(69),
        )
        @test all(isfinite, gibbs_hostile_chain.logjoint_values)
        @test all(value -> value in (0, 1), gibbs_hostile_chain.discrete_samples)
    end

    @testset "gibbs_polish" begin
        # the geometric-tailed walk stays a correct symmetric proposal: the
        # pure-discrete posterior still matches the enumeration oracle
        gibbs_tail_lp = [
            UncertainTea.logpdf(poisson(3.0), k) + UncertainTea.logpdf(normal(1.0 * k, 0.5), 4.2) for
            k = 0:60
        ]
        gibbs_tail_weights = exp.(gibbs_tail_lp .- maximum(gibbs_tail_lp))
        gibbs_tail_weights ./= sum(gibbs_tail_weights)
        gibbs_tail_oracle = sum((0:60) .* gibbs_tail_weights)
        gibbs_tail_chain = gibbs(
            gibbs_pure_poisson_model,
            (),
            choicemap((:y, 4.2));
            num_samples=3000,
            num_warmup=500,
            discrete_tail=0.3,
            rng=MersenneTwister(81),
        )
        @test gibbs_mean(gibbs_tail_chain.discrete_samples[1, :]) ≈ gibbs_tail_oracle atol = 0.15
        @test_throws ArgumentError gibbs(
            gibbs_pure_poisson_model,
            (),
            choicemap((:y, 4.2));
            num_samples=10,
            discrete_tail=1.5,
            rng=MersenneTwister(83),
        )

        # per-site split-chain ESS aligns with the addresses
        gibbs_ess_values = discrete_ess(gibbs_tail_chain)
        @test length(gibbs_ess_values) == 1
        @test all(value -> isfinite(value) && value > 0, gibbs_ess_values)

        # the SBC harness runs with the Gibbs kernel: the caller names the
        # observed data so :z stays a latent Gibbs site, and the continuous
        # ranks calibrate; without observation_addresses the default would
        # condition every non-slot choice (no free sites), so it must throw
        @test_throws ArgumentError sbc(
            gibbs_indicator_model;
            num_simulations=2,
            num_samples=4,
            num_warmup=10,
            sampler=:gibbs,
            rng=MersenneTwister(84),
        )
        gibbs_sbc_result = sbc(
            gibbs_indicator_model;
            num_simulations=40,
            num_samples=24,
            num_warmup=60,
            sampler=:gibbs,
            observation_addresses=[(:y,)],
            rng=MersenneTwister(85),
        )
        @test size(gibbs_sbc_result.ranks) == (2, 40)
        @test all(0 .<= gibbs_sbc_result.ranks .<= 24)
        @test all(pvalue -> pvalue > 1e-3, gibbs_sbc_result.pvalues)
    end

    @testset "gibbs_nuts_option_parity" begin
        # the mirrored NUTS options validate like nuts itself
        @test_throws ArgumentError gibbs(
            gibbs_indicator_model,
            (),
            choicemap((:y, 0.8));
            num_samples=10,
            step_size=0.0,
            rng=MersenneTwister(27),
        )
        @test_throws ArgumentError gibbs(
            gibbs_indicator_model,
            (),
            choicemap((:y, 0.8));
            num_samples=10,
            target_accept=1.5,
            rng=MersenneTwister(27),
        )

        # dense-metric runs keep the adapted dense inverse mass
        gibbs_dense_chain = gibbs(
            gibbs_indicator_model,
            (),
            choicemap((:y, 0.8));
            num_samples=100,
            num_warmup=150,
            metric=:dense,
            rng=MersenneTwister(29),
        )
        @test gibbs_dense_chain.dense_mass_matrix isa Matrix{Float64}
        @test size(gibbs_dense_chain.dense_mass_matrix) == (2, 2)
        gibbs_diag_chain = gibbs(
            gibbs_indicator_model,
            (),
            choicemap((:y, 0.8));
            num_samples=50,
            num_warmup=50,
            rng=MersenneTwister(31),
        )
        @test isnothing(gibbs_diag_chain.dense_mass_matrix)
    end
end
