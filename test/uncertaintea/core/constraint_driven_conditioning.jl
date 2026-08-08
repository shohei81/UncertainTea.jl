# Constraint-driven latent/observation classification (issue #95,
# docs/constraint-driven-conditioning.md). A choice is an observation iff its
# address is in the constraints at inference time; binding is orthogonal. These
# regression tests pin the PR-2 scope: the compiled `logjoint` path derives the
# latent/observation split from the conditioning signature, so it agrees with
# `generate`/`assess`, honors a constraint on a BOUND address (the #95 bug),
# treats an unbound unconstrained choice as a slotted latent, and memoizes the
# resolved plan per signature.

# The #95 reproduction model: `:y` is bound (`y = ({:y} ~ ...)`) yet may be
# observed. Under the old syntactic rule the compiled scoring silently ignored a
# constraint on `:y`; under the conditioning rule it is honored.
@tea static function cdc_chain()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    {:z} ~ normal(y, 1.0)
    return mu
end

@tea static function cdc_gaussian()
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# PR-3 value-resolution models. Each threads a possibly-observed `:y` through a
# downstream expression, so the evaluator must resolve `:y` to its constrained
# value (when observed) or its latent value (when not) via one path, in both the
# scoring fold and the dependent-transform walk.

# `:y` observed value feeds a deterministic step that also depends on the latent
# `mu`, so the latent gradient must pick up the observed contribution.
@tea static function cdc_downstream()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    w = y * 2.0 + mu
    {:z} ~ normal(w, 1.0)
    return mu
end

# `:y` observed value feeds a reparam=:noncentered loc/scale (theta = y + z).
# Before PR-3 the transform walk poisoned the slotless `:y` binding and this
# raised, even though the observed value is known. Centered twin for crosscheck.
@tea static function cdc_noncentered_obs()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    theta ~ normal(y, 1.0; reparam=:noncentered)
    {:z} ~ normal(theta, 1.0)
    return mu
end

@tea static function cdc_centered_obs()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    theta ~ normal(y, 1.0)
    {:z} ~ normal(theta, 1.0)
    return mu
end

@testset "constraint_driven_conditioning" begin
    normlogpdf(mean, sd, x) = UncertainTea.logpdf(normal(mean, sd), x)

    @testset "#95 bound observation is honored and agrees with assess" begin
        mu0 = 0.5
        yobs = 100.0
        zobs = 0.0
        cons = choicemap(:y => yobs, :z => zobs)

        # signature {:y, :z}: only `mu` is a latent, so the compiled scoring
        # path takes a single parameter and reads y, z from the constraints.
        layout = UncertainTea._resolve_signature_plan(cdc_chain, cons, ()).plan.parameter_layout
        @test parametervaluecount(layout) == 1
        @test length(layout.slots) == 1
        @test layout.slots[1].binding == :mu

        lj = logjoint(cdc_chain, [mu0], (), cons)
        manual = normlogpdf(0.0, 1.0, mu0) + normlogpdf(mu0, 1.0, yobs) + normlogpdf(yobs, 1.0, zobs)
        assessed = assess(cdc_chain, (), choicemap(:mu => mu0, :y => yobs, :z => zobs))

        @test lj ≈ manual atol = 1e-9
        @test lj ≈ assessed atol = 1e-9

        # The observation is genuinely conditioned on: a different constrained
        # value gives a different density (the #95 bug returned the same value
        # regardless, because the bound `:y` was scored from a free parameter).
        @test logjoint(cdc_chain, [mu0], (), choicemap(:y => yobs, :z => zobs)) !=
              logjoint(cdc_chain, [mu0], (), choicemap(:y => yobs + 5.0, :z => zobs))
    end

    @testset "#95 conjugate posterior (closed form)" begin
        # mu ~ N(0,1), y | mu ~ N(mu,1) observed at yobs. The z term does not
        # depend on mu, so as a function of mu the logjoint is the unnormalized
        # log posterior N(yobs/2, 1/2). Check the shape against the closed form.
        yobs = 3.0
        post_mean = yobs / 2
        post_sd = sqrt(0.5)
        cons = choicemap(:y => yobs, :z => 1.0)

        lj(mu) = logjoint(cdc_chain, [mu], (), cons)
        kernel(mu) = normlogpdf(post_mean, post_sd, mu)
        for mu in (-1.0, 0.0, post_mean, 1.7)
            @test (lj(mu) - lj(post_mean)) ≈ (kernel(mu) - kernel(post_mean)) atol = 1e-9
        end
    end

    @testset "unbound choice left unconstrained is a slotted latent" begin
        # `{:y} ~ normal(mu, 1)` with no constraint: under the conditioning rule
        # `:y` is a latent, gets a parameter slot, and its prior is scored.
        layout = UncertainTea._resolve_signature_plan(cdc_gaussian, choicemap(), ()).plan.parameter_layout
        @test parametervaluecount(layout) == 2
        @test Set(slot.binding for slot in layout.slots) == Set([:mu, Symbol("")])

        mu0 = 0.2
        y0 = 0.9
        lj = logjoint(cdc_gaussian, [mu0, y0], (), choicemap())
        @test lj ≈ normlogpdf(0.0, 1.0, mu0) + normlogpdf(mu0, 1.0, y0) atol = 1e-9
    end

    @testset "constrained vs unconstrained differ by the observation log-density" begin
        mu0 = 0.4
        v = 2.5
        # constrained: `:y` observed -> 1 latent (mu); unconstrained: `:y`
        # latent -> 2 latents (mu, y).
        con_layout = UncertainTea._resolve_signature_plan(cdc_gaussian, choicemap(:y => v), ()).plan.parameter_layout
        unc_layout = UncertainTea._resolve_signature_plan(cdc_gaussian, choicemap(), ()).plan.parameter_layout
        @test parametervaluecount(unc_layout) - parametervaluecount(con_layout) == 1

        l_con = logjoint(cdc_gaussian, [mu0], (), choicemap(:y => v))
        l_unc = logjoint(cdc_gaussian, [mu0, v], (), choicemap())
        y_term = normlogpdf(mu0, 1.0, v)
        # at matching values the two agree, and the y address contributes exactly
        # its log-density in both roles (observation vs prior draw).
        @test l_con ≈ l_unc atol = 1e-9
        @test (l_con - y_term) ≈ normlogpdf(0.0, 1.0, mu0) atol = 1e-9
    end

    @testset "signature memoization reuses the compiled plan" begin
        @tea static function cdc_memo()
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu, 1.0)
            return mu
        end

        @test isnothing(cdc_memo.signature_cache[])
        # Two runs at the same observed address but different data reuse the plan.
        logjoint(cdc_memo, [0.1], (), choicemap(:y => 1.0))
        first_resolved = UncertainTea._resolve_signature_plan(cdc_memo, choicemap(:y => 1.0), ())
        logjoint(cdc_memo, [0.1], (), choicemap(:y => 42.0))
        second_resolved = UncertainTea._resolve_signature_plan(cdc_memo, choicemap(:y => 42.0), ())
        @test first_resolved === second_resolved
        @test length(cdc_memo.signature_cache[]) == 1

        # A different signature (empty vs {:y}) gets its own cached entry.
        logjoint(cdc_memo, [0.1, 0.2], (), choicemap())
        @test length(cdc_memo.signature_cache[]) == 2
    end

    # PR-3: unified value resolution in the evaluator. The bound value of a
    # choice resolves to the constraint value when observed and to the
    # transformed parameter value when latent, through one path -- so a
    # downstream deterministic step / noncentered loc/scale expression sees the
    # right value regardless of classification.
    @testset "observed value used downstream in a deterministic step" begin
        mu0 = 0.4
        yv = 3.0
        zv = 0.5
        cons = choicemap(:y => yv, :z => zv)
        w = yv * 2.0 + mu0

        # scoring resolves the observed `:y` and feeds it to the deterministic
        # `w`; matches the manual density and `assess` on the same choices.
        manual = normlogpdf(0.0, 1.0, mu0) + normlogpdf(mu0, 1.0, yv) + normlogpdf(w, 1.0, zv)
        lj = logjoint(cdc_downstream, [mu0], (), cons)
        assessed = assess(cdc_downstream, (), choicemap(:mu => mu0, :y => yv, :z => zv))
        @test lj ≈ manual atol = 1e-9
        @test lj ≈ assessed atol = 1e-9
        # no dependent transform here, so the unconstrained path agrees exactly.
        @test logjoint_unconstrained(cdc_downstream, [mu0], (), cons) ≈ lj atol = 1e-9

        # the constrained observation genuinely reaches `w`: a different `:y`
        # value shifts the density (via the z term through w).
        @test logjoint(cdc_downstream, [mu0], (), choicemap(:y => yv, :z => zv)) !=
              logjoint(cdc_downstream, [mu0], (), choicemap(:y => yv + 1.0, :z => zv))

        # scalar gradient w.r.t. the latent `mu` is correct even though an
        # observed value feeds the downstream expression (it is a constant
        # w.r.t. the parameters). Analytic + finite-difference crosscheck.
        grad = logjoint_gradient_unconstrained(cdc_downstream, [mu0], (), cons)
        analytic = -mu0 + (yv - mu0) + (zv - w)
        f(m) = logjoint_unconstrained(cdc_downstream, [m], (), cons)
        h = 1e-6
        fd = (f(mu0 + h) - f(mu0 - h)) / (2h)
        @test grad[1] ≈ analytic atol = 1e-9
        @test grad[1] ≈ fd atol = 1e-5
    end

    @testset "constrained vs unconstrained with the address used downstream" begin
        mu0 = 0.4
        yv = 3.0
        zv = 0.5
        # `:y` observed -> latent {mu}; `:y` latent -> latents {mu, y}. With the
        # latent y set to the observed value the two agree, and `y` flows into
        # `w` in both roles (observation and prior draw).
        con_layout =
            UncertainTea._resolve_signature_plan(cdc_downstream, choicemap(:y => yv, :z => zv), ()).plan.parameter_layout
        unc_layout =
            UncertainTea._resolve_signature_plan(cdc_downstream, choicemap(:z => zv), ()).plan.parameter_layout
        @test parametervaluecount(con_layout) == 1
        @test parametervaluecount(unc_layout) == 2

        l_con = logjoint(cdc_downstream, [mu0], (), choicemap(:y => yv, :z => zv))
        l_unc = logjoint(cdc_downstream, [mu0, yv], (), choicemap(:z => zv))
        @test l_con ≈ l_unc atol = 1e-9
    end

    # PR-5: signature-aware API validation and observation/predictive alignment.
    @testset "length mismatch under a non-default signature names the conditioning" begin
        cons = choicemap(:y => 3.0, :z => 1.0)
        # signature {:y, :z}: one latent (mu), so a two-element vector is wrong.
        err = try
            logjoint(cdc_chain, [0.5, 2.0], (), cons)
            nothing
        catch e
            e
        end
        @test err isa DimensionMismatch
        msg = sprint(showerror, err)
        # the message names the observed addresses and the latent it leaves.
        @test occursin("expected 1", msg)
        @test occursin("observed", msg)
        @test occursin("(:y,)", msg)
        @test occursin("(:z,)", msg)
        @test occursin("mu", msg)
        @test occursin("constraint-driven-conditioning", msg)

        # the same signature-aware message is produced by the unconstrained-space
        # and transform entry points that take a raw parameter vector.
        for f in (
            () -> logjoint_unconstrained(cdc_chain, [0.5, 2.0], (), cons),
            () -> logjoint_gradient_unconstrained(cdc_chain, [0.5, 2.0], (), cons),
            () -> transform_to_constrained(cdc_chain, [0.5, 2.0], (), cons),
            () -> transform_to_unconstrained(cdc_chain, [0.5, 2.0], (), cons),
        )
            e = try
                f()
                nothing
            catch err
                err
            end
            @test e isa DimensionMismatch
            @test occursin("observed", sprint(showerror, e))
        end

        # a correctly-sized vector for this signature is accepted.
        @test isfinite(logjoint(cdc_chain, [0.5], (), cons))
        @test transform_to_constrained(cdc_chain, [0.5], (), cons) == [0.5]
    end

    @testset "observation_addresses returns exactly the constrained-present addresses" begin
        # cdc_chain has a bound `:y` and an unbound `:z`. Binding is orthogonal:
        # both are observations iff constrained.
        @test observation_addresses(cdc_chain, (), choicemap(:y => 1.0, :z => 2.0)) ==
              Any[(:y,), (:z,)]
        # only `:z` constrained -> `:y` becomes a latent and drops out.
        @test observation_addresses(cdc_chain, (), choicemap(:z => 2.0)) == Any[(:z,)]
        # only the bound `:y` constrained -> `:z` is unconstrained. `:z` is an
        # unbound latent under the rule (not a required observation), so the set
        # is exactly {:y}.
        @test observation_addresses(cdc_chain, (), choicemap(:y => 1.0)) == Any[(:y,)]
        # nothing constrained -> no observations.
        @test observation_addresses(cdc_chain, (), choicemap()) == Any[]
    end

    @testset "predictive classification is consistent with the signature" begin
        # cdc_gaussian: mu latent, `{:y}` observed. Inference conditions on `:y`;
        # the posterior-predictive re-samples exactly the observation addresses
        # returned by observation_addresses (the non-latent choices).
        cons = choicemap(:y => 0.5)
        chains = hmc_chains(
            cdc_gaussian,
            (),
            cons;
            num_chains=2,
            num_samples=40,
            num_warmup=40,
            rng=MersenneTwister(95),
        )
        pd = predict(cdc_gaussian, (), chains; num_draws=10, rng=MersenneTwister(1))
        @test Set(UncertainTea.addresses(pd)) ==
              Set(observation_addresses(cdc_gaussian, (), cons))
        @test Set(UncertainTea.addresses(pd)) == Set(Any[(:y,)])
    end

    # PR-6: the CPU inference entry points size and initialize from the same
    # conditioning signature, so a constraint on the bound `:y` (which made
    # hmc/nuts throw DimensionMismatch under the syntactic layout) now runs and
    # targets the conditioned posterior.
    @testset "#95 hmc/nuts condition on a bound address (closed-form posterior)" begin
        # mu ~ N(0,1), y | mu ~ N(mu,1) observed at yobs, z | y observed. The z
        # term is constant in mu (y is data), so the mu-posterior is the standard
        # conjugate normal N(yobs/2, 1/2).
        yobs = 3.0
        cons = choicemap(:y => yobs, :z => 1.0)
        post_mean = yobs / 2
        post_var = 0.5

        # signature {:y, :z}: exactly one latent (mu), so the sampler takes a
        # one-element state -- the count that made the old syntactic sizing throw.
        layout = UncertainTea._resolve_signature_plan(cdc_chain, cons, ()).plan.parameter_layout
        @test parametervaluecount(layout) == 1

        sample_mean(v) = sum(v) / length(v)
        sample_var(v) = sum(abs2, v .- sample_mean(v)) / (length(v) - 1)

        for sampler in (hmc, nuts)
            res = first(sampler(cdc_chain, (), cons; num_samples=4000, num_warmup=1000, rng=MersenneTwister(95)))
            @test size(res.constrained_samples, 1) == 1
            mus = vec(res.constrained_samples[1, :])
            # posterior mean / variance match the conjugate closed form within
            # Monte Carlo error (not merely cross-path agreement).
            @test sample_mean(mus) ≈ post_mean atol = 0.1
            @test sample_var(mus) ≈ post_var atol = 0.1
        end
    end

    @testset "#95 same model with :y unconstrained samples the larger latent set" begin
        # Only `:z` observed: `:y` is now a latent, so [mu, y] are both sampled.
        cons = choicemap(:z => 0.5)
        layout = UncertainTea._resolve_signature_plan(cdc_chain, cons, ()).plan.parameter_layout
        @test parametervaluecount(layout) == 2

        res = first(nuts(cdc_chain, (), cons; num_samples=800, num_warmup=800, rng=MersenneTwister(7)))
        @test size(res.constrained_samples, 1) == 2
        # every draw is finite in both latent coordinates.
        @test all(isfinite, res.constrained_samples)
    end

    @testset "#95 smc/sir condition on a bound address (closed-form posterior)" begin
        # Same conjugate setup as the HMC/NUTS acceptance: signature {:y, :z}
        # leaves exactly one latent (mu) with posterior N(yobs/2, 1/2). Before
        # PR-7 the SMC path sized from the syntactic default layout ([mu, y, z])
        # and threw a DimensionMismatch against the signature-sized logjoint; now
        # it sizes particles, proposal, tempering, and result from the signature.
        yobs = 3.0
        cons = choicemap(:y => yobs, :z => 1.0)
        post_mean = yobs / 2
        post_var = 0.5

        layout = UncertainTea._conditioned_parameter_layout(cdc_chain, cons, ())
        @test parametervaluecount(layout) == 1

        wmean(v, w) = sum(v .* w)
        wvar(v, w, mn) = sum(w .* abs2.(v .- mn))

        smc = batched_smc(
            cdc_chain,
            (),
            cons;
            num_particles=6000,
            move_kernel=:random_walk,
            move_steps=4,
            rng=MersenneTwister(95),
        )
        @test size(smc.importance.constrained_particles, 1) == 1
        smc_mu = vec(smc.importance.constrained_particles[1, :])
        smc_w = smc.importance.normalized_weights
        smc_mean = wmean(smc_mu, smc_w)
        # posterior mean / variance match the conjugate closed form within Monte
        # Carlo error (not merely cross-path agreement).
        @test smc_mean ≈ post_mean atol = 0.1
        @test wvar(smc_mu, smc_w, smc_mean) ≈ post_var atol = 0.12

        # SIR shares the smc/api.jl sizing; a proposal centered at the posterior
        # keeps the importance estimator low-variance for the closed-form check.
        sir = batched_sir(
            cdc_chain,
            (),
            cons;
            num_particles=20000,
            proposal_loc=[post_mean],
            rng=MersenneTwister(95),
        )
        @test size(sir.importance.constrained_particles, 1) == 1
        sir_mu = vec(sir.importance.constrained_particles[1, :])
        sir_w = sir.importance.normalized_weights
        @test wmean(sir_mu, sir_w) ≈ post_mean atol = 0.1
    end

    @testset "#95 smc with :y unconstrained samples the larger latent set" begin
        # Only `:z` observed: `:y` becomes a latent, so SMC sizes [mu, y].
        cons = choicemap(:z => 0.5)
        layout = UncertainTea._conditioned_parameter_layout(cdc_chain, cons, ())
        @test parametervaluecount(layout) == 2

        smc = batched_smc(cdc_chain, (), cons; num_particles=2000, rng=MersenneTwister(7))
        @test size(smc.importance.constrained_particles, 1) == 2
        @test all(isfinite, smc.importance.constrained_particles)
    end

    @testset "#95 batched hmc/nuts condition on a bound address (closed-form posterior)" begin
        # The multichain batched samplers sized their position and constrained
        # buffers from the syntactic default layout ([mu, y, z]) and threw a
        # DimensionMismatch against the signature-aware batched workspaces; now
        # they size and reconstruct from the conditioning signature {:y, :z},
        # leaving the single latent mu with posterior N(yobs/2, 1/2).
        yobs = 3.0
        cons = choicemap(:y => yobs, :z => 1.0)
        post_mean = yobs / 2
        post_var = 0.5
        sample_mean(v) = sum(v) / length(v)
        sample_var(v) = sum(abs2, v .- sample_mean(v)) / (length(v) - 1)

        for sampler in (batched_hmc, batched_nuts)
            res = sampler(cdc_chain, (), cons; num_chains=4, num_samples=1500, num_warmup=1000, rng=MersenneTwister(95))
            @test size(res.chains[1].constrained_samples, 1) == 1
            mus = reduce(vcat, [vec(chain.constrained_samples[1, :]) for chain in res.chains])
            @test sample_mean(mus) ≈ post_mean atol = 0.1
            @test sample_var(mus) ≈ post_var atol = 0.1
        end

        # same model, :y unconstrained -> larger [mu, y] latent set.
        cons_wide = choicemap(:z => 0.5)
        wide = batched_nuts(cdc_chain, (), cons_wide; num_chains=2, num_samples=400, num_warmup=400, rng=MersenneTwister(7))
        @test size(wide.chains[1].constrained_samples, 1) == 2
        @test all(isfinite, wide.chains[1].constrained_samples)
    end

    @testset "observed value feeds a reparam=:noncentered loc/scale" begin
        mu0 = 0.4
        yv = 3.0
        zv = 0.5
        zstd = 0.1
        cons = choicemap(:y => yv, :z => zv)
        theta = yv + zstd  # theta = location(y) + scale(1) * z

        # Before PR-3 the transform walk poisoned the slotless observed `:y`
        # feeding theta's location and this raised; now it resolves to yv.
        lj_nc = logjoint_unconstrained(cdc_noncentered_obs, [mu0, zstd], (), cons)
        manual =
            normlogpdf(0.0, 1.0, mu0) +
            normlogpdf(mu0, 1.0, yv) +
            normlogpdf(yv, 1.0, theta) +
            normlogpdf(theta, 1.0, zv)
        @test lj_nc ≈ manual atol = 1e-9

        # centered twin at the matching constrained theta gives the same joint
        # (scale 1 -> zero log-abs-det), confirming the noncentered walk used the
        # observed location.
        lj_centered = logjoint_unconstrained(cdc_centered_obs, [mu0, theta], (), cons)
        @test lj_nc ≈ lj_centered atol = 1e-9

        # gradient w.r.t. [mu, z] crosschecks against finite differences with the
        # observed location flowing into theta.
        grad = logjoint_gradient_unconstrained(cdc_noncentered_obs, [mu0, zstd], (), cons)
        g(p) = logjoint_unconstrained(cdc_noncentered_obs, p, (), cons)
        h = 1e-6
        fd = [
            (g([mu0 + h, zstd]) - g([mu0 - h, zstd])) / (2h),
            (g([mu0, zstd + h]) - g([mu0, zstd - h])) / (2h),
        ]
        @test grad ≈ fd atol = 1e-5
    end
end
