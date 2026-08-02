# issue #88: correct MARGINAL pointwise log-likelihood for models with a
# marginalize=:enumerate discrete latent. An enumerated latent is never an
# observation; the per-observation likelihood integrates it out with the same
# logsumexp the marginalized logjoint uses.
#
# Supported (factorizing) structures and the rejected non-factorizing ones are
# documented in `_record_marginalized_choice!` (src/evaluator_pointwise.jl) and
# docs/discrete-enumeration.md. This suite pins:
#   * observation_addresses excludes the enumerated latent;
#   * the bernoulli repro and a categorical variant match a hand logsumexp oracle;
#   * a z-independent extra observation factorizes out cleanly;
#   * WAIC / PSIS-LOO on an enumerate model agree with an independent oracle matrix;
#   * shared-latent (multiple z-dependent observations), free-latent-in-suffix, and
#     nested-marginalize suffixes raise the documented errors.

pme_logsumexp = function (v)
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

# the issue #88 repro
@tea static function pme_bernoulli_model()
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y} ~ normal(mu + z, 1.0)
    return mu
end

# a categorical-enumerate variant
@tea static function pme_categorical_model()
    mu ~ normal(0.0, 1.0)
    z ~ categorical([0.2, 0.3, 0.5]; marginalize=:enumerate)
    {:y} ~ normal(mu * z, 0.4)
    return mu
end

# one z-dependent observation plus one z-independent observation: the marginal
# factorizes (the independent factor comes out of the enumeration sum)
@tea static function pme_mixed_model()
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(0.4; marginalize=:enumerate)
    {:y1} ~ normal(mu + z, 1.0)
    {:y2} ~ normal(mu, 1.0)
    return mu
end

# two z-dependent observations sharing the enumerated latent: non-factorizing
@tea static function pme_shared_model()
    mu ~ normal(0.0, 1.0)
    z ~ bernoulli(0.3; marginalize=:enumerate)
    {:y1} ~ normal(mu + z, 1.0)
    {:y2} ~ normal(mu - z, 1.0)
    return mu
end

# a free (slotted) latent inside the enumeration suffix: rejected
@tea static function pme_free_latent_suffix_model()
    z ~ bernoulli(0.3; marginalize=:enumerate)
    mu ~ normal(1.0 * z, 1.0)
    {:y} ~ normal(mu, 1.0)
    return mu
end

# two nested enumerated latents both feeding a SINGLE observation: supported (the one
# column gets its full double-sum marginal, which factorizes trivially over one point)
@tea static function pme_nested_single_obs_model()
    mu ~ normal(0.0, 1.0)
    a ~ bernoulli(0.4; marginalize=:enumerate)
    b ~ bernoulli(0.7; marginalize=:enumerate)
    {:y} ~ normal(mu + a + 2 * b, 0.5)
    return mu
end

# independent enumerated latents each local to their own observation: supported (the
# per-column marginals decompose because the observations are conditionally independent)
@tea static function pme_independent_enum_model()
    mu ~ normal(0.0, 1.0)
    z1 ~ bernoulli(0.3; marginalize=:enumerate)
    {:y1} ~ normal(mu + z1, 1.0)
    z2 ~ bernoulli(0.6; marginalize=:enumerate)
    {:y2} ~ normal(mu + z2, 1.0)
    return mu
end

# pointwise_loglikelihood accepts a raw constrained-draws matrix (columns are
# draws), so the matrix can be checked against a hand oracle on chosen
# parameter draws without depending on sampler noise (issue #337: the old
# duck-typed chains argument became a typed interface + matrix method)

@testset "pointwise_marginal_enum" begin
    @testset "pme_bernoulli_oracle" begin
        pme_cm = choicemap((:y, 0.4))
        @test observation_addresses(pme_bernoulli_model, (), pme_cm) == Any[(:y,)]

        pme_mus = [0.1, -0.5, 1.2, 0.0]
        pme_chains = reshape(pme_mus, 1, length(pme_mus))
        pme_ll = pointwise_loglikelihood(pme_bernoulli_model, (), pme_cm, pme_chains)
        @test size(pme_ll) == (length(pme_mus), 1)
        for (j, mu) in enumerate(pme_mus)
            pme_terms = [
                log(0.7) + UncertainTea.logpdf(normal(mu, 1.0), 0.4),
                log(0.3) + UncertainTea.logpdf(normal(mu + 1.0, 1.0), 0.4),
            ]
            @test pme_ll[j, 1] ≈ pme_logsumexp(pme_terms) atol = 1e-12
        end
    end

    @testset "pme_categorical_oracle" begin
        pme_cm = choicemap((:y, 1.1))
        @test observation_addresses(pme_categorical_model, (), pme_cm) == Any[(:y,)]

        pme_mus = [0.6, -0.3, 1.5]
        pme_chains = reshape(pme_mus, 1, length(pme_mus))
        pme_ll = pointwise_loglikelihood(pme_categorical_model, (), pme_cm, pme_chains)
        for (j, mu) in enumerate(pme_mus)
            pme_terms = [
                log(w) + UncertainTea.logpdf(normal(mu * k, 0.4), 1.1) for
                (k, w) in enumerate([0.2, 0.3, 0.5])
            ]
            @test pme_ll[j, 1] ≈ pme_logsumexp(pme_terms) atol = 1e-12
        end
    end

    @testset "pme_mixed_factorizes" begin
        pme_cm = choicemap((:y1, 0.2), (:y2, -0.3))
        @test observation_addresses(pme_mixed_model, (), pme_cm) == Any[(:y1,), (:y2,)]

        pme_mus = [0.5, -0.4]
        pme_chains = reshape(pme_mus, 1, length(pme_mus))
        pme_ll = pointwise_loglikelihood(pme_mixed_model, (), pme_cm, pme_chains)
        for (j, mu) in enumerate(pme_mus)
            pme_y1_terms = [
                log(0.6) + UncertainTea.logpdf(normal(mu, 1.0), 0.2),
                log(0.4) + UncertainTea.logpdf(normal(mu + 1.0, 1.0), 0.2),
            ]
            @test pme_ll[j, 1] ≈ pme_logsumexp(pme_y1_terms) atol = 1e-12
            # y2 does not depend on z, so it is its own logpdf
            @test pme_ll[j, 2] ≈ UncertainTea.logpdf(normal(mu, 1.0), -0.3) atol = 1e-12
        end
    end

    @testset "pme_nonfactorizing_rejections" begin
        pme_shared_cm = choicemap((:y1, 0.2), (:y2, -0.3))
        pme_shared_err = try
            observation_addresses(pme_shared_model, (), pme_shared_cm)
            nothing
        catch err
            err
        end
        @test pme_shared_err isa ArgumentError
        @test occursin("shared across multiple observations", sprint(showerror, pme_shared_err))

        pme_free_cm = choicemap((:y, 0.3))
        pme_free_err = try
            observation_addresses(pme_free_latent_suffix_model, (), pme_free_cm)
            nothing
        catch err
            err
        end
        @test pme_free_err isa ArgumentError
        @test occursin("free (slotted) latent", sprint(showerror, pme_free_err))
    end

    @testset "pme_nested_supported" begin
        # nested enumerated latents feeding a single observation: the column gets the
        # full multi-sum marginal, which equals logjoint minus the mu prior term
        pme_cm = choicemap((:y, 2.2))
        @test observation_addresses(pme_nested_single_obs_model, (), pme_cm) == Any[(:y,)]
        pme_mus = [0.3, -0.6]
        pme_chains = reshape(pme_mus, 1, length(pme_mus))
        pme_ll = pointwise_loglikelihood(pme_nested_single_obs_model, (), pme_cm, pme_chains)
        for (j, mu) in enumerate(pme_mus)
            pme_terms = [
                log(a == 1 ? 0.4 : 0.6) + log(b == 1 ? 0.7 : 0.3) +
                UncertainTea.logpdf(normal(mu + a + 2 * b, 0.5), 2.2) for a = 0:1, b = 0:1
            ]
            @test pme_ll[j, 1] ≈ pme_logsumexp(vec(pme_terms)) atol = 1e-12
            # and equals the full marginal logjoint minus the mu prior
            pme_lj = logjoint(pme_nested_single_obs_model, [mu], (), pme_cm)
            pme_prior = UncertainTea.logpdf(normal(0.0, 1.0), mu)
            @test pme_ll[j, 1] ≈ pme_lj - pme_prior atol = 1e-10
        end

        # independent enumerated latents feeding separate observations: per-column
        # marginals, and their sum equals logjoint minus the prior
        pme_ind_cm = choicemap((:y1, 0.5), (:y2, -0.2))
        @test observation_addresses(pme_independent_enum_model, (), pme_ind_cm) ==
              Any[(:y1,), (:y2,)]
        pme_ind_chains = reshape([0.4, -0.3], 1, 2)
        pme_ind_ll = pointwise_loglikelihood(pme_independent_enum_model, (), pme_ind_cm, pme_ind_chains)
        for (j, mu) in enumerate([0.4, -0.3])
            pme_o1 = pme_logsumexp([
                log(0.7) + UncertainTea.logpdf(normal(mu, 1.0), 0.5),
                log(0.3) + UncertainTea.logpdf(normal(mu + 1.0, 1.0), 0.5),
            ])
            pme_o2 = pme_logsumexp([
                log(0.4) + UncertainTea.logpdf(normal(mu, 1.0), -0.2),
                log(0.6) + UncertainTea.logpdf(normal(mu + 1.0, 1.0), -0.2),
            ])
            @test pme_ind_ll[j, 1] ≈ pme_o1 atol = 1e-12
            @test pme_ind_ll[j, 2] ≈ pme_o2 atol = 1e-12
            pme_prior = UncertainTea.logpdf(normal(0.0, 1.0), mu)
            @test sum(pme_ind_ll[j, :]) ≈
                  logjoint(pme_independent_enum_model, [mu], (), pme_ind_cm) - pme_prior atol = 1e-10
        end
    end

    @testset "pme_waic_psis_loo" begin
        # A model with two independent enumerated indicators, each local to its own
        # observation, so both columns factorize. WAIC/PSIS-LOO on the pointwise matrix
        # from real NUTS draws must equal the same statistics computed from an
        # independent hand-built oracle matrix (checking the pointwise matrix itself and
        # that the model-comparison estimators consume it correctly).
        pme_waic_cm = choicemap((:y1, 0.4), (:y2, -0.7))
        pme_waic_addr = observation_addresses(pme_independent_enum_model, (), pme_waic_cm)
        @test pme_waic_addr == Any[(:y1,), (:y2,)]

        pme_waic_chains = nuts_chains(
            pme_independent_enum_model,
            (),
            pme_waic_cm;
            num_chains=2,
            num_samples=200,
            num_warmup=200,
            rng=MersenneTwister(88),
        )
        pme_ll = pointwise_loglikelihood(pme_independent_enum_model, (), pme_waic_cm, pme_waic_chains)
        @test size(pme_ll, 2) == 2

        # independent oracle: rebuild the marginal pointwise matrix by hand from the
        # pooled mu draws.
        pme_pooled = reduce(hcat, [c.constrained_samples for c in pme_waic_chains.chains])
        @test size(pme_pooled, 2) == size(pme_ll, 1)
        pme_ys = [0.4, -0.7]
        pme_ps = [0.3, 0.6]
        pme_oracle = Matrix{Float64}(undef, size(pme_ll)...)
        for s = 1:size(pme_pooled, 2)
            mu = pme_pooled[1, s]
            for i = 1:2
                pme_terms = [
                    log(1 - pme_ps[i]) + UncertainTea.logpdf(normal(mu, 1.0), pme_ys[i]),
                    log(pme_ps[i]) + UncertainTea.logpdf(normal(mu + 1.0, 1.0), pme_ys[i]),
                ]
                pme_oracle[s, i] = pme_logsumexp(pme_terms)
            end
        end
        @test maximum(abs.(pme_ll .- pme_oracle)) < 1e-10

        pme_waic_from_ll = waic(pme_ll)
        pme_waic_from_oracle = waic(pme_oracle)
        @test pme_waic_from_ll.elpd ≈ pme_waic_from_oracle.elpd atol = 1e-10
        @test pme_waic_from_ll.p_eff ≈ pme_waic_from_oracle.p_eff atol = 1e-10

        pme_loo_from_ll = psis_loo(pme_ll)
        pme_loo_from_oracle = psis_loo(pme_oracle)
        @test pme_loo_from_ll.elpd ≈ pme_loo_from_oracle.elpd atol = 1e-10
        @test pme_loo_from_ll.pareto_k ≈ pme_loo_from_oracle.pareto_k atol = 1e-10

        # convenience wrappers route through pointwise_loglikelihood
        pme_waic_conv = waic(pme_independent_enum_model, (), pme_waic_cm, pme_waic_chains)
        pme_loo_conv = loo(pme_independent_enum_model, (), pme_waic_cm, pme_waic_chains)
        @test pme_waic_conv.elpd ≈ pme_waic_from_ll.elpd atol = 1e-10
        @test pme_loo_conv.elpd ≈ pme_loo_from_ll.elpd atol = 1e-10
    end
end
