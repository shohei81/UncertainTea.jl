# Result-type composability (issue #337). Contracts:
#   * single-chain `nuts` returns a one-chain HMCChains on which the full
#     diagnostics surface works (summarize / rhat / ess / posterior_array /
#     to_arviz_dict / predict / loo);
#   * GibbsChain gets the same continuous-block surface, with the discrete part
#     staying on discrete_ess / the chain fields;
#   * every inference result implements the posterior-draws interface
#     `constrained_draws(result) -> (num_params x num_draws matrix, names)`;
#   * `predict` / `loo` route through the interface and reject non-result
#     arguments with a MethodError (not a downstream field error).

@tea static function ri_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    for i = 1:n
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea function ri_gibbs_model(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        z = ({:z => i} ~ bernoulli(0.3))
        {:y => i} ~ normal(mu + 2.0 * z, 1.0)
    end
    return mu
end

@testset "result_interface" begin
    ri_n = 5
    ri_cm = choicemap((:y => i, 0.5 + 0.1 * i) for i = 1:ri_n)
    ri_args = (ri_n,)

    @testset "ri_single_chain_nuts" begin
        ri_res = nuts(
            ri_gauss_model,
            ri_args,
            ri_cm;
            num_samples=60,
            num_warmup=60,
            rng=MersenneTwister(37),
        )
        @test ri_res isa HMCChains
        @test length(ri_res) == 1
        @test first(ri_res) isa HMCChain

        ri_summary = summarize(ri_res)
        @test length(ri_summary.parameters) == 2
        @test all(isfinite, [p.mean for p in ri_summary.parameters])

        ri_rhat = rhat(ri_res)
        ri_ess = ess(ri_res)
        @test length(ri_rhat) == 2 && all(isfinite, ri_rhat)
        @test length(ri_ess) == 2 && all(v -> isfinite(v) && v > 0, ri_ess)

        @test size(posterior_array(ri_res)) == (60, 1, 2)
        ri_arviz = to_arviz_dict(ri_res)
        @test Set(keys(ri_arviz["posterior"])) == Set(["mu", "sigma"])

        ri_pred = predict(ri_gauss_model, ri_args, ri_res; num_draws=10, rng=MersenneTwister(3))
        @test length(ri_pred) == 10
        @test length(ri_pred[:y=>1]) == 10

        ri_loo = loo(ri_gauss_model, ri_args, ri_cm, ri_res)
        @test length(ri_loo.pointwise) == ri_n
        @test isfinite(ri_loo.elpd)
        ri_waic = waic(ri_gauss_model, ri_args, ri_cm, ri_res)
        @test isfinite(ri_waic.elpd)

        ri_draws, ri_names = constrained_draws(ri_res)
        @test size(ri_draws) == (2, 60)
        @test ri_names == ["mu", "sigma"]
        @test ri_draws == first(ri_res).constrained_samples
        # evenly-spread subset selection
        ri_sub, _ = constrained_draws(ri_res; num_draws=10)
        @test size(ri_sub, 2) == 10
    end

    @testset "ri_gibbs_chain" begin
        ri_gc = gibbs(
            ri_gibbs_model,
            ri_args,
            ri_cm;
            num_samples=60,
            num_warmup=40,
            rng=MersenneTwister(11),
        )
        @test ri_gc isa GibbsChain

        ri_summary = summarize(ri_gc)
        @test length(ri_summary.parameters) == 1
        @test ri_summary.parameters[1].binding === :mu

        @test length(rhat(ri_gc)) == 1
        @test all(v -> isfinite(v) && v > 0, ess(ri_gc))
        @test size(posterior_array(ri_gc)) == (60, 1, 1)
        @test parameter_names(ri_gc) == ["mu"]

        ri_draws, ri_names = constrained_draws(ri_gc)
        @test ri_draws == ri_gc.constrained_samples
        @test ri_names == ["mu"]

        # predict pins the sampled discrete values, so only the observation
        # addresses are re-drawn
        ri_pred = predict(ri_gibbs_model, ri_args, ri_gc; num_draws=8, rng=MersenneTwister(5))
        @test length(ri_pred) == 8
        ri_pred_addresses = Set(map(a -> a, addresses(ri_pred)))
        @test ri_pred_addresses == Set(Any[(:y, i) for i = 1:ri_n])

        # loo conditions each draw's pointwise walk on that draw's discrete
        # values: one column per USER observation
        ri_loo = loo(ri_gibbs_model, ri_args, ri_cm, ri_gc)
        @test length(ri_loo.pointwise) == ri_n
        @test isfinite(ri_loo.elpd)

        # pointwise oracle: p(y_i | mu_s, z_s) at a few draws
        ri_ll = pointwise_loglikelihood(ri_gibbs_model, ri_args, ri_cm, ri_gc)
        @test size(ri_ll) == (60, ri_n)
        for ri_s in (1, 30, 60), ri_i in (1, ri_n)
            ri_mu = ri_gc.constrained_samples[1, ri_s]
            ri_site = findfirst(==((:z, ri_i)), ri_gc.discrete_addresses)
            ri_z = ri_gc.discrete_samples[ri_site, ri_s]
            ri_expected = UncertainTea.logpdf(normal(ri_mu + 2.0 * ri_z, 1.0), 0.5 + 0.1 * ri_i)
            @test ri_ll[ri_s, ri_i] ≈ ri_expected atol = 1e-10
        end

        # the discrete part keeps its dedicated surface
        @test length(discrete_ess(ri_gc)) == ri_n
    end

    @testset "ri_constrained_draws_all_results" begin
        ri_rng = MersenneTwister(101)

        ri_check = function (result; num_params=2, names=["mu", "sigma"])
            ri_d, ri_nm = constrained_draws(result)
            @test ri_d isa AbstractMatrix
            @test size(ri_d, 1) == num_params
            @test size(ri_d, 2) >= 1
            @test all(isfinite, ri_d)
            @test ri_nm == names
            # sigma rows are constrained-positive
            num_params == 2 && @test all(>(0.0), view(ri_d, 2, :))
            return ri_d
        end

        ri_check(nuts(ri_gauss_model, ri_args, ri_cm; num_samples=20, num_warmup=20, rng=ri_rng))
        ri_check(hmc(ri_gauss_model, ri_args, ri_cm; num_samples=20, num_warmup=20, rng=ri_rng))

        ri_advi = batched_advi(ri_gauss_model, ri_args, ri_cm; num_iterations=60, num_particles=8, rng=ri_rng)
        ri_advi_draws, _ = constrained_draws(ri_advi; num_draws=30, rng=ri_rng)
        @test size(ri_advi_draws, 2) == 30
        ri_check(ri_advi)

        ri_svgd = batched_svgd(ri_gauss_model, ri_args, ri_cm; num_iterations=30, num_particles=12, rng=ri_rng)
        @test size(ri_check(ri_svgd), 2) == 12

        ri_is = batched_importance_sampling(ri_gauss_model, ri_args, ri_cm; num_particles=32, rng=ri_rng)
        @test size(ri_check(ri_is), 2) == 32

        ri_sir = batched_sir(ri_gauss_model, ri_args, ri_cm; num_particles=32, num_samples=16, rng=ri_rng)
        @test size(ri_check(ri_sir), 2) == 16

        ri_smc = batched_smc(ri_gauss_model, ri_args, ri_cm; num_particles=32, rng=ri_rng)
        @test size(ri_check(ri_smc), 2) == 32

        ri_ns = nested_sampling(ri_gauss_model, ri_args, ri_cm; num_live_points=40, rng=ri_rng)
        ri_check(ri_ns)

        ri_la = laplace_approximation(ri_gauss_model, ri_args, ri_cm; rng=ri_rng)
        ri_la_draws, _ = constrained_draws(ri_la; num_draws=25, rng=ri_rng)
        @test size(ri_la_draws, 2) == 25
        ri_check(ri_la)

        ri_map = map_estimate(ri_gauss_model, ri_args, ri_cm; rng=ri_rng)
        ri_map_draws = ri_check(ri_map)
        @test size(ri_map_draws, 2) == 1
        @test vec(ri_map_draws) ≈ ri_map.constrained_mode

        # elliptical slice: no model attached, generic coordinate names
        ri_es_L = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        ri_es = elliptical_slice(
            f -> -0.5 * sum(abs2, f .- 0.3),
            ri_es_L;
            num_samples=15,
            rng=ri_rng,
        )
        ri_es_draws, ri_es_names = constrained_draws(ri_es)
        @test size(ri_es_draws) == (3, 15)
        @test ri_es_names == ["f[1]", "f[2]", "f[3]"]
        @test ri_es_draws == ri_es.samples

        # the interface powers predict/loo for the approximate results too
        @test length(predict(ri_gauss_model, ri_args, ri_advi; num_draws=5, rng=ri_rng)) == 5
        @test isfinite(loo(ri_gauss_model, ri_args, ri_cm, ri_la; num_draws=50, rng=ri_rng).elpd)
        @test isfinite(loo(ri_gauss_model, ri_args, ri_cm, ri_is).elpd)
    end

    @testset "ri_pathfinder_constrained_accessor" begin
        ri_pf = pathfinder(ri_gauss_model, ri_args, ri_cm; rng=MersenneTwister(7))
        ri_pf_draws, ri_pf_names = constrained_draws(ri_pf)
        @test ri_pf_names == ["mu", "sigma"]
        @test size(ri_pf_draws) == (2, size(ri_pf.draws, 2))
        # round-trip: each constrained column is the transform of the stored
        # unconstrained draw
        for ri_j in (1, 2, size(ri_pf.draws, 2))
            ri_expected = transform_to_constrained(
                ri_gauss_model,
                collect(ri_pf.draws[:, ri_j]),
                ri_args,
                ri_cm,
            )
            @test ri_pf_draws[:, ri_j] ≈ ri_expected atol = 1e-12
        end
        @test all(>(0.0), view(ri_pf_draws, 2, :))
        @test length(predict(ri_gauss_model, ri_args, ri_pf; num_draws=4, rng=MersenneTwister(2))) == 4
    end

    @testset "ri_wrong_input_dies_with_methoderror" begin
        @test_throws MethodError loo(ri_gauss_model, ri_args, ri_cm, "not a result")
        @test_throws MethodError loo(ri_gauss_model, ri_args, ri_cm, (; chains=[1, 2]))
        @test_throws MethodError predict(ri_gauss_model, ri_args, 42)
        @test_throws MethodError constrained_draws("not a result")
        @test_throws MethodError pointwise_loglikelihood(ri_gauss_model, ri_args, ri_cm, 1.0)
    end
end
