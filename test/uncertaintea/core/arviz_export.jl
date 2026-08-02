# ArviZ export (issue #339): the log_likelihood group, the layout kwarg
# (:draw_chain vs the Python :chain_draw convention), the attrs entry, the
# step_size/n_steps sample stats, and the loo/predict arity alignment.

@tea static function avz_model(n)
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    for i = 1:n
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@testset "arviz_export" begin
    avz_n = 4
    avz_cm = choicemap((:y => i, 0.4 + 0.2 * i) for i = 1:avz_n)
    avz_args = (avz_n,)
    avz_S = 30
    avz_C = 2
    avz_chains = nuts_chains(
        avz_model,
        avz_args,
        avz_cm;
        num_chains=avz_C,
        num_samples=avz_S,
        num_warmup=40,
        rng=MersenneTwister(339),
    )

    @testset "avz_log_likelihood_group" begin
        avz_dict = to_arviz_dict(avz_model, avz_args, avz_cm, avz_chains)
        @test haskey(avz_dict, "log_likelihood")
        avz_ll_group = avz_dict["log_likelihood"]
        @test Set(keys(avz_ll_group)) == Set(["y[$i]" for i = 1:avz_n])

        # Values match pointwise_loglikelihood, whose rows pool the chains in
        # order (chain 1's draws first), i.e. the posterior-group shape.
        avz_ll = pointwise_loglikelihood(avz_model, avz_args, avz_cm, avz_chains)
        @test size(avz_ll) == (avz_S * avz_C, avz_n)
        avz_addresses = observation_addresses(avz_model, avz_args, avz_cm)
        @test length(avz_addresses) == avz_n
        for (avz_col, avz_address) in enumerate(avz_addresses)
            avz_key = string(avz_address[1], "[", avz_address[2], "]")
            avz_matrix = avz_ll_group[avz_key]
            @test size(avz_matrix) == (avz_S, avz_C)
            @test avz_matrix == reshape(avz_ll[:, avz_col], avz_S, avz_C)
        end

        # The chains-only method keeps its shape and gains no group.
        avz_plain = to_arviz_dict(avz_chains)
        @test !haskey(avz_plain, "log_likelihood")
        @test avz_plain["posterior"] == avz_dict["posterior"]
    end

    @testset "avz_layouts_and_attrs" begin
        avz_dc = to_arviz_dict(avz_model, avz_args, avz_cm, avz_chains)
        avz_cd = to_arviz_dict(avz_model, avz_args, avz_cm, avz_chains; layout=:chain_draw)

        @test avz_dc["attrs"]["layout"] == "draw_chain"
        @test avz_cd["attrs"]["layout"] == "chain_draw"
        @test avz_dc["attrs"]["inference_library"] == "UncertainTea"
        @test haskey(avz_dc["attrs"], "inference_library_version")

        # Round-trip shape check: every group matrix is the exact transpose.
        for avz_group in ("posterior", "sample_stats", "log_likelihood")
            @test Set(keys(avz_dc[avz_group])) == Set(keys(avz_cd[avz_group]))
            for avz_key in keys(avz_dc[avz_group])
                avz_a = avz_dc[avz_group][avz_key]
                avz_b = avz_cd[avz_group][avz_key]
                @test size(avz_a) == (avz_S, avz_C)
                @test size(avz_b) == (avz_C, avz_S)
                @test avz_b == permutedims(avz_a)
            end
        end

        @test_throws ArgumentError to_arviz_dict(avz_chains; layout=:draws_first)
    end

    @testset "avz_sample_stats_step_size_n_steps" begin
        avz_stats = to_arviz_dict(avz_chains)["sample_stats"]
        @test haskey(avz_stats, "step_size")
        @test haskey(avz_stats, "n_steps")
        @test size(avz_stats["step_size"]) == (avz_S, avz_C)
        @test size(avz_stats["n_steps"]) == (avz_S, avz_C)
        for (avz_ci, avz_chain) in enumerate(avz_chains)
            @test all(==(avz_chain.step_size), avz_stats["step_size"][:, avz_ci])
            @test avz_stats["n_steps"][:, avz_ci] == avz_chain.integration_steps
        end
        @test all(>(0), avz_stats["n_steps"])
    end

    @testset "avz_loo_predict_arity_alignment" begin
        # One argument list drives loo, predict, and the arviz export.
        avz_loo = loo(avz_model, avz_args, avz_cm, avz_chains)
        @test isfinite(avz_loo.elpd)
        @test length(avz_loo.pointwise) == avz_n

        avz_pred = predict(
            avz_model, avz_args, avz_cm, avz_chains;
            num_draws=6, rng=MersenneTwister(7),
        )
        @test length(avz_pred) == 6
        @test Set(addresses(avz_pred)) == Set(Any[(:y, i) for i = 1:avz_n])

        # The constraints-accepting form matches the three-argument form draw
        # for draw under the same RNG (constraint values are unused).
        avz_pred3 = predict(
            avz_model, avz_args, avz_chains;
            num_draws=6, rng=MersenneTwister(7),
        )
        @test [avz_pred[(:y, i)] for i = 1:avz_n] == [avz_pred3[(:y, i)] for i = 1:avz_n]

        # Wrong result argument stays a MethodError on the aligned arity.
        @test_throws MethodError predict(avz_model, avz_args, avz_cm, "not a result")
    end
end
