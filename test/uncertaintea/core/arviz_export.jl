# ArviZ export (issue #339): the log_likelihood group, the layout kwarg
# (:draw_chain vs the Python :chain_draw convention), the attrs entry, the
# step_size/n_steps sample stats, and the loo/predict arity alignment.
# Issue #366: flatten_vectors=false groups vector parameters/observations into
# single arrays with a trailing coord dim plus coords/dims metadata.

@tea static function avz_model(n)
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 0.5)
    for i = 1:n
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea static function avz_vec_model(k)
    mu ~ normal(0.0, 1.0)
    theta ~ iid(normal(mu, 1.0), 3)
    for i = 1:k
        {:y => i} ~ normal(theta[i], 1.0)
    end
    return theta
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

    @testset "avz_flatten_vectors_false" begin
        avz_vk = 3
        avz_vcm = choicemap((:y => i, 0.3 * i) for i = 1:avz_vk)
        avz_vargs = (avz_vk,)
        avz_vchains = nuts_chains(
            avz_vec_model,
            avz_vargs,
            avz_vcm;
            num_chains=avz_C,
            num_samples=avz_S,
            num_warmup=40,
            rng=MersenneTwister(366),
        )

        avz_flat = to_arviz_dict(avz_vec_model, avz_vargs, avz_vcm, avz_vchains)
        avz_gdc = to_arviz_dict(
            avz_vec_model, avz_vargs, avz_vcm, avz_vchains;
            flatten_vectors=false,
        )
        avz_gcd = to_arviz_dict(
            avz_vec_model, avz_vargs, avz_vcm, avz_vchains;
            layout=:chain_draw, flatten_vectors=false,
        )

        # Default stays the flattened per-component convention with no
        # coords/dims entries.
        @test Set(keys(avz_flat["posterior"])) ==
              Set(["mu"; ["theta[$i]" for i = 1:3]])
        @test !haskey(avz_flat, "coords")
        @test !haskey(avz_flat, "dims")

        # Grouped posterior: scalar params keep their matrices, the vector
        # param becomes one array with a trailing component dim in both
        # layouts, matching the flattened columns exactly.
        @test Set(keys(avz_gdc["posterior"])) == Set(["mu", "theta"])
        @test avz_gdc["posterior"]["mu"] == avz_flat["posterior"]["mu"]
        @test size(avz_gdc["posterior"]["theta"]) == (avz_S, avz_C, 3)
        @test size(avz_gcd["posterior"]["theta"]) == (avz_C, avz_S, 3)
        for avz_i = 1:3
            @test avz_gdc["posterior"]["theta"][:, :, avz_i] ==
                  avz_flat["posterior"]["theta[$avz_i]"]
            @test avz_gcd["posterior"]["theta"][:, :, avz_i] ==
                  permutedims(avz_flat["posterior"]["theta[$avz_i]"])
        end

        # Grouped log_likelihood: indexed observation addresses collapse into
        # one array per base name, matching the flattened matrices.
        @test Set(keys(avz_gdc["log_likelihood"])) == Set(["y"])
        @test size(avz_gdc["log_likelihood"]["y"]) == (avz_S, avz_C, avz_vk)
        @test size(avz_gcd["log_likelihood"]["y"]) == (avz_C, avz_S, avz_vk)
        for avz_i = 1:avz_vk
            @test avz_gdc["log_likelihood"]["y"][:, :, avz_i] ==
                  avz_flat["log_likelihood"]["y[$avz_i]"]
        end

        # coords/dims follow the az.from_dict convention and are consistent
        # with the emitted array shapes.
        for avz_grouped in (avz_gdc, avz_gcd)
            @test avz_grouped["dims"] ==
                  Dict("theta" => ["theta_dim_0"], "y" => ["y_dim_0"])
            @test avz_grouped["coords"] ==
                  Dict("theta_dim_0" => collect(1:3), "y_dim_0" => collect(1:avz_vk))
        end

        # Scalar-only model: grouping is a no-op on the posterior keys, the
        # observations still collapse, and the chains-only method carries
        # (empty-parameter) coords/dims without a log_likelihood group.
        avz_sgrouped = to_arviz_dict(
            avz_model, avz_args, avz_cm, avz_chains;
            flatten_vectors=false,
        )
        avz_sflat = to_arviz_dict(avz_model, avz_args, avz_cm, avz_chains)
        @test avz_sgrouped["posterior"] == avz_sflat["posterior"]
        @test Set(keys(avz_sgrouped["log_likelihood"])) == Set(["y"])
        @test size(avz_sgrouped["log_likelihood"]["y"]) == (avz_S, avz_C, avz_n)
        @test avz_sgrouped["dims"] == Dict("y" => ["y_dim_0"])
        @test avz_sgrouped["coords"] == Dict("y_dim_0" => collect(1:avz_n))
        avz_splain = to_arviz_dict(avz_chains; flatten_vectors=false)
        @test avz_splain["posterior"] == avz_sflat["posterior"]
        @test avz_splain["coords"] == Dict{String,Any}()
        @test avz_splain["dims"] == Dict{String,Any}()
        @test !haskey(avz_splain, "log_likelihood")
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
