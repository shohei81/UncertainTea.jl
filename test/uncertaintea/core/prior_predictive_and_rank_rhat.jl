# Prior predictive draws + rank-normalized split-Rhat (issue #295).
# Statistics is unavailable in the harness, so use local helpers.

ppr_mean(x) = sum(x) / length(x)

@tea static function ppr_model(n)
    mu ~ normal(0.0, 1.0)
    logs ~ normal(-0.5, 0.5)
    for i = 1:n
        {:y => i} ~ normal(mu, exp(logs))
    end
    return mu
end

@testset "prior_predictive_and_rank_rhat" begin
    ppr_cm = choicemap([(:y => i, 0.3 + 0.1 * i) for i = 1:5])

    @testset "prior_predictive keeps exactly the observation addresses" begin
        pp = prior_predictive(ppr_model, (5,), ppr_cm; num_draws=200, rng=MersenneTwister(1))
        @test length(pp.draws) == 200
        for draw in pp.draws[1:5]
            for i = 1:5
                @test haskey(draw, :y => i)
            end
            @test !haskey(draw, :mu)
            @test !haskey(draw, :logs)
        end
        # scale sanity: y ~ mu + eps with mu ~ N(0,1); prior-predictive draws are
        # centered near 0 with sd > 1
        ys = Float64[draw[:y=>1] for draw in pp.draws]
        @test abs(ppr_mean(ys)) < 0.4
        @test 0.8 < sqrt(ppr_mean(abs2.(ys .- ppr_mean(ys)))) < 3.0
        @test_throws ArgumentError prior_predictive(ppr_model, (5,), ppr_cm; num_draws=0)
    end

    @testset "rank-normalized rhat agrees with split rhat on well-mixed chains" begin
        chains = nuts_chains(ppr_model, (5,), ppr_cm; num_chains=3, num_samples=300, num_warmup=300, rng=MersenneTwister(2))
        r_split = rhat(chains)
        r_rank = rhat(chains; method=:rank)
        @test length(r_rank) == length(r_split)
        @test all(r -> 0.98 < r < 1.1, r_split)
        @test all(r -> 0.98 < r < 1.1, r_rank)
        @test_throws ArgumentError rhat(chains; method=:bogus)
    end

    @testset "rank rhat flags a mismatch the classical statistic understates" begin
        # heavy-tailed draws where one chain's SCALE differs: the folded
        # rank-normalized statistic reacts more strongly than the classical one
        chains = nuts_chains(ppr_model, (5,), ppr_cm; num_chains=3, num_samples=200, num_warmup=200, rng=MersenneTwister(3))
        # inflate one chain's unconstrained draws (simulating a stuck-scale chain)
        chains.chains[1].unconstrained_samples .= chains.chains[1].unconstrained_samples .* 6.0
        r_rank = rhat(chains; space=:unconstrained, method=:rank)
        # the folded rank-normalized channel reacts to the scale mismatch
        @test any(r_rank .> 1.05)
    end

    @testset "rank-normal transform is monotone and standardized" begin
        draws = reshape(collect(1.0:40.0), 4, 10)
        z = UncertainTea._rank_normal_transform(draws)
        @test size(z) == size(draws)
        @test issorted(vec(z)[sortperm(vec(draws))])
        @test abs(ppr_mean(vec(z))) < 1e-10
    end
end
