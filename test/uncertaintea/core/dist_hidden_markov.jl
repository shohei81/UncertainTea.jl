# Hidden Markov model family (issue #261): a Gaussian-emission HMM scored through
# the forward algorithm, which marginalizes the hidden state path in O(T*K^2).
# Fixed dynamics (init/transition are model arguments) with latent emission means
# + a shared emission sd. CPU-reference only (the loop-carried recursion is not
# device-lowered). Statistics is not available in the harness, so use local
# helpers.

using LinearAlgebra

hmm_mean(x) = sum(x) / length(x)

# forward-algorithm log marginal likelihood, computed independently of the family
# implementation so the test pins the intended math, not the same code.
function hmm_reference_loglik(init, transition, means, sigma, y)
    k = length(init)
    lse(v) = (m=maximum(v); m + log(sum(x -> exp(x - m), v)))
    emit(yt, s) = -0.5 * ((yt - means[s]) / sigma)^2 - log(sigma) - 0.5 * log(2pi)
    logalpha = [log(init[s]) + emit(y[1], s) for s = 1:k]
    for t = 2:length(y)
        logalpha = [lse([logalpha[p] + log(transition[p, s]) for p = 1:k]) + emit(y[t], s) for s = 1:k]
    end
    return lse(logalpha)
end

@tea static function hmm_emission_model(init, trans, seqlen)
    m1 ~ normal(-1.0, 2.0)
    log_gap ~ normal(0.0, 1.0)
    logs ~ normal(-0.5, 0.5)
    {:y} ~ hmm(init, trans, [m1, m1 + exp(log_gap)], exp(logs))
    return m1
end

@testset "dist_hidden_markov" begin
    hmm_rng = MersenneTwister(261)
    hmm_init = [0.6, 0.4]
    hmm_trans = [0.8 0.2; 0.3 0.7]
    hmm_means = [-1.5, 2.0]
    hmm_sigma = 0.5
    hmm_T = 120
    # simulate a 2-state Gaussian HMM path + emissions
    hmm_y = let ys = Float64[], s = rand(hmm_rng) < hmm_init[1] ? 1 : 2
        for _ = 1:hmm_T
            push!(ys, hmm_means[s] + hmm_sigma * randn(hmm_rng))
            s = rand(hmm_rng) < hmm_trans[s, 1] ? 1 : 2
        end
        ys
    end

    @testset "logpdf matches the forward-algorithm marginal" begin
        h = hmm(hmm_init, hmm_trans, hmm_means, hmm_sigma)
        ref = hmm_reference_loglik(hmm_init, hmm_trans, hmm_means, hmm_sigma, hmm_y)
        @test UncertainTea.logpdf(h, hmm_y) ≈ ref rtol = 1e-12

        # a single observation reduces to the mixture over initial states
        h1 = hmm(hmm_init, hmm_trans, hmm_means, hmm_sigma)
        ref1 = hmm_reference_loglik(hmm_init, hmm_trans, hmm_means, hmm_sigma, hmm_y[1:1])
        @test UncertainTea.logpdf(h1, hmm_y[1:1]) ≈ ref1 rtol = 1e-12
    end

    @testset "constructor validates the state dimensions" begin
        @test_throws ArgumentError hmm(hmm_init, [0.8 0.2; 0.3 0.7; 0.0 1.0], hmm_means, hmm_sigma)
        @test_throws ArgumentError hmm(hmm_init, hmm_trans, [-1.5, 0.0, 2.0], hmm_sigma)
        @test_throws ArgumentError UncertainTea.logpdf(hmm(hmm_init, hmm_trans, hmm_means, hmm_sigma), Float64[])
    end

    @testset "backend/device honestly unsupported (CPU-reference only)" begin
        @test !UncertainTea.backend_report(hmm_emission_model).supported
        @test !device_lowering_report(hmm_emission_model)[1]
    end

    @testset "NUTS recovers the emission parameters" begin
        hmm_cm = choicemap((:y, hmm_y))
        hmm_chain = nuts(
            hmm_emission_model, (hmm_init, hmm_trans, hmm_T), hmm_cm;
            num_samples=300, num_warmup=300, rng=MersenneTwister(7),
        )
        hmm_draws = hmm_chain.constrained_samples
        @test all(isfinite, hmm_draws)
        hmm_m1 = hmm_mean(hmm_draws[1, :])
        hmm_m2 = hmm_m1 + exp(hmm_mean(hmm_draws[2, :]))
        hmm_sd = exp(hmm_mean(hmm_draws[3, :]))
        @test abs(hmm_m1 - hmm_means[1]) < 0.4    # low emission mean ~ -1.5
        @test abs(hmm_m2 - hmm_means[2]) < 0.4    # high emission mean ~ 2.0
        @test abs(hmm_sd - hmm_sigma) < 0.2       # shared emission sd ~ 0.5
    end
end
