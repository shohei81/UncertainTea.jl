# PR 47: dense-covariance (Cholesky-parameterized) multivariate normal.
#
# Contract: `mvnormaldense(mu, scale_tril)` scores a d-dimensional gaussian with
# covariance L * L', where L = scale_tril is a d x d lower-triangular factor with
# strictly positive diagonal. Only the lower triangle of `scale_tril` is ever
# read (upper-triangular content is ignored), and the logpdf is computed by a
# hand-rolled forward substitution solving L z = x - mu — no LinearAlgebra
# factorization objects — so ForwardDiff Duals flow through unchanged. The family
# is CPU-reference only: it is honestly reported unsupported by the GPU backend,
# and batched calls take the ForwardDiff fallback. As a latent (parameter slot)
# the mean must have a statically known length, mirroring the diagonal
# `mvnormal`; with a non-static mean the choice is observation-only (no slot).
#
# Note: matrix literals like `[1.0 0.0; 0.8 0.6]` are not supported by the
# compiled lower-level expression compiler (`:vcat` heads), so in-model scale
# factors are supplied as module constants or model arguments below.

const mvd_LA = UncertainTea.LinearAlgebra
const mvd_FD = UncertainTea.ForwardDiff

mvd_mean(xs) = sum(xs) / length(xs)
function mvd_cor(xs, ys)
    mx = mvd_mean(xs)
    my = mvd_mean(ys)
    return sum((xs .- mx) .* (ys .- my)) /
           sqrt(sum(abs2, xs .- mx) * sum(abs2, ys .- my))
end

# --- logpdf against a LinearAlgebra reference ------------------------------
@testset "mvd_logpdf_reference" begin
    mvd_L = [2.0 0.0 0.0; 0.6 1.5 0.0; -0.3 0.4 0.9]
    mvd_mu = [0.5, -1.0, 2.0]
    mvd_d = mvnormaldense(mvd_mu, mvd_L)
    mvd_sigma = mvd_L * mvd_L'
    for mvd_x in (
        [0.0, 0.0, 0.0],
        [1.0, -2.0, 3.0],
        [0.5, -1.0, 2.0],
        [-4.0, 0.3, 1.7],
    )
        mvd_resid = mvd_x - mvd_mu
        mvd_ref = -0.5 * (
            mvd_LA.logdet(mvd_sigma) +
            mvd_LA.dot(mvd_resid, mvd_sigma \ mvd_resid) +
            3 * log(2 * pi)
        )
        @test UncertainTea.logpdf(mvd_d, mvd_x) ≈ mvd_ref atol = 1e-10
    end

    # Tuple values are accepted; a wrong-length value scores -Inf (mirroring
    # the diagonal mvnormal contract).
    @test UncertainTea.logpdf(mvd_d, (0.5, -1.0, 2.0)) ≈
          UncertainTea.logpdf(mvd_d, [0.5, -1.0, 2.0]) atol = 1e-12
    @test UncertainTea.logpdf(mvd_d, [0.0, 0.0]) == -Inf

    # Only the lower triangle is read: junk above the diagonal changes nothing.
    mvd_Lfull = copy(mvd_L)
    mvd_Lfull[1, 2] = 99.0
    mvd_Lfull[1, 3] = -7.0
    mvd_Lfull[2, 3] = 5.0
    @test UncertainTea.logpdf(mvnormaldense(mvd_mu, mvd_Lfull), [1.0, 2.0, 3.0]) ==
          UncertainTea.logpdf(mvd_d, [1.0, 2.0, 3.0])
end

# --- constructor validation -------------------------------------------------
@testset "mvd_validation" begin
    mvd_L = [2.0 0.0 0.0; 0.6 1.5 0.0; -0.3 0.4 0.9]
    mvd_mu = [0.5, -1.0, 2.0]
    # Mean/scale dimension mismatches throw.
    @test_throws ArgumentError mvnormaldense([0.0, 1.0], mvd_L)
    @test_throws ArgumentError mvnormaldense(mvd_mu, [1.0 0.0; 0.0 1.0])
    # Non-square scale matrices throw.
    @test_throws ArgumentError mvnormaldense(mvd_mu, ones(3, 2))
    # Non-positive diagonal entries throw.
    @test_throws ArgumentError mvnormaldense(
        mvd_mu, [2.0 0.0 0.0; 0.6 -1.5 0.0; -0.3 0.4 0.9],
    )
    @test_throws ArgumentError mvnormaldense(
        mvd_mu, [2.0 0.0 0.0; 0.6 0.0 0.0; -0.3 0.4 0.9],
    )
    # Empty means and non-vector/non-matrix arguments throw.
    @test_throws ArgumentError mvnormaldense(Float64[], zeros(0, 0))
    @test_throws ArgumentError mvnormaldense(1.0, mvd_L)
    @test_throws ArgumentError mvnormaldense(mvd_mu, [1.0, 2.0, 3.0])
    # Tuple means are accepted.
    @test mvnormaldense((0.5, -1.0, 2.0), mvd_L) isa UncertainTea.MvNormalDenseDist
end

# --- rand: sample moments match mu and L * L' -------------------------------
@testset "mvd_rand" begin
    mvd_L = [1.0 0.0 0.0; 0.8 0.6 0.0; -0.5 0.3 0.7]
    mvd_mu = [1.0, -2.0, 0.5]
    mvd_sigma = mvd_L * mvd_L'
    mvd_rng = MersenneTwister(20260706)
    mvd_n = 20000
    mvd_dist = mvnormaldense(mvd_mu, mvd_L)
    mvd_draws = reduce(hcat, [rand(mvd_rng, mvd_dist) for _ = 1:mvd_n])
    mvd_m = vec(sum(mvd_draws; dims=2)) ./ mvd_n
    @test maximum(abs.(mvd_m - mvd_mu)) < 0.05
    mvd_centered = mvd_draws .- mvd_m
    mvd_cov = mvd_centered * mvd_centered' ./ (mvd_n - 1)
    @test mvd_LA.norm(mvd_cov - mvd_sigma) / mvd_LA.norm(mvd_sigma) < 0.15
end

# --- observation end-to-end with a runtime scale argument -------------------
@testset "mvd_observation_end_to_end" begin
    @tea static function mvd_obs_model(Larg)
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ mvnormaldense([mu, mu], Larg)
    end

    mvd_obs_L = [1.0 0.0; 0.8 0.6]
    # A single scalar latent slot; the dense observation carries no slot.
    mvd_obs_layout = parameterlayout(mvd_obs_model)
    @test length(mvd_obs_layout.slots) == 1
    @test parametercount(mvd_obs_layout) == 1
    @test mvd_obs_layout.slots[1].transform isa IdentityTransform

    # generate/logjoint agreement over the full joint: `mu` is scored as a
    # latent (from the parameter vector) and only the observation `:y` is
    # conditioned on (docs/constraint-driven-conditioning.md, issue #95).
    mvd_obs_full = choicemap((:mu, 0.4f0), (:y, [0.3, -0.2]))
    mvd_obs_trace, _ = generate(mvd_obs_model, (mvd_obs_L,), mvd_obs_full; rng=MersenneTwister(11))
    mvd_obs_pv = parameter_vector(mvd_obs_trace)
    @test mvd_obs_trace.log_weight ≈
          logjoint(mvd_obs_model, mvd_obs_pv, (mvd_obs_L,), choicemap((:y, [0.3, -0.2]))) atol = 1e-6

    # Batched logjoint (ForwardDiff fallback) matches the per-column value.
    mvd_obs = choicemap((:y, [0.3, -0.2]))
    mvd_obs_u = transform_to_unconstrained(mvd_obs_model, [0.4])
    mvd_obs_params = hcat(mvd_obs_u, mvd_obs_u .+ 0.6, mvd_obs_u .- 0.8)
    mvd_obs_batched = batched_logjoint_unconstrained(
        mvd_obs_model, mvd_obs_params, (mvd_obs_L,), mvd_obs,
    )
    mvd_obs_percol = [
        logjoint_unconstrained(mvd_obs_model, mvd_obs_params[:, i], (mvd_obs_L,), mvd_obs)
        for i = 1:size(mvd_obs_params, 2)
    ]
    @test mvd_obs_batched ≈ mvd_obs_percol atol = 1e-6

    # Batched gradient parity against per-column ForwardDiff.
    mvd_obs_grad = batched_logjoint_gradient_unconstrained(
        mvd_obs_model, mvd_obs_params, (mvd_obs_L,), mvd_obs,
    )
    for i = 1:size(mvd_obs_params, 2)
        mvd_obs_col = mvd_FD.gradient(
            theta -> logjoint_unconstrained(mvd_obs_model, theta, (mvd_obs_L,), mvd_obs),
            mvd_obs_params[:, i],
        )
        @test mvd_obs_grad[:, i] ≈ mvd_obs_col atol = 1e-8
    end
end

# --- latent end-to-end: vector slot, z[1] indexing, NUTS correlation --------
const mvd_latent_L = [1.0 0.0; 0.8 0.6]

@testset "mvd_latent_end_to_end" begin
    @tea static function mvd_latent_model()
        z ~ mvnormaldense([0.0f0, 0.0f0], mvd_latent_L)
        {:y} ~ normal(z[1] + z[2], 0.5f0)
    end

    # A 2-wide vector identity slot from the static mean literal.
    mvd_latent_layout = parameterlayout(mvd_latent_model)
    @test length(mvd_latent_layout.slots) == 1
    @test parametercount(mvd_latent_layout) == 2
    @test parametervaluecount(mvd_latent_layout) == 2
    @test mvd_latent_layout.slots[1].transform isa VectorIdentityTransform

    # generate/logjoint agreement, exercising `z[1]`/`z[2]` indexing of the
    # vector latent binding in both the impl and the compiled plan path. `z` is
    # scored as a latent (from the parameter vector); only the observation `:y`
    # is conditioned on (docs/constraint-driven-conditioning.md, issue #95).
    mvd_latent_full = choicemap((:z, [0.2, -0.1]), (:y, 0.5))
    mvd_latent_trace, _ = generate(mvd_latent_model, (), mvd_latent_full; rng=MersenneTwister(12))
    mvd_latent_pv = parameter_vector(mvd_latent_trace)
    @test mvd_latent_trace.log_weight ≈
          logjoint(mvd_latent_model, mvd_latent_pv, (), choicemap((:y, 0.5))) atol = 1e-6

    # NUTS runs finite; observing the sum z1 + z2 bends the positively
    # correlated prior (cor 0.8) into a negatively correlated posterior along
    # the likelihood constraint.
    mvd_latent_chain = first(nuts(
        mvd_latent_model,
        (),
        choicemap((:y, 1.0));
        num_samples=200,
        num_warmup=200,
        rng=MersenneTwister(3),
    ))
    mvd_latent_samples = mvd_latent_chain.constrained_samples
    @test size(mvd_latent_samples) == (2, 200)
    @test all(isfinite, mvd_latent_samples)

    # Posterior correlation sign-flip check. The analytic posterior covariance
    # (Gaussian conditional of z ~ N(0, LL') given z1 + z2 + N(0, 0.5^2) = 1.0)
    # has correlation exactly -0.262, versus the prior +0.8. A single 200-draw
    # chain estimates this at only ~1 sigma (cor scatter ~ (1-r^2)/sqrt(n)), so
    # the check is done on the pooled draws of a multi-chain run to shrink the
    # estimator variance; the threshold keeps genuine margin over the analytic
    # value and is robust across Julia versions / the sampler RNG stream.
    mvd_latent_chains = nuts_chains(
        mvd_latent_model,
        (),
        choicemap((:y, 1.0));
        num_chains=6,
        num_samples=600,
        num_warmup=400,
        rng=MersenneTwister(3),
    )
    mvd_pooled_z1 = vcat((c.constrained_samples[1, :] for c in mvd_latent_chains.chains)...)
    mvd_pooled_z2 = vcat((c.constrained_samples[2, :] for c in mvd_latent_chains.chains)...)
    @test all(isfinite, mvd_pooled_z1)
    @test all(isfinite, mvd_pooled_z2)
    @test mvd_cor(mvd_pooled_z1, mvd_pooled_z2) < -0.18
end

# --- non-static mean lengths resolve from the arguments (issue #289) --------
@testset "mvd_dynamic_mean_runtime_dims" begin
    # The macro-time static-size probes are unchanged; a runtime-length mean is
    # simply deferred to (signature, dims) resolution instead of being
    # observation-only (the pre-#289 behavior).
    @test UncertainTea._mvnormaldense_static_size(:(mvnormaldense([a, b], L))) == 2
    @test UncertainTea._mvnormaldense_static_size(:(mvnormaldense(muvec, L))) === nothing
    @test UncertainTea._supported_distribution_family(:(mvnormaldense([a, b], L))) === :mvnormaldense
    @test UncertainTea._supported_distribution_family(:(mvnormaldense(muvec, L))) === nothing

    @tea static function mvd_dynamic_mu_model(muvec, Larg)
        w ~ mvnormaldense(muvec, Larg)
        {:o} ~ normal(1.0, 1.0)
    end
    # The args-independent default-layout API cannot size the slot and says so.
    @test_throws ArgumentError parameterlayout(mvd_dynamic_mu_model)

    mvd_dyn_mu = [0.0, 0.0, 0.0]
    mvd_dyn_L = [1.0 0.0 0.0; 0.2 1.0 0.0; 0.1 -0.3 1.0]
    # Unconstrained `w` is now a real latent with an args-resolved slot ...
    mvd_dyn_w = [0.4, -0.6, 0.2]
    mvd_dyn_latent_expected =
        UncertainTea.logpdf(mvnormaldense(mvd_dyn_mu, mvd_dyn_L), mvd_dyn_w) +
        UncertainTea.logpdf(normal(1.0, 1.0), 1.0)
    @test logjoint(mvd_dynamic_mu_model, mvd_dyn_w, (mvd_dyn_mu, mvd_dyn_L), choicemap((:o, 1.0))) ≈
          mvd_dyn_latent_expected atol = 1e-10
    # ... and constraining it still scores it as an observation.
    mvd_dyn_full = choicemap((:w, mvd_dyn_w), (:o, 1.0))
    @test logjoint(mvd_dynamic_mu_model, Float64[], (mvd_dyn_mu, mvd_dyn_L), mvd_dyn_full) ≈
          mvd_dyn_latent_expected atol = 1e-10
end

# --- backend now natively supports the family (see PR 50) --------------------
# An mvnormaldense observation with a constant (model-argument) scale factor is
# lowered to a BackendMvNormalDenseChoicePlanStep and scored natively.
@testset "mvd_backend_report" begin
    @tea static function mvd_report_model(Larg)
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ mvnormaldense([mu, mu], Larg)
    end
    mvd_report = backend_report(mvd_report_model)
    @test mvd_report.supported == true
    @test backend_execution_plan(mvd_report_model).steps[2] isa
          UncertainTea.BackendMvNormalDenseChoicePlanStep
end
