# Systematic cross-checks between the hand-written analytic batched gradients,
# the backend-native batched scoring, and the CPU reference logjoint (issue #9).
#
# The family list is driven by UncertainTea.GPU_BACKEND_SUPPORTED_DISTRIBUTIONS,
# so registering a new backend-native family makes this suite FAIL with
# "family not covered" until a cross-check model is added to gxc_models below.
#
# For every model, at several seeded random unconstrained points:
#   1. the model must lower (backend_report) and sit on the ANALYTIC gradient
#      tier (backend_cache set) -- otherwise the comparisons below would
#      silently compare ForwardDiff against itself;
#   2. batched backend-native scoring == per-column CPU reference logjoint;
#   3. analytic batched gradient == central finite differences of the CPU
#      reference logjoint (an oracle independent of ForwardDiff, so it also
#      catches wrong custom Dual rules);
#   4. analytic batched gradient == per-column ForwardDiff gradient;
#   5. the Float32 batched gradient tracks the Float64 one (device precision).
@testset "gradient_crosscheck" begin
    gxc_fd_gradient = function (model, x, args, constraints)
        g = similar(x)
        for i in eachindex(x)
            h = cbrt(eps(Float64)) * max(1.0, abs(x[i]))
            xp = copy(x)
            xp[i] += h
            xm = copy(x)
            xm[i] -= h
            g[i] =
                (
                    logjoint_unconstrained(model, xp, args, constraints) -
                    logjoint_unconstrained(model, xm, args, constraints)
                ) / (2h)
        end
        return g
    end

    gxc_check = function (model, args, constraints, seed)
        rng = MersenneTwister(seed)
        trace, _ = generate(model, args, constraints; rng=rng)
        base = transform_to_unconstrained(trace)
        points = base .+ 0.4 .* randn(rng, length(base), 3)

        # 1. lowering + analytic tier
        @test backend_report(model).supported == true
        cache = BatchedLogjointGradientCache(model, points, args, constraints)
        @test !isnothing(cache.backend_cache)
        @test isnothing(cache.flat_cache)
        @test isempty(cache.column_caches)

        # 2. backend-native scoring vs CPU reference logjoint
        batched_values = batched_logjoint_unconstrained(model, points, args, constraints)
        reference_values = [
            logjoint_unconstrained(model, points[:, i], args, constraints) for i = 1:size(points, 2)
        ]
        @test batched_values ≈ reference_values atol = 1e-8

        # 3./4. analytic gradient vs finite differences and vs ForwardDiff
        analytic = batched_logjoint_gradient_unconstrained(model, points, args, constraints)
        for i = 1:size(points, 2)
            @test analytic[:, i] ≈ gxc_fd_gradient(model, points[:, i], args, constraints) atol = 5e-6
            @test analytic[:, i] ≈
                  logjoint_gradient_unconstrained(model, points[:, i], args, constraints) atol = 1e-8
        end

        # 5. Float32 gradient tracks Float64 (the Metal device path runs f32)
        analytic32 = batched_logjoint_gradient_unconstrained(
            model, Float32.(points), args, constraints,
        )
        @test eltype(analytic32) == Float32
        @test Float64.(analytic32) ≈ analytic atol = 2e-3
        return nothing
    end

    # ------------------------------------------------------------------
    # Cross-check models. Every parameter slot the analytic gradient
    # differentiates should depend on a latent (so d/dmu, d/dsigma, ... all
    # flow), and latent-form entries exercise each family's d/dvalue partial
    # plus its transform jacobian.
    # ------------------------------------------------------------------

    @tea static function gxc_normal_obs()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ normal(a, exp(s))
        return a
    end

    # reparam=:noncentered lowers to the z-space backend step (PR-4 of
    # docs/noncentered-reparam.md); the cross-check exercises its analytic
    # gradient against finite differences like any other family entry
    @tea static function gxc_normal_noncentered()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        theta ~ normal(a, exp(s); reparam=:noncentered)
        {:y} ~ normal(theta, 0.5f0)
        return theta
    end

    @tea static function gxc_lognormal_obs()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ lognormal(a, exp(s))
        return a
    end

    @tea static function gxc_lognormal_latent()
        x ~ lognormal(0.2f0, 0.6f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_laplace_obs()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ laplace(a, exp(s))
        return a
    end

    @tea static function gxc_laplace_latent()
        x ~ laplace(0.5f0, 1.2f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_exponential_obs()
        s ~ normal(0.0f0, 0.4f0)
        {:y} ~ exponential(exp(s))
        return s
    end

    @tea static function gxc_exponential_latent()
        x ~ exponential(1.5f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_gamma_obs()
        a ~ normal(0.5f0, 0.3f0)
        b ~ normal(0.0f0, 0.3f0)
        {:y} ~ gamma(exp(a), exp(b))
        return a
    end

    @tea static function gxc_gamma_latent()
        x ~ gamma(2.5f0, 1.5f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_inversegamma_obs()
        a ~ normal(0.8f0, 0.25f0)
        b ~ normal(0.3f0, 0.25f0)
        {:y} ~ inversegamma(exp(a), exp(b))
        return a
    end

    @tea static function gxc_inversegamma_latent()
        x ~ inversegamma(3.0f0, 2.0f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_weibull_obs()
        a ~ normal(0.4f0, 0.25f0)
        b ~ normal(0.2f0, 0.25f0)
        {:y} ~ weibull(exp(a), exp(b))
        return a
    end

    @tea static function gxc_weibull_latent()
        x ~ weibull(1.8f0, 1.2f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_beta_obs()
        a ~ normal(0.6f0, 0.25f0)
        b ~ normal(0.9f0, 0.25f0)
        {:y} ~ beta(exp(a), exp(b))
        return a
    end

    @tea static function gxc_beta_latent()
        x ~ beta(2.0f0, 3.0f0)
        {:y} ~ normal(x, 0.5f0)
        return x
    end

    @tea static function gxc_studentt_obs()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ studentt(4.0f0, a, exp(s))
        return a
    end

    @tea static function gxc_studentt_latent()
        x ~ studentt(5.0f0, 0.3f0, 1.1f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    # marginalize=:enumerate latents (issue #13 PR-5): the suffix-owning
    # backend step's analytic gradient is the responsibility-weighted sum of
    # branch gradients; the bernoulli entry chains d(log pmf)/dp through a
    # latent probability, the categorical entry adds a suffix rebind and
    # latent-dependent probabilities
    @tea static function gxc_bernoulli_marginalized()
        m1 ~ normal(-2.0, 1.0)
        m2 ~ normal(2.0, 1.0)
        p ~ beta(2.0, 2.0)
        z ~ bernoulli(p; marginalize=:enumerate)
        {:y} ~ normal(z * m1 + (1 - z) * m2, 0.5)
        return m1
    end

    @tea static function gxc_categorical_marginalized()
        x ~ beta(2.0, 2.0)
        s = 1.0
        z ~ categorical([x * 0.5, 0.3, 0.7 - x * 0.5]; marginalize=:enumerate)
        s = s + 1.0
        {:y} ~ normal(1.0 * z * s, 0.8)
        return x
    end

    @tea static function gxc_bernoulli_obs()
        p ~ beta(2.0f0, 3.0f0)
        {:y} ~ bernoulli(p)
        return p
    end

    # generic scalar eta: exercises the bernoullilogit step with a plain numeric
    # eta expression (issue #149/#150)
    @tea static function gxc_bernoullilogit_obs()
        s ~ normal(0.0f0, 1.0f0)
        {:y} ~ bernoullilogit(2.0f0 * s - 0.5f0)
        return s
    end

    # fused GLM linear predictor: exercises BackendLinearPredictorExpr — the
    # intercept `a` flows through its own gradient, and each coefficient
    # `b[d]`'s partial is `X[d, 1]` seeded onto its unconstrained row (issue #150)
    @tea static function gxc_bernoullilogit_glm(X)
        a ~ normal(0.0f0, 1.0f0)
        b ~ mvnormal((0.0f0, 0.0f0), (1.0f0, 1.0f0))
        {:y} ~ bernoullilogit(a + sum(b .* X[:, 1]))
        return a
    end

    @tea static function gxc_binomial_obs()
        logit ~ normal(0.0f0, 0.5f0)
        probability = 1.0f0 / (1.0f0 + exp(-logit))
        {:y} ~ binomial(8, probability)
        return probability
    end

    @tea static function gxc_geometric_obs()
        logit ~ normal(0.0f0, 0.6f0)
        probability = 1.0f0 / (1.0f0 + exp(-logit))
        {:y} ~ geometric(probability)
        return probability
    end

    @tea static function gxc_negativebinomial_obs()
        log_successes ~ normal(1.2f0, 0.3f0)
        logit ~ normal(0.0f0, 0.5f0)
        successes = exp(log_successes)
        probability = 1.0f0 / (1.0f0 + exp(-logit))
        {:y} ~ negativebinomial(successes, probability)
        return successes
    end

    @tea static function gxc_poisson_obs()
        a ~ normal(0.5f0, 0.4f0)
        {:y} ~ poisson(exp(a))
        return a
    end

    @tea static function gxc_categorical_obs()
        logit1 ~ normal(0.0f0, 0.5f0)
        logit2 ~ normal(0.0f0, 0.5f0)
        w1 = exp(logit1)
        w2 = exp(logit2)
        denom = w1 + w2 + 1.0f0
        {:y} ~ categorical(w1 / denom, w2 / denom, 1.0f0 / denom)
        return logit1
    end

    # Float64 concentrations: with Float32 alphas the CPU reference computes
    # its lgamma normalization at Float32 while the backend slots are Float64,
    # leaving a constant ~6e-7 value offset that is about precision, not math.
    @tea static function gxc_dirichlet_latent()
        weights ~ dirichlet([2.0, 3.0, 4.0])
        return weights
    end

    @tea static function gxc_mvnormal_latent()
        state ~ mvnormal([0.0f0, 1.0f0], [1.5f0, 0.8f0])
        return state
    end

    # vector latent followed by a scalar latent: the slot ordinal and the
    # constrained-matrix row differ, the exact gap of issue #36
    @tea static function gxc_mvnormal_then_scalar()
        w ~ mvnormal([0.0f0, 1.0f0], [1.5f0, 0.8f0])
        s ~ normal(0.0f0, 1.0f0)
        {:y} ~ normal(s, 0.5f0)
        return s
    end

    # DIMENSION-CHANGING (simplex) latent followed by scalar latents: every
    # later slot has index != value_index, exercising the value-row ->
    # seed-row split (issue #36, previously an honest fallback). The gamma
    # latent additionally exercises the log-transform chain rule on shifted
    # rows.
    @tea static function gxc_dirichlet_then_scalar()
        weights ~ dirichlet([2.0, 3.0, 4.0])
        s ~ normal(0.0f0, 1.0f0)
        tau ~ gamma(2.0f0, 1.5f0)
        {:y} ~ normal(s, tau)
        return s
    end

    # simplex before a latent MIXTURE: the mixture lowering stores a value row
    # like every scalar-valued step (it used to keep the raw slot ordinal,
    # which only the pre-fix rejection kept from misreading a simplex
    # component as the mixture value). Float64 literals: mixed-precision
    # literals shift the log-sum-exp by ~2e-9 between the batched and
    # per-column reference paths, which is precision, not math.
    @tea static function gxc_dirichlet_then_mixture()
        weights ~ dirichlet([2.0, 3.0, 4.0])
        x ~ mixture((0.5, 0.5), normal(-2.0, 0.5), normal(2.0, 0.5))
        {:y} ~ normal(x, 0.5)
        return x
    end

    # Float64 eta literals throughout the lkjcholesky entries: the compiled
    # reference computes the normalizing constant in the literal's precision,
    # so an f32 literal shifts it by ~1e-8 against the batched Float64 path.
    @tea static function gxc_lkjcholesky_latent()
        L ~ lkjcholesky(3, 2.0)
        return L
    end

    # dimension-changing (cholesky) latent followed by scalar latents: 3
    # unconstrained rows vs 6 value rows shift every later slot, exercising
    # the value-row -> seed-row split like the dirichlet variant. The gamma
    # latent adds the log-transform chain rule on shifted rows.
    @tea static function gxc_lkjcholesky_then_scalar()
        L ~ lkjcholesky(2, 1.5)
        s ~ normal(0.0f0, 1.0f0)
        tau ~ gamma(2.0f0, 1.5f0)
        {:y} ~ normal(s, tau)
        return s
    end

    # latent-dependent concentration: d(logpdf)/deta (digamma normalizer plus
    # the 2 log L[i,i] density term) chained through the eta expression
    @tea static function gxc_lkjcholesky_latent_eta()
        x ~ normal(0.5f0, 0.3f0)
        L ~ lkjcholesky(3, exp(x))
        return L
    end

    @tea static function gxc_mvnormal_obs()
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ mvnormal([m, m], [1.0f0, 0.8f0])
        return m
    end

    @tea static function gxc_truncatednormal_obs()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ truncatednormal(0.5f0, exp(s), 0.0, Inf)
        return s
    end

    @tea static function gxc_truncatednormal_mean_obs()
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ truncatednormal(m, 1.0f0, -1.0, 2.0)
        return m
    end

    @tea static function gxc_truncatedstudentt_obs()
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ truncatedstudentt(4.0f0, m, 1.0f0, -1.0, 2.0)
        return m
    end

    @tea static function gxc_mixture_obs()
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ mixture((0.3, 0.7), normal(m, 1.0f0), normal(2.0f0, 0.5f0))
        return m
    end

    @tea static function gxc_mixture_latent()
        x ~ mixture((0.5f0, 0.5f0), normal(-2.0f0, 0.5f0), normal(2.0f0, 0.5f0))
        {:y} ~ normal(x, 0.5f0)
        return x
    end

    @tea static function gxc_mvnormaldense_obs(Larg)
        m ~ normal(0.0f0, 1.0f0)
        {:y} ~ mvnormaldense([m, m], Larg)
        return m
    end

    # scalar prior families (issue #229). cauchy/logistic/gumbel are real-line
    # (identity transform), halfnormal/halfcauchy positive (log transform), so all
    # five are backend-native as both observations and latents. `uniform` is
    # observation-only on the backend (its bounded transform falls back for
    # latents, exactly like the truncated families), so it gets an obs entry whose
    # bounds depend on a latent to exercise the dlower/dupper channels.
    @tea static function gxc_cauchy_obs()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ cauchy(a, exp(s))
        return a
    end

    @tea static function gxc_cauchy_latent()
        x ~ cauchy(0.3f0, 1.2f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_halfnormal_obs()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ halfnormal(exp(s))
        return s
    end

    @tea static function gxc_halfnormal_latent()
        x ~ halfnormal(1.5f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_halfcauchy_obs()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ halfcauchy(exp(s))
        return s
    end

    @tea static function gxc_halfcauchy_latent()
        x ~ halfcauchy(1.2f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_logistic_obs()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ logistic(a, exp(s))
        return a
    end

    @tea static function gxc_logistic_latent()
        x ~ logistic(0.3f0, 1.2f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    @tea static function gxc_gumbel_obs()
        a ~ normal(0.0f0, 1.0f0)
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ gumbel(a, exp(s))
        return a
    end

    @tea static function gxc_gumbel_latent()
        x ~ gumbel(0.3f0, 1.2f0)
        {:y} ~ normal(x, 1.0f0)
        return x
    end

    # observation-only uniform: the observed y stays deep inside the widening
    # bounds across the seeded unconstrained points and the finite-difference
    # steps, so the density (and the analytic dlower/dupper gradient) is smooth.
    @tea static function gxc_uniform_obs()
        w ~ normal(0.0f0, 0.3f0)
        {:y} ~ uniform(-2.0f0 - exp(w), 2.0f0 + exp(w))
        return w
    end

    gxc_dense_factor = [1.0 0.0; 0.3 0.8]
    gxc_glm_covariates = reshape(Float32[0.7, -1.1], 2, 1)  # d x n = 2 x 1

    # family => [(model, args, constraints), ...]; every registry family must
    # have an entry (enforced below).
    gxc_models = Dict(
        :normal => [
            (gxc_normal_obs, (), choicemap((:y, 0.4f0))),
            (gxc_normal_noncentered, (), choicemap((:y, 0.4f0))),
        ],
        :lognormal => [
            (gxc_lognormal_obs, (), choicemap((:y, 1.3f0))),
            (gxc_lognormal_latent, (), choicemap((:y, 1.1f0))),
        ],
        :laplace => [
            (gxc_laplace_obs, (), choicemap((:y, 0.7f0))),
            (gxc_laplace_latent, (), choicemap((:y, 0.9f0))),
        ],
        :exponential => [
            (gxc_exponential_obs, (), choicemap((:y, 0.8f0))),
            (gxc_exponential_latent, (), choicemap((:y, 1.2f0))),
        ],
        :gamma => [
            (gxc_gamma_obs, (), choicemap((:y, 1.4f0))),
            (gxc_gamma_latent, (), choicemap((:y, 1.6f0))),
        ],
        :inversegamma => [
            (gxc_inversegamma_obs, (), choicemap((:y, 0.9f0))),
            (gxc_inversegamma_latent, (), choicemap((:y, 1.1f0))),
        ],
        :weibull => [
            (gxc_weibull_obs, (), choicemap((:y, 1.1f0))),
            (gxc_weibull_latent, (), choicemap((:y, 0.9f0))),
        ],
        :beta => [
            (gxc_beta_obs, (), choicemap((:y, 0.45f0))),
            (gxc_beta_latent, (), choicemap((:y, 0.5f0))),
        ],
        :studentt => [
            (gxc_studentt_obs, (), choicemap((:y, 0.6f0))),
            (gxc_studentt_latent, (), choicemap((:y, 0.4f0))),
        ],
        :bernoulli => [
            (gxc_bernoulli_obs, (), choicemap((:y, true))),
            (gxc_bernoulli_marginalized, (), choicemap((:y, 0.8))),
        ],
        :bernoullilogit => [
            (gxc_bernoullilogit_obs, (), choicemap((:y, true))),
            (gxc_bernoullilogit_glm, (gxc_glm_covariates,), choicemap((:y, false))),
        ],
        :binomial => [(gxc_binomial_obs, (), choicemap((:y, 5)))],
        :geometric => [(gxc_geometric_obs, (), choicemap((:y, 3)))],
        :negativebinomial => [(gxc_negativebinomial_obs, (), choicemap((:y, 4)))],
        :poisson => [(gxc_poisson_obs, (), choicemap((:y, 2)))],
        :categorical => [
            (gxc_categorical_obs, (), choicemap((:y, 2))),
            (gxc_categorical_marginalized, (), choicemap((:y, 2.2))),
        ],
        :dirichlet => [
            (gxc_dirichlet_latent, (), choicemap()),
            (gxc_dirichlet_then_scalar, (), choicemap((:y, 0.4f0))),
            (gxc_dirichlet_then_mixture, (), choicemap((:y, 1.4))),
        ],
        :mvnormal => [
            (gxc_mvnormal_latent, (), choicemap()),
            (gxc_mvnormal_obs, (), choicemap((:y, Float32[0.4, -0.2]))),
            (gxc_mvnormal_then_scalar, (), choicemap((:y, 0.4f0))),
        ],
        :truncatednormal => [
            (gxc_truncatednormal_obs, (), choicemap((:y, 0.9f0))),
            (gxc_truncatednormal_mean_obs, (), choicemap((:y, 0.3f0))),
        ],
        :truncatedstudentt => [(gxc_truncatedstudentt_obs, (), choicemap((:y, 0.3f0)))],
        :mixture => [
            (gxc_mixture_obs, (), choicemap((:y, 1.2f0))),
            (gxc_mixture_latent, (), choicemap((:y, 1.5f0))),
        ],
        :mvnormaldense => [
            (gxc_mvnormaldense_obs, (gxc_dense_factor,), choicemap((:y, Float32[0.4, -0.2]))),
        ],
        :lkjcholesky => [
            (gxc_lkjcholesky_latent, (), choicemap()),
            (gxc_lkjcholesky_then_scalar, (), choicemap((:y, 0.4f0))),
            (gxc_lkjcholesky_latent_eta, (), choicemap()),
        ],
        :cauchy => [
            (gxc_cauchy_obs, (), choicemap((:y, 0.6f0))),
            (gxc_cauchy_latent, (), choicemap((:y, 0.4f0))),
        ],
        :halfnormal => [
            (gxc_halfnormal_obs, (), choicemap((:y, 0.8f0))),
            (gxc_halfnormal_latent, (), choicemap((:y, 1.1f0))),
        ],
        :halfcauchy => [
            (gxc_halfcauchy_obs, (), choicemap((:y, 0.9f0))),
            (gxc_halfcauchy_latent, (), choicemap((:y, 1.1f0))),
        ],
        :uniform => [(gxc_uniform_obs, (), choicemap((:y, 0.3f0)))],
        :logistic => [
            (gxc_logistic_obs, (), choicemap((:y, 0.7f0))),
            (gxc_logistic_latent, (), choicemap((:y, 0.9f0))),
        ],
        :gumbel => [
            (gxc_gumbel_obs, (), choicemap((:y, 0.7f0))),
            (gxc_gumbel_latent, (), choicemap((:y, 0.9f0))),
        ],
    )

    for (gxc_index, gxc_family) in enumerate(UncertainTea.GPU_BACKEND_SUPPORTED_DISTRIBUTIONS)
        @testset "gxc_$gxc_family" begin
            haskey(gxc_models, gxc_family) || error(
                "backend-native family :$gxc_family is not covered by " *
                "gradient_crosscheck.jl -- add a cross-check model for it",
            )
            for (gxc_entry_index, (gxc_model, gxc_args, gxc_constraints)) in
                enumerate(gxc_models[gxc_family])
                gxc_check(gxc_model, gxc_args, gxc_constraints, 1000 * gxc_index + gxc_entry_index)
            end
        end
    end

    # A latent vector binding consumed by a gradient-bearing expression must
    # NOT sit on the analytic tier: the binding is stored as a generic vector
    # with no per-row gradients, so the analytic path would silently treat the
    # latent as constant data in the likelihood (issue #49 review). The
    # support gate rejects these models to the per-column fallback, whose
    # gradients must still match finite differences.
    @tea static function gxc_lkjcholesky_bound_consumed()
        L ~ lkjcholesky(2, 2.0)
        {:y} ~ normal.(L, 1.0)
        return L
    end

    @tea static function gxc_dirichlet_bound_consumed()
        w ~ dirichlet([2.0, 3.0, 4.0])
        {:y} ~ normal.(w, 1.0)
        return w
    end

    # ------------------------------------------------------------------
    # Boundary regressions (issues #77 and #86): the manual analytic
    # gradients must stay finite and match the CPU/ForwardDiff path at
    # support/parameter boundaries, on the batched CPU backend (Float64 and
    # Float32) and on the device dual mirror.
    # ------------------------------------------------------------------

    @tea static function gxc_geometric_boundary(pv)
        x ~ normal(0.0, 1.0)
        {:y} ~ geometric(pv)
    end

    @tea static function gxc_negbin_boundary(pv)
        x ~ normal(0.0, 1.0)
        {:y} ~ negativebinomial(2.0, pv)
    end

    # latent-dependent probability: p = logistic(x + off) reaches exactly 1
    # at large off (the Float32 clamp/logistic pipeline rounds to 1 easily)
    @tea static function gxc_geometric_boundary_latent(off)
        x ~ normal(0.0, 1.0)
        probability = 1.0 / (1.0 + exp(-x - off))
        {:y} ~ geometric(probability)
    end

    @tea static function gxc_negbin_boundary_latent(off)
        x ~ normal(0.0, 1.0)
        probability = 1.0 / (1.0 + exp(-x - off))
        {:y} ~ negativebinomial(2.0, probability)
    end

    @testset "gxc_count_probability_boundary" begin
        gxc_b_cm = choicemap(:y => 0)
        gxc_b_point = reshape([0.0], 1, 1)
        for gxc_b_model in (gxc_geometric_boundary, gxc_negbin_boundary)
            # issue #77: p == 1 with count == 0 used to evaluate 0 / (1 - p)
            # and return NaN from the manual gradient; CPU gives 0.0
            for gxc_b_p in (1.0, 1.0 - 1e-12, 0.999)
                gxc_b_cpu = logjoint_gradient_unconstrained(gxc_b_model, [0.0], (gxc_b_p,), gxc_b_cm)
                gxc_b_bat =
                    batched_logjoint_gradient_unconstrained(gxc_b_model, gxc_b_point, (gxc_b_p,), gxc_b_cm)
                gxc_b_bat32 = batched_logjoint_gradient_unconstrained(
                    gxc_b_model, Float32.(gxc_b_point), (gxc_b_p,), gxc_b_cm,
                )
                @test !any(isnan, gxc_b_bat)
                @test gxc_b_bat[:, 1] ≈ gxc_b_cpu atol = 1e-12
                @test !any(isnan, gxc_b_bat32)
                @test Float64.(gxc_b_bat32[:, 1]) ≈ gxc_b_cpu atol = 1e-6
            end
            # the device dual mirror is NaN-free and matches at p == 1 too
            gxc_b_v, gxc_b_dev =
                device_batched_logjoint_gradient(gxc_b_model, gxc_b_point, (1.0,), gxc_b_cm)
            @test collect(gxc_b_dev)[:, 1] ≈ [0.0] atol = 1e-12
            gxc_b_v32, gxc_b_dev32 = device_batched_logjoint_gradient(
                gxc_b_model, Float32.(gxc_b_point), (1.0f0,), gxc_b_cm; precision=Float32,
            )
            @test !any(isnan, collect(gxc_b_dev32))
        end

        # latent-flowing probability: gradients must agree with CPU/ForwardDiff
        # near the boundary (p = 0.999 gives a nonzero derivative) and stay
        # finite when the logistic saturates to exactly 1 (off = 800)
        for gxc_b_model in (gxc_geometric_boundary_latent, gxc_negbin_boundary_latent)
            for gxc_b_off in (log(0.999 / 0.001), 30.0, 800.0)
                gxc_b_cpu = logjoint_gradient_unconstrained(gxc_b_model, [0.0], (gxc_b_off,), gxc_b_cm)
                gxc_b_bat = batched_logjoint_gradient_unconstrained(
                    gxc_b_model, gxc_b_point, (gxc_b_off,), gxc_b_cm,
                )
                @test !any(isnan, gxc_b_bat)
                @test gxc_b_bat[:, 1] ≈ gxc_b_cpu atol = 1e-12
                gxc_b_bat32 = batched_logjoint_gradient_unconstrained(
                    gxc_b_model, Float32.(gxc_b_point), (gxc_b_off,), gxc_b_cm,
                )
                @test !any(isnan, gxc_b_bat32)
            end
        end
    end

    @tea static function gxc_weibull_boundary(shp)
        x ~ normal(0.0, 1.0)
        s = exp(x)
        {:y} ~ weibull(shp, s)
    end

    @testset "gxc_weibull_zero_boundary" begin
        # issue #86: at x == 0 with shape == 1 the logpdf is -log(scale), so
        # the scale partial is -1/scale; the manual gradient used to drop it
        # via an early continue (CPU -1.0 vs batched 0.0)
        gxc_w_cm = choicemap(:y => 0)
        gxc_w_point = reshape([0.0], 1, 1)
        gxc_w_cpu = logjoint_gradient_unconstrained(gxc_weibull_boundary, [0.0], (1.0,), gxc_w_cm)
        @test gxc_w_cpu ≈ [-1.0] atol = 1e-12
        gxc_w_bat =
            batched_logjoint_gradient_unconstrained(gxc_weibull_boundary, gxc_w_point, (1.0,), gxc_w_cm)
        @test gxc_w_bat[:, 1] ≈ [-1.0] atol = 1e-12
        gxc_w_bat32 = batched_logjoint_gradient_unconstrained(
            gxc_weibull_boundary, Float32.(gxc_w_point), (1.0,), gxc_w_cm,
        )
        @test Float64.(gxc_w_bat32[:, 1]) ≈ [-1.0] atol = 1e-6

        # device dual mirror hits the same boundary value
        gxc_w_v, gxc_w_dev =
            device_batched_logjoint_gradient(gxc_weibull_boundary, gxc_w_point, (1.0,), gxc_w_cm)
        @test collect(gxc_w_dev)[:, 1] ≈ [-1.0] atol = 1e-12
        gxc_w_v32, gxc_w_dev32 = device_batched_logjoint_gradient(
            gxc_weibull_boundary, Float32.(gxc_w_point), (1.0f0,), gxc_w_cm; precision=Float32,
        )
        @test Float64.(collect(gxc_w_dev32)[:, 1]) ≈ [-1.0] atol = 1e-5

        # just inside the support the analytic formulas take over smoothly
        gxc_w_near_cm = choicemap(:y => 1e-8)
        gxc_w_near_cpu =
            logjoint_gradient_unconstrained(gxc_weibull_boundary, [0.0], (1.0,), gxc_w_near_cm)
        gxc_w_near_bat = batched_logjoint_gradient_unconstrained(
            gxc_weibull_boundary, gxc_w_point, (1.0,), gxc_w_near_cm,
        )
        @test gxc_w_near_bat[:, 1] ≈ gxc_w_near_cpu atol = 1e-10
        @test gxc_w_near_bat[:, 1] ≈ [-1.0] atol = 1e-6

        # shape != 1 at x == 0 keeps the zero contribution (logpdf is +-Inf,
        # matching the CPU path's zero derivative through the branch)
        for gxc_w_shape in (2.0, 0.5)
            gxc_w_off_cpu =
                logjoint_gradient_unconstrained(gxc_weibull_boundary, [0.0], (gxc_w_shape,), gxc_w_cm)
            gxc_w_off_bat = batched_logjoint_gradient_unconstrained(
                gxc_weibull_boundary, gxc_w_point, (gxc_w_shape,), gxc_w_cm,
            )
            @test !any(isnan, gxc_w_off_bat)
            @test gxc_w_off_bat[:, 1] ≈ gxc_w_off_cpu atol = 1e-12
        end
    end

    @testset "gxc_latent_vector_binding_fallback" begin
        for (gxc_fb_index, (gxc_fb_model, gxc_fb_constraints)) in enumerate([
            (gxc_lkjcholesky_bound_consumed, choicemap((:y, Float32[0.9, 0.3, 0.8]))),
            (gxc_dirichlet_bound_consumed, choicemap((:y, Float32[0.3, 0.3, 0.4]))),
        ])
            @test backend_report(gxc_fb_model).supported == true
            gxc_fb_rng = MersenneTwister(9000 + gxc_fb_index)
            gxc_fb_trace, _ = generate(gxc_fb_model, (), gxc_fb_constraints; rng=gxc_fb_rng)
            gxc_fb_base = transform_to_unconstrained(gxc_fb_trace)
            gxc_fb_points = gxc_fb_base .+ 0.4 .* randn(gxc_fb_rng, length(gxc_fb_base), 3)
            gxc_fb_cache = BatchedLogjointGradientCache(gxc_fb_model, gxc_fb_points, (), gxc_fb_constraints)
            @test isnothing(gxc_fb_cache.backend_cache)
            gxc_fb_gradient = batched_logjoint_gradient_unconstrained(gxc_fb_cache, gxc_fb_points)
            for i = 1:size(gxc_fb_points, 2)
                @test gxc_fb_gradient[:, i] ≈
                      gxc_fd_gradient(gxc_fb_model, gxc_fb_points[:, i], (), gxc_fb_constraints) atol = 5e-6
            end
        end
    end
end
