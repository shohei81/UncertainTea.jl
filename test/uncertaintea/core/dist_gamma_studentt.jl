@testset "dist_gamma_studentt" begin
    @test UncertainTea.logpdf(gamma(2.0, 3.0), 1.5) ≈
          2.0 * log(3.0) - UncertainTea.loggamma(2.0) + log(1.5) - 4.5 atol=1e-8
    @test UncertainTea.logpdf(gamma(2.0, 3.0), 0.0) == -Inf
    @test UncertainTea.logpdf(studentt(5.0, 0.0, 1.0), 0.25) ≈
          UncertainTea.loggamma(3.0) - UncertainTea.loggamma(2.5) -
          (log(5.0) + log(pi)) / 2 -
          3.0 * log1p(0.25^2 / 5.0) atol=1e-8

    @tea static function gamma_latent_model()
        rate ~ gamma(2.5f0, 1.5f0)
        {:y} ~ normal(rate, 0.5f0)
        return rate
    end

    gamma_latent_constraints = choicemap((:y, 0.9f0))
    gamma_latent_trace, _ = generate(
        gamma_latent_model,
        (),
        gamma_latent_constraints;
        rng=MersenneTwister(130),
    )
    gamma_latent_spec = modelspec(gamma_latent_model)
    gamma_latent_plan = executionplan(gamma_latent_model)
    gamma_latent_backend_plan = backend_execution_plan(gamma_latent_model)
    gamma_latent_params = parameter_vector(gamma_latent_trace)
    gamma_latent_unconstrained = transform_to_unconstrained(gamma_latent_trace)

    @test gamma_latent_spec.choices[1].rhs.family == :gamma
    @test gamma_latent_spec.parameter_layout.slots[1].transform isa LogTransform
    @test gamma_latent_plan.steps[1].parameter_slot == 1
    @test gamma_latent_backend_plan.steps[1] isa UncertainTea.BackendGammaChoicePlanStep
    @test gamma_latent_params[1] > 0
    @test gamma_latent_unconstrained[1] ≈ log(gamma_latent_params[1])
    @test logjoint(gamma_latent_model, gamma_latent_params, (), gamma_latent_constraints) ≈
          assess(
        gamma_latent_model,
        (),
        choicemap((:rate, gamma_latent_trace[:rate]), (:y, 0.9f0)),
    ) atol=1e-6
    @test logjoint_unconstrained(
        gamma_latent_model,
        gamma_latent_unconstrained,
        (),
        gamma_latent_constraints,
    ) ≈ logjoint(gamma_latent_model, gamma_latent_params, (), gamma_latent_constraints) +
          gamma_latent_unconstrained[1] atol=1e-6

    @tea static function studentt_latent_model()
        state ~ studentt(6.0f0, 0.0f0, 1.0f0)
        {:y} ~ normal(state, 0.5f0)
        return state
    end

    studentt_latent_constraints = choicemap((:y, -0.2f0))
    studentt_latent_trace, _ = generate(
        studentt_latent_model,
        (),
        studentt_latent_constraints;
        rng=MersenneTwister(131),
    )
    studentt_latent_spec = modelspec(studentt_latent_model)
    studentt_latent_plan = executionplan(studentt_latent_model)
    studentt_latent_backend_plan = backend_execution_plan(studentt_latent_model)
    studentt_latent_params = parameter_vector(studentt_latent_trace)
    studentt_latent_unconstrained = transform_to_unconstrained(studentt_latent_trace)

    @test studentt_latent_spec.choices[1].rhs.family == :studentt
    @test studentt_latent_spec.parameter_layout.slots[1].transform isa IdentityTransform
    @test studentt_latent_plan.steps[1].parameter_slot == 1
    @test studentt_latent_backend_plan.steps[1] isa UncertainTea.BackendStudentTChoicePlanStep
    @test studentt_latent_unconstrained ≈ studentt_latent_params atol=1e-8
    @test logjoint(studentt_latent_model, studentt_latent_params, (), studentt_latent_constraints) ≈
          assess(
        studentt_latent_model,
        (),
        choicemap((:state, studentt_latent_trace[:state]), (:y, -0.2f0)),
    ) atol=1e-6
    @test logjoint_unconstrained(
        studentt_latent_model,
        studentt_latent_unconstrained,
        (),
        studentt_latent_constraints,
    ) ≈ logjoint(studentt_latent_model, studentt_latent_params, (), studentt_latent_constraints) atol=1e-6

    @tea static function gamma_shape_model()
        log_shape ~ normal(0.0f0, 0.4f0)
        shape = exp(log_shape)
        {:y} ~ gamma(shape, 2.0f0)
        return shape
    end

    gamma_shape_constraints = choicemap((:y, 1.2f0))
    gamma_shape_trace, _ = generate(gamma_shape_model, (), gamma_shape_constraints; rng=MersenneTwister(132))
    gamma_shape_backend_plan = backend_execution_plan(gamma_shape_model)
    gamma_shape_params = parameter_vector(gamma_shape_trace)
    gamma_shape_batch_params = reshape(gamma_shape_params .+ Float64[-0.2, 0.0, 0.15], 1, 3)
    gamma_shape_batch_constraints = [
        choicemap((:y, 0.8f0)),
        choicemap((:y, 1.2f0)),
        choicemap((:y, 1.5f0)),
    ]
    gamma_shape_batch_gradient = batched_logjoint_gradient_unconstrained(
        gamma_shape_model,
        gamma_shape_batch_params,
        (),
        gamma_shape_batch_constraints,
    )
    gamma_shape_batch_cache = BatchedLogjointGradientCache(
        gamma_shape_model,
        gamma_shape_batch_params,
        (),
        gamma_shape_batch_constraints,
    )

    @test gamma_shape_backend_plan.steps[3] isa UncertainTea.BackendGammaChoicePlanStep
    @test gamma_shape_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                gamma_shape_model,
                gamma_shape_batch_params[:, index],
                (),
                gamma_shape_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(gamma_shape_batch_cache.backend_cache)
    @test isnothing(gamma_shape_batch_cache.flat_cache)
    @test isempty(gamma_shape_batch_cache.column_caches)

    @tea static function studentt_location_model()
        mu ~ normal(0.0f0, 1.0f0)
        {:y} ~ studentt(7.0f0, mu, 1.0f0)
        return mu
    end

    studentt_location_constraints = choicemap((:y, -0.35f0))
    studentt_location_trace, _ = generate(
        studentt_location_model,
        (),
        studentt_location_constraints;
        rng=MersenneTwister(133),
    )
    studentt_location_backend_plan = backend_execution_plan(studentt_location_model)
    studentt_location_params = parameter_vector(studentt_location_trace)
    studentt_location_batch_params = reshape(studentt_location_params .+ Float64[-0.1, 0.0, 0.2], 1, 3)
    studentt_location_batch_constraints = [
        choicemap((:y, -0.7f0)),
        choicemap((:y, -0.35f0)),
        choicemap((:y, 0.1f0)),
    ]
    studentt_location_batch_gradient = batched_logjoint_gradient_unconstrained(
        studentt_location_model,
        studentt_location_batch_params,
        (),
        studentt_location_batch_constraints,
    )
    studentt_location_batch_cache = BatchedLogjointGradientCache(
        studentt_location_model,
        studentt_location_batch_params,
        (),
        studentt_location_batch_constraints,
    )

    @test studentt_location_backend_plan.steps[2] isa UncertainTea.BackendStudentTChoicePlanStep
    @test studentt_location_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                studentt_location_model,
                studentt_location_batch_params[:, index],
                (),
                studentt_location_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(studentt_location_batch_cache.backend_cache)
    @test isnothing(studentt_location_batch_cache.flat_cache)
    @test isempty(studentt_location_batch_cache.column_caches)

    @tea static function studentt_scale_model()
        s ~ normal(0.0f0, 0.3f0)
        {:y} ~ studentt(7.0f0, 0.5f0, exp(s))
        return s
    end

    studentt_scale_batch_params = reshape(Float64[-0.4, 0.1, 0.6], 1, 3)
    studentt_scale_batch_constraints = [
        choicemap((:y, -0.9f0)),
        choicemap((:y, 0.5f0)),
        choicemap((:y, 2.3f0)),
    ]
    studentt_scale_batch_gradient = batched_logjoint_gradient_unconstrained(
        studentt_scale_model,
        studentt_scale_batch_params,
        (),
        studentt_scale_batch_constraints,
    )
    studentt_scale_batch_cache = BatchedLogjointGradientCache(
        studentt_scale_model,
        studentt_scale_batch_params,
        (),
        studentt_scale_batch_constraints,
    )
    @test studentt_scale_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                studentt_scale_model,
                studentt_scale_batch_params[:, index],
                (),
                studentt_scale_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(studentt_scale_batch_cache.backend_cache)
    @test isnothing(studentt_scale_batch_cache.flat_cache)
    @test isempty(studentt_scale_batch_cache.column_caches)

    @tea static function studentt_dof_model()
        t ~ normal(0.0f0, 0.3f0)
        {:y} ~ studentt(2.0f0 + exp(t), 0.0f0, 1.5f0)
        return t
    end

    studentt_dof_batch_params = reshape(Float64[-0.5, 0.0, 0.7], 1, 3)
    studentt_dof_batch_constraints = [
        choicemap((:y, -1.4f0)),
        choicemap((:y, 0.3f0)),
        choicemap((:y, 3.1f0)),
    ]
    studentt_dof_batch_gradient = batched_logjoint_gradient_unconstrained(
        studentt_dof_model,
        studentt_dof_batch_params,
        (),
        studentt_dof_batch_constraints,
    )
    studentt_dof_batch_cache = BatchedLogjointGradientCache(
        studentt_dof_model,
        studentt_dof_batch_params,
        (),
        studentt_dof_batch_constraints,
    )
    @test studentt_dof_batch_gradient ≈ hcat(
        [
            logjoint_gradient_unconstrained(
                studentt_dof_model,
                studentt_dof_batch_params[:, index],
                (),
                studentt_dof_batch_constraints[index],
            ) for index = 1:3
        ]...,
    ) atol=1e-8
    @test !isnothing(studentt_dof_batch_cache.backend_cache)
    @test isnothing(studentt_dof_batch_cache.flat_cache)
    @test isempty(studentt_dof_batch_cache.column_caches)
end

# --- issue #53: the Float32 large-nu normalizing constant ------------------
# loggamma((nu+1)/2) - loggamma(nu/2) differences two ~nu*log(nu)-sized values;
# at Float32 with nu = 1e5 that lost ~0.03 absolute. The constant is now
# computed in (at least) Float64 and narrowed, leaving only representation
# rounding, on both the compiled and the backend-native paths.
@testset "studentt_f32_large_nu_constant" begin
    reference = UncertainTea.logpdf(studentt(1.0e5, 0.0, 1.0), 15.2)
    lp32 = UncertainTea.logpdf(studentt(1.0f5, 0.0f0, 1.0f0), 15.2f0)
    @test abs(Float64(lp32) - reference) < 1e-3
    backend32 = UncertainTea._backend_studentt_logpdf(1.0f5, 0.0f0, 1.0f0, 15.2f0)
    @test abs(Float64(backend32) - reference) < 1e-3
    # Float64 results are unchanged bit-for-bit relative to the plain formula
    plain =
        UncertainTea.loggamma(3.0) - UncertainTea.loggamma(2.5) -
        (log(5.0) + log(pi)) / 2 - log(1.2) -
        6.0 * log1p(((0.9 - 0.3) / 1.2)^2 / 5.0) / 2
    @test UncertainTea.logpdf(studentt(5.0, 0.3, 1.2), 0.9) == plain

    # the analytic dnu term must use the same widened computation the
    # ForwardDiff reference differentiates (codex review): with a
    # parameter-dependent large nu at Float32, the Float32 digamma difference
    # would otherwise diverge materially from the Float64-widened value path
    @tea static function studentt_f32_latent_nu_model()
        t ~ normal(10.0f0, 1.0f0)
        {:y} ~ studentt(2.0f0 + exp(t), 0.0f0, 1.0f0)
        return t
    end
    latent_nu_cm = choicemap((:y, 0.7f0))
    latent_nu_params = reshape(Float32[10.0, 11.0], 1, 2) # nu ~ 2 + e^10
    latent_nu_g32 =
        batched_logjoint_gradient_unconstrained(studentt_f32_latent_nu_model, latent_nu_params, (), latent_nu_cm)
    latent_nu_fd = hcat(
        [
            UncertainTea.ForwardDiff.gradient(
                t -> logjoint_unconstrained(studentt_f32_latent_nu_model, t, (), latent_nu_cm),
                Float64.(latent_nu_params[:, index]),
            ) for index = 1:2
        ]...,
    )
    @test all(
        isapprox(Float64(a), b; rtol=2e-3, atol=1e-4) for (a, b) in zip(vec(latent_nu_g32), vec(latent_nu_fd))
    )
end

# --- issue #345: studentt at extreme nu + gamma at the exp-subnormal boundary -

# The loggamma((nu+1)/2) - loggamma(nu/2) difference cancels at Float64 for
# extreme nu (err 0.228 at nu = 1e14, wrong sign past 1e15); the constant now
# switches to the Stirling-ratio asymptotic series above nu = 1e3. Pin the full
# logpdf against 256-bit BigFloat across the warmup-excursion range.
@testset "studentt_extreme_nu_345" begin
    setprecision(BigFloat, 256) do
        st345_ref =
            (nu, x) -> Float64(
                UncertainTea.loggamma((big(nu) + 1) / 2) - UncertainTea.loggamma(big(nu) / 2) -
                (log(big(nu)) + log(big(pi))) / 2 -
                (big(nu) + 1) / 2 * log1p(big(x)^2 / big(nu)),
            )
        for st345_nu in (1.0e6, 1.0e8, 1.0e12, 1.0e15, 1.0e16), st345_x in (0.0, 1.3, -2.7)
            st345_expected = st345_ref(st345_nu, st345_x)
            @test UncertainTea.logpdf(studentt(st345_nu, 0.0, 1.0), st345_x) ≈ st345_expected rtol = 1e-10
            @test UncertainTea._backend_studentt_logpdf(st345_nu, 0.0, 1.0, st345_x) ≈ st345_expected rtol =
                1e-10
        end
    end

    # the analytic dnu helper must equal what ForwardDiff extracts from the
    # value branch on both sides of the crossover (the batched analytic
    # gradient and the single dual path share these)
    for st345_nu in (999.0, 1000.0, 1.0e8, 1.0e15)
        @test UncertainTea._studentt_log_constant_dnu(st345_nu) ≈
              UncertainTea.ForwardDiff.derivative(UncertainTea._studentt_log_constant, st345_nu) rtol = 1e-8
    end

    # the exact and asymptotic branches agree at the crossover (no logpdf jump
    # an integrator could see)
    @test UncertainTea._studentt_log_constant(prevfloat(1000.0)) ≈
          UncertainTea._studentt_log_constant(1000.0) atol = 1e-12

    # the truncated-t density path reuses the same constant: light-tail
    # normalizers stay finite and accurate at extreme nu
    @test isfinite(UncertainTea._std_t_log_pdf(0.5, 1.0e15))
end

# exp(theta) lands in the subnormal range for theta in ~(-745.1, -708.4); the
# surviving couple of mantissa bits made the gamma logpdf silently wrong by
# O(1) (0.56 nats at theta = -745) with an Inf gradient. Boundary semantics
# now: -Inf value with a non-finite (poisoned) gradient on the single AND
# batched paths, consistent with the issue-#343 `_offsupport_neginf` machinery.
@testset "gamma_exp_subnormal_boundary_345" begin
    # observed subnormal values are boundary points
    @test UncertainTea.logpdf(gamma(2.0, 1.0), 5.0e-324) == -Inf
    @test UncertainTea.logpdf(gamma(2.0, 1.0), prevfloat(floatmin(Float64))) == -Inf
    @test UncertainTea._backend_gamma_logpdf(0.5, 1.0, 1.0e-320) == -Inf
    # the smallest NORMAL value still scores (finite, very negative)
    @test isfinite(UncertainTea.logpdf(gamma(2.0, 1.0), floatmin(Float64)))

    @tea static function g345_gamma_model()
        x ~ gamma(2.0, 1.0)
    end
    for g345_theta in (-745.0, -720.0)
        @test logjoint_unconstrained(g345_gamma_model, [g345_theta], ()) == -Inf
        g345_g = logjoint_gradient_unconstrained(g345_gamma_model, [g345_theta], ())
        @test !isfinite(g345_g[1])
        @test batched_logjoint_unconstrained(g345_gamma_model, reshape([g345_theta], 1, 1), ())[1] == -Inf
        g345_bg =
            batched_logjoint_gradient_unconstrained(g345_gamma_model, reshape([g345_theta], 1, 1), ())
        @test !isfinite(g345_bg[1, 1])
    end

    # just above the subnormal band everything stays finite and consistent
    g345_ok = logjoint_unconstrained(g345_gamma_model, [-700.0], ())
    @test isfinite(g345_ok)
    g345_ok_g = logjoint_gradient_unconstrained(g345_gamma_model, [-700.0], ())
    @test isfinite(g345_ok_g[1])
end
