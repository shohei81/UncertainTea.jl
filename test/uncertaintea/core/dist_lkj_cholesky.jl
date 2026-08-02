# PR 49: LKJ prior over correlation Cholesky factors.
#
# Contract: `lkjcholesky(d, eta)` scores the column-major PACKED lower triangle
# (length d*(d+1)/2, diagonal included) of the Cholesky factor of a d x d
# correlation matrix, with density proportional to
# prod_{i=2..d} L[i,i]^(d - i + 2*eta - 2) over the free below-diagonal
# coordinates, normalized by the exact LKJ constant. Latents flow through
# `CholeskyCorrTransform` — Stan's canonical partial correlation map from
# d*(d-1)/2 unconstrained coordinates — whose logabsdetjac is the determinant of
# the below-diagonal map z -> (L[i,j])_{i>j} (the diagonal entries are pinned by
# the rows' unit norms). `d` must be a literal integer >= 2 for latents and
# observations alike. The family is CPU-reference only (honestly reported
# unsupported by the backend). `scale_cholesky(scales, packed)` un-packs the
# factor and scales row i by scales[i], producing the dense `scale_tril` for
# `mvnormaldense` — usable as a deterministic binding inside `@tea`.

const lkj_FD = UncertainTea.ForwardDiff
const lkj_LA = UncertainTea.LinearAlgebra
const lkj_loggamma = UncertainTea.loggamma

lkj_pack_index(d, i, j) = UncertainTea._packed_lower_index(d, i, j)

lkj_mean(xs) = sum(xs) / length(xs)
lkj_var(xs) = sum(abs2, xs .- lkj_mean(xs)) / (length(xs) - 1)

# Un-pack a column-major packed lower triangle into a dense d x d matrix.
function lkj_unpack(packed, d)
    full = zeros(d, d)
    for j = 1:d, i = j:d
        full[i, j] = packed[lkj_pack_index(d, i, j)]
    end
    return full
end

# --- transform round trip, unit rows, positive diagonal ----------------------
@testset "lkj_transform_roundtrip" begin
    lkj_rt_rng = MersenneTwister(4901)
    for lkj_rt_d in (2, 3, 4)
        lkj_rt_t = CholeskyCorrTransform(lkj_rt_d)
        for _ = 1:8
            lkj_rt_z = randn(lkj_rt_rng, (lkj_rt_d * (lkj_rt_d - 1)) ÷ 2) .* 1.5
            lkj_rt_packed = UncertainTea.to_constrained(lkj_rt_t, lkj_rt_z)
            @test length(lkj_rt_packed) == (lkj_rt_d * (lkj_rt_d + 1)) ÷ 2
            # every row of the full factor (diagonal included) is a unit vector
            for lkj_rt_i = 1:lkj_rt_d
                lkj_rt_row = sum(
                    lkj_rt_packed[lkj_pack_index(lkj_rt_d, lkj_rt_i, lkj_rt_j)]^2
                    for lkj_rt_j = 1:lkj_rt_i
                )
                @test abs(lkj_rt_row - 1) < 1e-12
                @test lkj_rt_packed[lkj_pack_index(lkj_rt_d, lkj_rt_i, lkj_rt_i)] > 0
            end
            lkj_rt_back = UncertainTea.to_unconstrained(lkj_rt_t, lkj_rt_packed)
            @test lkj_rt_back ≈ lkj_rt_z atol = 1e-9
        end
    end
    # d >= 2 is enforced by the transform constructor.
    @test_throws ArgumentError CholeskyCorrTransform(1)
end

# --- logabsdetjac == logabsdet of the below-diagonal Jacobian (d = 3) --------
@testset "lkj_logabsdet_forwarddiff" begin
    lkj_lad_t = CholeskyCorrTransform(3)
    lkj_lad_free = ((2, 1), (3, 1), (3, 2))
    # The forward map restricted to the free coordinates: R^3 -> R^3. The
    # diagonal entries are determined by the unit rows, so the authoritative
    # Jacobian is over the below-diagonal entries only.
    lkj_lad_g(z) = [
        UncertainTea.to_constrained(lkj_lad_t, z)[lkj_pack_index(3, ij[1], ij[2])]
        for ij in lkj_lad_free
    ]
    lkj_lad_rng = MersenneTwister(4902)
    for _ = 1:10
        lkj_lad_z = randn(lkj_lad_rng, 3) .* 1.5
        lkj_lad_J = lkj_FD.jacobian(lkj_lad_g, lkj_lad_z)
        lkj_lad_ref = lkj_LA.logabsdet(lkj_lad_J)[1]
        @test UncertainTea.logabsdetjac(lkj_lad_t, lkj_lad_z) ≈ lkj_lad_ref atol = 1e-9
    end
end

# --- d = 2 closed form: density, sampling moments, NUTS prior sanity ---------
@tea static function lkj_prior2_model()
    Omega ~ lkjcholesky(2, 2.0)
    return Omega
end

@testset "lkj_d2_closed_form" begin
    lkj_d2_eta = 2.0
    lkj_d2_dist = UncertainTea.lkjcholesky(2, lkj_d2_eta)

    # rho = L[2,1] has marginal density (1 - rho^2)^(eta-1) / (2^(2eta-1) B(eta, eta)):
    # the packed density evaluated on [1, rho, sqrt(1-rho^2)] must equal it exactly,
    # which validates the LKJ normalizing constant against the Beta closed form.
    for lkj_d2_rho in (-0.85, -0.3, 0.0, 0.45, 0.9)
        lkj_d2_packed = [1.0, lkj_d2_rho, sqrt(1 - lkj_d2_rho^2)]
        lkj_d2_ref =
            (lkj_d2_eta - 1) * log1p(-lkj_d2_rho^2) -
            ((2 * lkj_d2_eta - 1) * log(2.0) +
             2 * lkj_loggamma(lkj_d2_eta) - lkj_loggamma(2 * lkj_d2_eta))
        @test UncertainTea.logpdf(lkj_d2_dist, lkj_d2_packed) ≈ lkj_d2_ref atol = 1e-12
    end

    # Support checks: wrong length, non-positive diagonal, over-unit rows.
    @test UncertainTea.logpdf(lkj_d2_dist, [1.0, 0.5]) == -Inf
    @test UncertainTea.logpdf(lkj_d2_dist, [1.0, 0.5, -0.2]) == -Inf
    @test UncertainTea.logpdf(lkj_d2_dist, [1.0, 0.9, 0.9]) == -Inf
    @test_throws ArgumentError UncertainTea.lkjcholesky(1, 2.0)
    @test_throws ArgumentError UncertainTea.lkjcholesky(2, 0.0)

    # cvine sampling: rho ~ 2 * Beta(eta, eta) - 1, so E[rho] = 0 and
    # Var[rho] = 1 / (2*eta + 1).
    lkj_d2_rng = MersenneTwister(4903)
    lkj_d2_rhos = [rand(lkj_d2_rng, lkj_d2_dist)[2] for _ = 1:30000]
    lkj_d2_target_var = 1 / (2 * lkj_d2_eta + 1)
    @test abs(lkj_mean(lkj_d2_rhos)) < 0.01
    @test abs(lkj_var(lkj_d2_rhos) - lkj_d2_target_var) / lkj_d2_target_var < 0.05

    # NUTS on the bare prior recovers the same marginal moments through the
    # CholeskyCorrTransform + packed logpdf pair.
    lkj_d2_layout = parameterlayout(lkj_prior2_model)
    @test length(lkj_d2_layout.slots) == 1
    @test parametercount(lkj_d2_layout) == 1
    @test parametervaluecount(lkj_d2_layout) == 3
    @test lkj_d2_layout.slots[1].transform isa CholeskyCorrTransform

    lkj_d2_chain = first(nuts(
        lkj_prior2_model,
        (),
        choicemap();
        num_samples=400,
        num_warmup=400,
        rng=MersenneTwister(4904),
    ))
    @test all(isfinite, lkj_d2_chain.constrained_samples)
    lkj_d2_draws = lkj_d2_chain.constrained_samples[2, :]
    @test abs(lkj_mean(lkj_d2_draws)) < 0.1
    @test abs(lkj_var(lkj_d2_draws) - lkj_d2_target_var) < 0.05
end

# --- hierarchical covariance: lkjcholesky + iid lognormal + scale_cholesky ---
const lkj_h_zeros3 = [0.0, 0.0, 0.0]

@tea static function lkj_hier_model(zeros3_arg, n)
    Omega ~ lkjcholesky(3, 2.0)
    tau ~ iid(lognormal(0.0f0, 0.3f0), 3)
    Ltril = scale_cholesky(tau, Omega)
    for i = 1:n
        {:y => i} ~ mvnormaldense(zeros3_arg, Ltril)
    end
    return Omega
end

@testset "lkj_hierarchical_posterior" begin
    # Data generated from a known covariance with a strong positive (2,1)
    # correlation.
    lkj_h_Lcorr = [1.0 0.0 0.0; 0.75 sqrt(1 - 0.75^2) 0.0; 0.5 0.3 sqrt(1 - 0.5^2 - 0.3^2)]
    lkj_h_tau_true = [1.0, 1.2, 0.8]
    lkj_h_scale = Matrix(lkj_LA.Diagonal(lkj_h_tau_true) * lkj_h_Lcorr)
    lkj_h_n = 60
    lkj_h_rng = MersenneTwister(4905)
    lkj_h_dist = mvnormaldense(lkj_h_zeros3, lkj_h_scale)
    lkj_h_ys = [rand(lkj_h_rng, lkj_h_dist) for _ = 1:lkj_h_n]
    lkj_h_cm = choicemap((:y => i, lkj_h_ys[i]) for i = 1:lkj_h_n)

    # Layout: one packed correlation slot (3 params -> 6 values) and one iid
    # lognormal scale slot (3 params -> 3 values).
    lkj_h_layout = parameterlayout(lkj_hier_model)
    @test length(lkj_h_layout.slots) == 2
    @test parametercount(lkj_h_layout) == 6
    @test parametervaluecount(lkj_h_layout) == 9
    @test lkj_h_layout.slots[1].transform isa CholeskyCorrTransform

    # The compiled evaluator resolves `scale_cholesky` as a deterministic
    # binding: logjoint over constrained values matches a manual sum of
    # densities through the exact same helper.
    lkj_h_omega = rand(lkj_h_rng, UncertainTea.lkjcholesky(3, 2.0))
    lkj_h_tau = [0.9, 1.1, 1.0]
    lkj_h_params = vcat(lkj_h_omega, lkj_h_tau)
    lkj_h_manual =
        UncertainTea.logpdf(UncertainTea.lkjcholesky(3, 2.0), lkj_h_omega) +
        sum(UncertainTea.logpdf(lognormal(0.0f0, 0.3f0), lkj_h_tau[k]) for k = 1:3)
    lkj_h_Ltril = scale_cholesky(lkj_h_tau, lkj_h_omega)
    for i = 1:lkj_h_n
        lkj_h_manual += UncertainTea.logpdf(mvnormaldense(lkj_h_zeros3, lkj_h_Ltril), lkj_h_ys[i])
    end
    @test logjoint(lkj_hier_model, lkj_h_params, (lkj_h_zeros3, lkj_h_n), lkj_h_cm) ≈
          lkj_h_manual atol = 1e-8

    lkj_h_chains = nuts_chains(
        lkj_hier_model,
        (lkj_h_zeros3, lkj_h_n),
        lkj_h_cm;
        num_chains=2,
        num_samples=300,
        num_warmup=300,
        rng=MersenneTwister(4906),
    )
    for lkj_h_chain in lkj_h_chains.chains
        @test all(isfinite, lkj_h_chain.constrained_samples)
    end
    # The (1,1) entry of a Cholesky correlation factor is structurally 1.0, so
    # its Rhat is NaN by the Stan/posterior constant-chain convention (issue
    # #103); every genuinely sampled value must still mix well.
    lkj_h_rhats = rhat(lkj_h_chains)
    @test count(isnan, lkj_h_rhats) == 1
    @test all(<(1.3), filter(isfinite, lkj_h_rhats))
    # Posterior mean of the (2,1) correlation (packed value index 2) keeps the
    # strongly positive sign the data were generated with.
    lkj_h_rho = vcat((chain.constrained_samples[2, :] for chain in lkj_h_chains.chains)...)
    @test lkj_mean(lkj_h_rho) > 0.1
end

# --- literal-d rule: non-literal dimensions throw at macro expansion ---------
@testset "lkj_literal_d_rule" begin
    lkj_bad_latent = :(@tea static function lkj_bad_latent_model(d)
        Omega ~ lkjcholesky(d, 2.0)
        {:o} ~ normal(0.0, 1.0)
    end)
    @test_throws ArgumentError macroexpand(@__MODULE__, lkj_bad_latent)

    lkj_bad_observation = :(@tea static function lkj_bad_observation_model(d)
        mu ~ normal(0.0, 1.0)
        {:c} ~ lkjcholesky(d, 2.0)
    end)
    @test_throws ArgumentError macroexpand(@__MODULE__, lkj_bad_observation)

    lkj_bad_dimension = :(@tea static function lkj_bad_dimension_model()
        Omega ~ lkjcholesky(1, 2.0)
        {:o} ~ normal(0.0, 1.0)
    end)
    @test_throws ArgumentError macroexpand(@__MODULE__, lkj_bad_dimension)

    # eta stays a free expression; a literal d with a runtime eta expands fine
    # and the observation path scores the packed value.
    @tea static function lkj_eta_expr_model(eta_arg)
        {:c} ~ lkjcholesky(2, eta_arg)
    end
    lkj_eta_packed = [1.0, 0.4, sqrt(1 - 0.16)]
    @test logjoint(lkj_eta_expr_model, Float64[], (1.5,), choicemap((:c, lkj_eta_packed))) ≈
          UncertainTea.logpdf(UncertainTea.lkjcholesky(2, 1.5), lkj_eta_packed) atol = 1e-12
end

# --- backend-native lowering and batched scoring (issue #49) -----------------
@tea static function lkj_obs_model()
    {:c} ~ lkjcholesky(2, 1.5)
end

@testset "lkj_backend_report" begin
    lkj_report = backend_report(lkj_prior2_model)
    @test lkj_report.supported == true
    @test isempty(lkj_report.issues)
    @test :lkjcholesky in lkj_report.supported_families
    lkj_backend_plan = backend_execution_plan(lkj_prior2_model)
    @test lkj_backend_plan.steps[1] isa UncertainTea.BackendLKJCholeskyChoicePlanStep
    @test lkj_backend_plan.steps[1].d == 2
    @test lkj_backend_plan.steps[1].value_length == 3

    # batched scoring of a latent slot matches the per-column CPU reference
    lkj_batch_unconstrained = reshape([-0.5, 0.0, 0.7], 1, 3)
    lkj_batch_values = batched_logjoint_unconstrained(
        lkj_prior2_model,
        lkj_batch_unconstrained,
        (),
        choicemap(),
    )
    @test lkj_batch_values ≈ [
        logjoint_unconstrained(lkj_prior2_model, lkj_batch_unconstrained[:, index], (), choicemap()) for index = 1:3
    ] atol = 1e-12

    # observed packed values score the compiled density, including the -Inf
    # support rejections (non-positive diagonal, row norm above one)
    lkj_obs_good = [1.0, 0.4, sqrt(1 - 0.16)]
    lkj_obs_bad_diagonal = [1.0, 0.4, -sqrt(1 - 0.16)]
    lkj_obs_bad_norm = [1.0, 0.9, 0.9]
    lkj_obs_values = UncertainTea.batched_logjoint(
        lkj_obs_model,
        zeros(0, 3),
        (),
        [
            choicemap((:c, lkj_obs_good)),
            choicemap((:c, lkj_obs_bad_diagonal)),
            choicemap((:c, lkj_obs_bad_norm)),
        ],
    )
    @test lkj_obs_values[1] ≈ UncertainTea.logpdf(UncertainTea.lkjcholesky(2, 1.5), lkj_obs_good) atol = 1e-12
    @test lkj_obs_values[2] == -Inf
    @test lkj_obs_values[3] == -Inf
end

# --- scale_cholesky: diag(scales) * unpack(L), AD-friendly -------------------
@testset "lkj_scale_cholesky" begin
    lkj_sc_rng = MersenneTwister(4907)
    for lkj_sc_d in (2, 3, 4)
        lkj_sc_packed = rand(lkj_sc_rng, UncertainTea.lkjcholesky(lkj_sc_d, 1.5))
        lkj_sc_scales = 0.5 .+ rand(lkj_sc_rng, lkj_sc_d)
        lkj_sc_result = scale_cholesky(lkj_sc_scales, lkj_sc_packed)
        lkj_sc_manual = lkj_LA.Diagonal(lkj_sc_scales) * lkj_unpack(lkj_sc_packed, lkj_sc_d)
        @test size(lkj_sc_result) == (lkj_sc_d, lkj_sc_d)
        @test maximum(abs.(lkj_sc_result - lkj_sc_manual)) < 1e-12
        @test lkj_LA.istril(lkj_sc_result)
    end
    # Dimension mismatches throw.
    @test_throws DimensionMismatch scale_cholesky([1.0, 2.0], [1.0, 0.5, 0.5, 1.0])
    @test_throws ArgumentError scale_cholesky(Float64[], Float64[])
    # ForwardDiff flows through the plain loops (gradient of an entry w.r.t.
    # the scales is the un-packed factor entry).
    lkj_sc_g = lkj_FD.gradient(
        s -> scale_cholesky(s, [1.0, 0.6, 0.8])[2, 1],
        [2.0, 3.0],
    )
    @test lkj_sc_g ≈ [0.0, 0.6] atol = 1e-12
end

# --- tanh saturation: -Inf rejection instead of DomainError (issue #99) ------
# Unconstrained space is all of R^n; when tanh(z) saturates to exactly +-1 a
# row's square sum can reach (or pass) 1 and the transform must reject with a
# -Inf log-abs-det like the device kernel, never throw.
@testset "lkj_tanh_saturation" begin
    @tea function lkj_sat_model()
        L ~ lkjcholesky(3, 1.5)
        {:y} ~ normal(L[2], 1.0)
    end
    lkj_sat_z = [0.0, -0.1636303098944617, 21.76268802388644]
    lkj_sat_obs = choicemap(:y => 0.2)
    @test logjoint_unconstrained(lkj_sat_model, lkj_sat_z, (), lkj_sat_obs) == -Inf
    lkj_sat_g = logjoint_gradient_unconstrained(lkj_sat_model, lkj_sat_z, (), lkj_sat_obs)
    @test length(lkj_sat_g) == 3
    # Here saturation lands on the diagonal: the entry clamps to 0 (rejected by
    # the lkj density above), the log-abs-det of the below-diagonal map stays
    # finite, and nothing throws — exactly the device kernel's contract.
    lkj_sat_packed = UncertainTea.to_constrained(CholeskyCorrTransform(3), lkj_sat_z)
    @test all(isfinite, lkj_sat_packed)
    @test lkj_sat_packed[lkj_pack_index(3, 3, 3)] == 0.0
    @test isfinite(UncertainTea.logabsdetjac(CholeskyCorrTransform(3), lkj_sat_z))
    # Saturating a below-diagonal step degrades the log-abs-det itself to -Inf.
    @test UncertainTea.logabsdetjac(CholeskyCorrTransform(3), [0.0, 21.0, 0.5]) == -Inf

    # d = 5 variant (the log1p(-1.0000000000000002) crash from the issue).
    lkj_sat_t5 = CholeskyCorrTransform(5)
    lkj_sat_z5 = fill(25.0, 10)
    lkj_sat_dest5 = zeros(15)
    lkj_sat_lad5 = UncertainTea._to_constrained_cholesky_corr!(lkj_sat_dest5, lkj_sat_t5, lkj_sat_z5)
    @test lkj_sat_lad5 == -Inf
    @test all(isfinite, lkj_sat_dest5)

    # Fuzz: extreme-|z| draws (scales up to 40) never throw; the log-abs-det is
    # finite or -Inf and the packed factor stays finite.
    lkj_sat_rng = MersenneTwister(4910)
    lkj_sat_bad = 0
    for lkj_sat_trial = 1:2000
        lkj_sat_d = isodd(lkj_sat_trial) ? 3 : 5
        lkj_sat_scale = 40.0 * rand(lkj_sat_rng)
        lkj_sat_draw = lkj_sat_scale .* randn(lkj_sat_rng, (lkj_sat_d * (lkj_sat_d - 1)) ÷ 2)
        lkj_sat_dest = zeros((lkj_sat_d * (lkj_sat_d + 1)) ÷ 2)
        lkj_sat_lad = UncertainTea._to_constrained_cholesky_corr!(
            lkj_sat_dest,
            CholeskyCorrTransform(lkj_sat_d),
            lkj_sat_draw,
        )
        (lkj_sat_lad == -Inf || isfinite(lkj_sat_lad)) && all(isfinite, lkj_sat_dest) ||
            (lkj_sat_bad += 1)
    end
    @test lkj_sat_bad == 0
end

# Issue #78: a packed factor is only in support when every row has UNIT norm
# within tolerance (a Cholesky factor of a correlation matrix); factors with
# shorter rows used to score finitely. CPU, backend, and device paths agree.
@testset "lkj_unit_row_support" begin
    lkj_short_row = [1.0, 0.0, 0.5] # second row has norm 0.5
    lkj_unit_rows = [1.0, 0.6, 0.8]
    lkj_short_diag = [0.9, 0.6, 0.8] # first row has norm 0.9

    @test UncertainTea.logpdf(lkjcholesky(2, 1.0), lkj_short_row) == -Inf
    @test UncertainTea.logpdf(lkjcholesky(2, 1.0), lkj_short_diag) == -Inf
    @test UncertainTea.logpdf(lkjcholesky(2, 1.0), lkj_unit_rows) ≈ -log(2.0) atol = 1e-12

    @test UncertainTea._backend_lkjcholesky_logpdf(2, 1.0, lkj_short_row) == -Inf
    @test UncertainTea._backend_lkjcholesky_logpdf(2, 1.0, lkj_short_diag) == -Inf
    @test UncertainTea._backend_lkjcholesky_logpdf(2, 1.0, lkj_unit_rows) ≈ -log(2.0) atol = 1e-12

    @test UncertainTea._device_lkjcholesky_logpdf(1.0, (lkj_short_row...,), Val(2)) == -Inf
    @test UncertainTea._device_lkjcholesky_logpdf(1.0, (lkj_short_diag...,), Val(2)) == -Inf
    @test UncertainTea._device_lkjcholesky_logpdf(1.0, (lkj_unit_rows...,), Val(2)) ≈ -log(2.0) atol = 1e-8

    # sampler output and transform output stay in support
    lkj_unit_dist = lkjcholesky(3, 1.5)
    lkj_unit_packed = rand(MersenneTwister(4), lkj_unit_dist)
    @test isfinite(UncertainTea.logpdf(lkj_unit_dist, lkj_unit_packed))
end
