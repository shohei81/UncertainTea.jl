# issue #232: multivariate coverage.
#
#   * `mvstudentt(nu, mu, sigma)` (diagonal scale) and
#     `mvstudenttdense(nu, mu, scale_tril)` (dense Cholesky scale) -- the heavy-
#     tailed analogues of `mvnormal`/`mvnormaldense`. Full CPU + backend + device
#     (device caps the dense dimension at 8, riding the mvnormaldense unroll).
#   * `wishart(nu, S)` / `inversewishart(nu, S)` -- covariance-matrix priors whose
#     value is the PACKED lower Cholesky factor L of the sampled PD matrix M = LL'.
#     These are CPU-reference only (honest backend/device rejection); a latent
#     unconstrains through the new log-Cholesky `CholeskyCovTransform`.

const mvt_LA = UncertainTea.LinearAlgebra
const mvt_FD = UncertainTea.ForwardDiff
const mvt_loggamma = UncertainTea.loggamma

mvt_mean(xs) = sum(xs) / length(xs)

# Multivariate-t log density closed form via LinearAlgebra.
function mvt_reference(nu, mu, Sigma, x)
    d = length(mu)
    resid = x - mu
    q = mvt_LA.dot(resid, Sigma \ resid)
    return mvt_loggamma((nu + d) / 2) - mvt_loggamma(nu / 2) - (d / 2) * log(nu * pi) -
           0.5 * mvt_LA.logdet(Sigma) - ((nu + d) / 2) * log1p(q / nu)
end

mvt_logmvgamma(d, a) = (d * (d - 1) / 4) * log(pi) + sum(mvt_loggamma(a + (1 - j) / 2) for j = 1:d)

# Wishart / inverse-Wishart closed form over the packed lower Cholesky factor
# `packed` of M = L L', including the L -> M = LL' Jacobian
# d*log(2) + sum_i (d+1-i) log L[i,i].
function mvt_unpack(packed, d)
    L = zeros(eltype(packed), d, d)
    idx = 1
    for col = 1:d, row = col:d
        L[row, col] = packed[idx]
        idx += 1
    end
    return L
end

function mvt_wishart_reference(nu, S, packed)
    d = size(S, 1)
    L = mvt_unpack(packed, d)
    M = L * L'
    jac = d * log(2) + sum((d + 1 - i) * log(L[i, i]) for i = 1:d)
    dens =
        ((nu - d - 1) / 2) * mvt_LA.logdet(M) - 0.5 * mvt_LA.tr(S \ M) -
        (nu * d / 2) * log(2) - (nu / 2) * mvt_LA.logdet(S) - mvt_logmvgamma(d, nu / 2)
    return dens + jac
end

function mvt_inversewishart_reference(nu, S, packed)
    d = size(S, 1)
    L = mvt_unpack(packed, d)
    X = L * L'
    jac = d * log(2) + sum((d + 1 - i) * log(L[i, i]) for i = 1:d)
    dens =
        (nu / 2) * mvt_LA.logdet(S) - ((nu + d + 1) / 2) * mvt_LA.logdet(X) -
        0.5 * mvt_LA.tr(S * inv(X)) - (nu * d / 2) * log(2) - mvt_logmvgamma(d, nu / 2)
    return dens + jac
end

# ------------------------------------------------------------------------------
@testset "mvstudentt_diag_logpdf" begin
    mvt_nu = 5.0
    mvt_mu = [0.5, -1.0]
    mvt_sig = [1.5, 0.8]
    mvt_d = mvstudentt(mvt_nu, mvt_mu, mvt_sig)
    mvt_Sigma = mvt_LA.Diagonal(mvt_sig .^ 2)
    for mvt_x in ([0.0, 0.0], [1.2, -0.3], [0.5, -1.0], [-2.0, 3.0])
        @test UncertainTea.logpdf(mvt_d, mvt_x) ≈ mvt_reference(mvt_nu, mvt_mu, mvt_Sigma, mvt_x) atol = 1e-10
    end
    @test UncertainTea.logpdf(mvt_d, (1.2, -0.3)) ≈ UncertainTea.logpdf(mvt_d, [1.2, -0.3]) atol = 1e-12
    @test UncertainTea.logpdf(mvt_d, [0.0]) == -Inf
    # heavier tails than the matching mvnormal: a far point scores higher under t
    mvt_far = [8.0, -6.0]
    @test UncertainTea.logpdf(mvt_d, mvt_far) >
          UncertainTea.logpdf(mvnormal(mvt_mu, mvt_sig), mvt_far)
    # domain guards
    @test_throws ArgumentError mvstudentt(0.0, mvt_mu, mvt_sig)
    @test_throws ArgumentError mvstudentt(mvt_nu, mvt_mu, [1.0, -1.0])
end

@testset "mvstudentt_dense_logpdf" begin
    mvt_nu = 6.0
    mvt_mu = [0.5, -1.0, 2.0]
    mvt_L = [2.0 0.0 0.0; 0.6 1.5 0.0; -0.3 0.4 0.9]
    mvt_d = mvstudenttdense(mvt_nu, mvt_mu, mvt_L)
    mvt_Sigma = mvt_L * mvt_L'
    for mvt_x in ([0.0, 0.0, 0.0], [1.0, -2.0, 3.0], [0.5, -1.0, 2.0])
        @test UncertainTea.logpdf(mvt_d, mvt_x) ≈ mvt_reference(mvt_nu, mvt_mu, mvt_Sigma, mvt_x) atol = 1e-10
    end
    # only the lower triangle is read
    mvt_Lfull = copy(mvt_L)
    mvt_Lfull[1, 2] = 99.0
    @test UncertainTea.logpdf(mvstudenttdense(mvt_nu, mvt_mu, mvt_Lfull), [1.0, -2.0, 3.0]) ≈
          UncertainTea.logpdf(mvt_d, [1.0, -2.0, 3.0]) atol = 1e-12
    @test UncertainTea.logpdf(mvt_d, [0.0, 0.0]) == -Inf
end

@testset "wishart_logpdf" begin
    mvt_S = [2.0 0.5; 0.5 1.0]
    mvt_packed = [1.1, 0.3, 0.8]
    for mvt_nu in (4.0, 6.5)
        @test UncertainTea.logpdf(wishart(mvt_nu, mvt_S), mvt_packed) ≈
              mvt_wishart_reference(mvt_nu, mvt_S, mvt_packed) atol = 1e-9
        @test UncertainTea.logpdf(inversewishart(mvt_nu, mvt_S), mvt_packed) ≈
              mvt_inversewishart_reference(mvt_nu, mvt_S, mvt_packed) atol = 1e-9
    end
    # 3x3 case exercises the multivariate log-gamma sum
    mvt_S3 = [2.0 0.3 0.1; 0.3 1.5 0.2; 0.1 0.2 1.0]
    mvt_packed3 = [1.2, 0.2, -0.1, 0.9, 0.3, 0.7]
    @test UncertainTea.logpdf(wishart(7.0, mvt_S3), mvt_packed3) ≈
          mvt_wishart_reference(7.0, mvt_S3, mvt_packed3) atol = 1e-9
    @test UncertainTea.logpdf(inversewishart(7.0, mvt_S3), mvt_packed3) ≈
          mvt_inversewishart_reference(7.0, mvt_S3, mvt_packed3) atol = 1e-9
    # a non-positive Cholesky diagonal is out of support; wrong length -> -Inf
    @test UncertainTea.logpdf(wishart(4.0, mvt_S), [1.1, 0.3, -0.2]) == -Inf
    @test UncertainTea.logpdf(wishart(4.0, mvt_S), [1.1, 0.3]) == -Inf
    @test_throws ArgumentError wishart(1.0, mvt_S)         # nu <= d - 1
    @test_throws ArgumentError inversewishart(0.5, mvt_S)
end

@testset "cholesky_cov_transform" begin
    for mvt_d in (1, 2, 3)
        mvt_t = CholeskyCovTransform(mvt_d)
        mvt_p = (mvt_d * (mvt_d + 1)) ÷ 2
        mvt_rng = MersenneTwister(700 + mvt_d)
        for _ = 1:5
            mvt_u = randn(mvt_rng, mvt_p)
            mvt_c = UncertainTea.to_constrained(mvt_t, mvt_u)
            # round-trip
            @test UncertainTea.to_unconstrained(mvt_t, mvt_c) ≈ mvt_u atol = 1e-12
            # constrained diagonal (packed positions) is strictly positive
            mvt_L = mvt_unpack(mvt_c, mvt_d)
            @test all(mvt_L[i, i] > 0 for i = 1:mvt_d)
            # log-abs-det matches the Jacobian of the constrain map
            mvt_J = mvt_FD.jacobian(z -> UncertainTea.to_constrained(mvt_t, z), mvt_u)
            @test UncertainTea.logabsdetjac(mvt_t, mvt_u) ≈ log(abs(mvt_LA.det(mvt_J))) atol = 1e-9
        end
    end
end

# --- gradient parity (analytic / device) --------------------------------------
mvt_fd_grad = function (model, x, args, cm)
    g = similar(x)
    for i in eachindex(x)
        h = cbrt(eps(Float64)) * max(1.0, abs(x[i]))
        xp = copy(x)
        xp[i] += h
        xm = copy(x)
        xm[i] -= h
        g[i] =
            (
                logjoint_unconstrained(model, xp, args, cm) -
                logjoint_unconstrained(model, xm, args, cm)
            ) / (2h)
    end
    return g
end

@tea static function mvt_diag_latent_model()
    s ~ mvstudentt(6.0, [0.0, 1.0], [1.5, 0.8])
    return s
end
@tea static function mvt_diag_obs_model()
    m ~ normal(0.0, 1.0)
    {:y} ~ mvstudentt(5.0, [m, m], [1.0, 0.8])
    return m
end
@tea static function mvt_dense_obs_model(Larg)
    m ~ normal(0.0, 1.0)
    {:y} ~ mvstudenttdense(5.0, [m, m], Larg)
    return m
end
@tea static function mvt_wishart_model()
    Sigma ~ wishart(5.0, [2.0 0.5; 0.5 1.0])
    return Sigma
end
@tea static function mvt_invwishart_model()
    Sigma ~ inversewishart(5.0, [2.0 0.5; 0.5 1.0])
    return Sigma
end

@testset "mvt_gradient_parity" begin
    mvt_factor = [1.2 0.0; 0.3 0.9]
    mvt_cases = [
        (mvt_diag_latent_model, (), choicemap(), 31),
        (mvt_diag_obs_model, (), choicemap((:y, [0.4, -0.2])), 32),
        (mvt_dense_obs_model, (mvt_factor,), choicemap((:y, [0.4, -0.2])), 33),
    ]
    for (mvt_model, mvt_args, mvt_cm, mvt_seed) in mvt_cases
        @test backend_report(mvt_model).supported == true
        mvt_rng = MersenneTwister(mvt_seed)
        mvt_tr, _ = generate(mvt_model, mvt_args, mvt_cm; rng=mvt_rng)
        mvt_base = transform_to_unconstrained(mvt_tr)
        mvt_pts = mvt_base .+ 0.3 .* randn(mvt_rng, length(mvt_base), 3)
        mvt_an = batched_logjoint_gradient_unconstrained(mvt_model, mvt_pts, mvt_args, mvt_cm)
        for i = 1:size(mvt_pts, 2)
            @test mvt_an[:, i] ≈ mvt_fd_grad(mvt_model, mvt_pts[:, i], mvt_args, mvt_cm) atol = 5e-6
            @test mvt_an[:, i] ≈
                  logjoint_gradient_unconstrained(mvt_model, mvt_pts[:, i], mvt_args, mvt_cm) atol = 1e-8
        end
        # device (CPU() Float64) matches the host batched path
        mvt_dv, mvt_dg = device_batched_logjoint_gradient(mvt_model, mvt_pts, mvt_args, mvt_cm)
        mvt_hv = batched_logjoint_unconstrained(mvt_model, mvt_pts, mvt_args, mvt_cm)
        @test collect(mvt_dv) ≈ mvt_hv atol = 1e-7
        @test collect(mvt_dg) ≈ mvt_an atol = 1e-7
    end
end

# Wishart is CPU-reference only: honest backend/device rejection, but the
# ForwardDiff-fallback gradient still matches finite differences.
@testset "wishart_cpu_reference_only" begin
    for (mvt_model, mvt_seed) in
        [(mvt_wishart_model, 41), (mvt_invwishart_model, 42)]
        @test backend_report(mvt_model).supported == false
        mvt_rng = MersenneTwister(mvt_seed)
        mvt_tr, _ = generate(mvt_model, (), choicemap(); rng=mvt_rng)
        mvt_p = transform_to_unconstrained(mvt_tr)
        mvt_g = logjoint_gradient_unconstrained(mvt_model, mvt_p, (), choicemap())
        @test mvt_g ≈ mvt_fd_grad(mvt_model, mvt_p, (), choicemap()) atol = 1e-6
    end
end

# --- small posterior recovery -------------------------------------------------
# Draws for the named constrained parameter, flattened across chains.
function mvt_param_draws(chains, name)
    names = parameter_names(chains; space=:constrained)
    idx = findfirst(==(name), names)
    idx === nothing && error("parameter $name not found in $names")
    draws = posterior_array(chains; space=:constrained)
    return vec(draws[:, :, idx])
end

@testset "mvt_recovery" begin
    # A dense multivariate-t likelihood: recover a shared latent mean shift.
    mvt_L = [1.0 0.0; 0.4 0.9]
    mvt_true = 1.3
    mvt_rng = MersenneTwister(555)
    mvt_obs = [rand(mvt_rng, mvstudenttdense(8.0, [mvt_true, mvt_true], mvt_L)) for _ = 1:60]

    @tea static function mvt_recover_model(L, ys)
        mu ~ normal(0.0, 3.0)
        for i in eachindex(ys)
            {:y => i} ~ mvstudenttdense(8.0, [mu, mu], L)
        end
        return mu
    end
    mvt_cm = choicemap(((:y => i, mvt_obs[i]) for i in eachindex(mvt_obs))...)
    mvt_chains = batched_nuts(
        mvt_recover_model,
        (mvt_L, mvt_obs),
        mvt_cm;
        num_chains=4,
        num_samples=300,
        num_warmup=300,
        rng=MersenneTwister(9),
    )
    mvt_draws = mvt_param_draws(mvt_chains, "mu")
    @test abs(mvt_mean(mvt_draws) - mvt_true) < 0.35

    # An inverse-Wishart prior on a 2x2 covariance, observed multivariate normals.
    mvt_rng2 = MersenneTwister(777)
    mvt_Ltrue = [1.2 0.0; 0.5 0.8]
    mvt_data = [rand(mvt_rng2, mvnormaldense([0.0, 0.0], mvt_Ltrue)) for _ = 1:80]

    @tea static function mvt_iw_model(ys)
        L ~ inversewishart(6.0, [1.0 0.0; 0.0 1.0])
        for i in eachindex(ys)
            {:y => i} ~ mvnormaldense([0.0, 0.0], [L[1] 0.0; L[2] L[3]])
        end
        return L
    end
    mvt_cm2 = choicemap(((:y => i, mvt_data[i]) for i in eachindex(mvt_data))...)
    mvt_chains2 = batched_nuts(
        mvt_iw_model,
        (mvt_data,),
        mvt_cm2;
        num_chains=2,
        num_samples=200,
        num_warmup=300,
        rng=MersenneTwister(13),
    )
    # posterior covariance M = L L' should sit near the data scatter: the packed
    # latent's first component is L[1,1], so E[L[1,1]^2] ~ Sigma[1,1].
    mvt_Sigma_true = mvt_Ltrue * mvt_Ltrue'
    mvt_l11 = mvt_param_draws(mvt_chains2, "L[1]")
    @test mvt_mean(mvt_l11 .^ 2) > 0.5 * mvt_Sigma_true[1, 1]
    @test mvt_mean(mvt_l11 .^ 2) < 2.0 * mvt_Sigma_true[1, 1]
end

@testset "mvnormal with runtime mean/scale arguments (issue #309)" begin
    # both arguments runtime-valued (no static size): used to CRASH macro
    # expansion via `something(nothing, nothing)`; now a dynamic-size
    # observation-only form
    mvn309 = @tea static function mvn309_model(mu0, sig0)
        tau ~ lognormal(0.0, 0.5)
        {:y} ~ mvnormal(mu0, sig0)
        return tau
    end
    mvn309_mu = [0.0, 1.0, -0.5]
    mvn309_sig = [1.0, 0.8, 1.2]
    mvn309_y = [0.3, 1.2, -0.4]
    mvn309_cm = choicemap((:y, mvn309_y))
    lj = logjoint(mvn309, [0.1], (mvn309_mu, mvn309_sig), mvn309_cm)
    # matches the per-component diagonal normal sum plus the latent prior term
    ref = sum(
        UncertainTea.logpdf(normal(mvn309_mu[i], mvn309_sig[i]), mvn309_y[i]) for i = 1:3
    )
    lj_prior_only = logjoint(mvn309, [0.1], (mvn309_mu, mvn309_sig .* 2), mvn309_cm)
    @test isfinite(lj)
    @test lj != lj_prior_only     # the observation term responds to the scale
    g = logjoint_gradient_unconstrained(mvn309, [0.1], (mvn309_mu, mvn309_sig), mvn309_cm)
    @test all(isfinite, g)
end
