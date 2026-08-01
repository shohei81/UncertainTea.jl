# Issue #144: type-stable generated single-chain scorer.
#
# The generated straight-line scorer (src/generated_scorer.jl) replaces the
# Any-boxed compiled-plan interpreter for `logjoint` / `logjoint_unconstrained`
# / `logjoint_gradient_unconstrained` (and the single-chain samplers' reject-mode
# gradient cache) whenever the plan is representable. It MUST produce results
# numerically identical to the interpreter, which stays the source of truth.
#
# This is an in-process before/after comparison: `_USE_GENERATED_SCORER[]`
# toggles the generated path off (interpreter) and on (generated), and the two
# are compared for the same model/params. No hardcoded goldens -- the
# interpreter itself is the reference. `_GEN_SCORER_LAST_USED[]` records whether
# the generated path actually served the last scalar/gradient call, so the test
# also pins which model classes are generated vs fall back to the interpreter.

const gsi_UT = UncertainTea

# generated classes: parameter-valued latents + staged Float64 loop observations
@tea static function gsi_gauss(n)
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea static function gsi_logistic(X, n)
    b1 ~ normal(0.0, 1.0)
    b2 ~ normal(0.0, 1.0)
    b3 ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ bernoulli(1.0 / (1.0 + exp(-(b1 * X[1, i] + b2 * X[2, i] + b3 * X[3, i]))))
    end
    return b1
end

@tea static function gsi_eight_nc(sigma)
    mu ~ normal(0.0, 5.0)
    tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

@tea static function gsi_iid(n)
    mu ~ normal(0.0, 1.0)
    xs ~ iid(normal(mu, 1.0), 3)
    for i = 1:n
        {:y => i} ~ normal(xs[1], 0.7)
    end
    return mu
end

# fallback classes (interpreter): marginalize=:enumerate, broadcast obs, Float32 obs
@tea static function gsi_marg()
    z = ({:z} ~ bernoulli(0.4; marginalize=:enumerate))
    mu = ifelse(z, 1.0, -1.0)
    {:y} ~ normal(mu, 1.0)
    return z
end

@tea static function gsi_broadcast(xs)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(slope .* xs, sigma)
end

@tea static function gsi_float32(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0f0)
    end
    return mu
end

# Evaluate the unconstrained scoring entry points -- `logjoint_unconstrained`
# and `logjoint_gradient_unconstrained` (the sampler hot paths) -- with the
# generated path forced OFF (interpreter) and ON (generated), plus the
# constrained `logjoint` on valid constrained-space params. `params` is
# UNCONSTRAINED-space; `logjoint` gets the transformed constrained image so its
# distribution arguments stay valid. Returns both results and whether the
# generated path served the ON run.
function gsi_compare(model, args, cons, params; reject::Bool=false)
    prev = gsi_UT._USE_GENERATED_SCORER[]
    try
        gsi_UT._USE_GENERATED_SCORER[] = false
        constrained = transform_to_constrained(model, params, args, cons)
        lj_i = logjoint(model, constrained, args, cons)
        lu_i = logjoint_unconstrained(model, params, args, cons; reject_invalid_parameters=reject)
        gi = gsi_UT._logjoint_gradient_cache(model, params, args, cons; reject_invalid_parameters=reject)
        gr_i = copy(gsi_UT._logjoint_gradient!(gi, params))

        gsi_UT._USE_GENERATED_SCORER[] = true
        lj_g = logjoint(model, constrained, args, cons)
        used_lj = gsi_UT._GEN_SCORER_LAST_USED[]
        lu_g = logjoint_unconstrained(model, params, args, cons; reject_invalid_parameters=reject)
        gg = gsi_UT._logjoint_gradient_cache(model, params, args, cons; reject_invalid_parameters=reject)
        gr_g = copy(gsi_UT._logjoint_gradient!(gg, params))
        return (; lj_i, lj_g, lu_i, lu_g, gr_i, gr_g, used_lj)
    finally
        gsi_UT._USE_GENERATED_SCORER[] = prev
    end
end

@testset "generated scorer numerical identity (issue #144)" begin
    gsi_rng = MersenneTwister(144)
    gsi_n = 300
    gsi_y = randn(gsi_rng, gsi_n) .* 1.3 .+ 0.4
    gsi_gauss_cons = choicemap(((:y => i, gsi_y[i]) for i = 1:gsi_n)...)

    gsi_ln = 120
    gsi_X = randn(gsi_rng, 3, gsi_ln)
    gsi_ly = Float64.(rand(gsi_rng, gsi_ln) .< 0.5)
    gsi_logi_cons = choicemap(((:y => i, gsi_ly[i]) for i = 1:gsi_ln)...)

    gsi_sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    gsi_es_y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    gsi_es_cons = choicemap(((:y => i, gsi_es_y[i]) for i = 1:8)...)

    gsi_in = 5
    gsi_iy = randn(gsi_rng, gsi_in)
    gsi_iid_cons = choicemap(((:y => i, gsi_iy[i]) for i = 1:gsi_in)...)

    @testset "generated classes match the interpreter to rtol 1e-12" begin
        cases = (
            ("gauss", gsi_gauss, (gsi_n,), gsi_gauss_cons, 2),
            ("logistic", gsi_logistic, (gsi_X, gsi_ln), gsi_logi_cons, 3),
            ("eight_schools_nc", gsi_eight_nc, (gsi_sigma,), gsi_es_cons, 10),
            ("iid", gsi_iid, (gsi_in,), gsi_iid_cons, 4),
        )
        for (name, model, args, cons, dim) in cases, seed = 1:3, reject in (false, true)
            theta = randn(MersenneTwister(31 * seed + dim), dim) .* 0.5
            r = gsi_compare(model, args, cons, theta; reject=reject)
            @test r.used_lj
            @test isapprox(r.lj_g, r.lj_i; rtol=1e-12)
            @test isapprox(r.lu_g, r.lu_i; rtol=1e-12)
            @test isapprox(r.gr_g, r.gr_i; rtol=1e-12)
        end
    end

    @testset "generated path agrees to within a few ulp on Float64 scoring" begin
        # The generated scorer sums in execution order (a left fold) while the
        # interpreter right-folds, so the two reassociate; the difference is a
        # couple of ulp for these plans (well inside the 1e-12 gate), not a
        # correctness gap. Values are otherwise identical operations.
        theta = [0.13, 0.22]
        r = gsi_compare(gsi_gauss, (gsi_n,), gsi_gauss_cons, theta)
        @test isapprox(r.lj_g, r.lj_i; rtol=1e-12)
        @test isapprox(r.gr_g, r.gr_i; rtol=1e-12)
    end

    @testset "reject-mode -Inf semantics agree" begin
        @tea static function gsi_reject(n)
            mu ~ normal(0.0, 1.0)
            ls ~ normal(0.0, 1.0)
            for i = 1:n
                {:y => i} ~ normal(mu, exp(ls))
            end
            return mu
        end
        cons = choicemap(((:y => i, gsi_y[i]) for i = 1:gsi_n)...)
        bad = [0.0, -800.0]  # exp(-800) underflows to 0 -> normal(mu, 0) invalid
        prev = gsi_UT._USE_GENERATED_SCORER[]
        try
            gsi_UT._USE_GENERATED_SCORER[] = false
            lu_i = logjoint_unconstrained(gsi_reject, bad, (gsi_n,), cons; reject_invalid_parameters=true)
            gsi_UT._USE_GENERATED_SCORER[] = true
            lu_g = logjoint_unconstrained(gsi_reject, bad, (gsi_n,), cons; reject_invalid_parameters=true)
            @test gsi_UT._GEN_SCORER_LAST_USED[]
            @test lu_i == -Inf
            @test lu_g == -Inf
        finally
            gsi_UT._USE_GENERATED_SCORER[] = prev
        end
    end

    @testset "fallback classes stay on the interpreter" begin
        # marginalize=:enumerate
        gsi_UT._USE_GENERATED_SCORER[] = true
        logjoint(gsi_marg, Float64[], (), choicemap((:y, 0.7)))
        @test gsi_UT._GEN_SCORER_LAST_USED[] == false
        # broadcast observation: now GENERATED via static whole-vector staging
        # (issue #288) -- and identical to the interpreter
        bxs = collect(0.0:0.2:1.0)
        by = 1.1 .* bxs
        bc_prev = gsi_UT._USE_GENERATED_SCORER[]
        gsi_UT._USE_GENERATED_SCORER[] = false
        bc_interp = logjoint(gsi_broadcast, [0.3, 0.5], (bxs,), choicemap(:y => by))
        gsi_UT._USE_GENERATED_SCORER[] = bc_prev
        bc_gen = logjoint(gsi_broadcast, [0.3, 0.5], (bxs,), choicemap(:y => by))
        @test gsi_UT._GEN_SCORER_LAST_USED[]
        @test bc_gen == bc_interp
        # Float32 observation (not densifiable as Float64)
        f32_cons = choicemap(((:y => i, Float32(gsi_y[i])) for i = 1:gsi_n)...)
        logjoint(gsi_float32, [0.2], (gsi_n,), f32_cons)
        @test gsi_UT._GEN_SCORER_LAST_USED[] == false
    end
end
