# Issue #188: the sampler's logjoint-VALUE path (`target_logdensity` on a
# `ModelDensityTarget`) re-staged all N observations on every call by routing
# through the public `logjoint_unconstrained` (an O(N) plan re-walk that rebuilds
# `(:y,i)` addresses). NUTS evaluates the Hamiltonian value at every tree node,
# so that cost dominated the per-draw loop once scoring became O(1) (#144/#189).
# The fix evaluates the value through the observations the gradient cache already
# staged (`_logjoint_value_from_cache`); observations are constant data, staged
# once with an O(1) staleness check.
#
# This is pure caching -- the value must be BITWISE identical to the public
# `logjoint_unconstrained(...; reject_invalid_parameters=true)` the sampler used
# before -- so this file pins that down bit-for-bit across the generated classes
# (gauss / logistic / eight-schools-nc / iid), the interpreter-fallback classes
# (marginalize / broadcast / Float32 obs), reject-mode -Inf, and an in-place
# constraint mutation (the gibbs refresh path).

const svp_UT = UncertainTea

@tea static function svp_gauss(n)
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea static function svp_logistic(X, n)
    b1 ~ normal(0.0, 1.0)
    b2 ~ normal(0.0, 1.0)
    b3 ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ bernoulli(1.0 / (1.0 + exp(-(b1 * X[1, i] + b2 * X[2, i] + b3 * X[3, i]))))
    end
    return b1
end

@tea static function svp_eight_nc(sigma)
    mu ~ normal(0.0, 5.0)
    tau ~ truncatedstudentt(1.0, 0.0, 5.0, 0.0, Inf)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for i = 1:8
        {:y => i} ~ normal(theta[i], sigma[i])
    end
    return mu
end

@tea static function svp_iid_latents(n)
    mu ~ normal(0.0, 1.0)
    xs ~ iid(normal(mu, 1.0), 3)
    for i = 1:n
        {:y => i} ~ normal(xs[1], 0.7)
    end
    return mu
end

# interpreter fallbacks (not densifiable / marginalized)
@tea static function svp_marg()
    z = ({:z} ~ bernoulli(0.4; marginalize=:enumerate))
    mu = ifelse(z, 1.0, -1.0)
    {:y} ~ normal(mu, 1.0)
    return z
end

@tea static function svp_broadcast(xs)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(slope .* xs, sigma)
end

@tea static function svp_float32(n)
    mu ~ normal(0.0f0, 1.0f0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0f0)
    end
    return mu
end

# bit-level equality (distinguishes -Inf/Inf, ±0.0, NaN payloads from `==`)
svp_same_bits(a::Float64, b::Float64) = reinterpret(UInt64, a) === reinterpret(UInt64, b)

# Build the sampler's value-path target exactly as `nuts` does (reject-on cache),
# then compare `target_logdensity` bit-for-bit to the public value path it
# replaces.
function svp_target(model, args, cons, position)
    cache = svp_UT._logjoint_gradient_cache(model, position, args, cons; reject_invalid_parameters=true)
    return svp_UT.ModelDensityTarget(model, args, cons, cache)
end

@testset "sampler value path reuses staged observations (issue #188)" begin
    svp_rng = MersenneTwister(188)
    n = 500
    gy = randn(svp_rng, n) .* 1.3 .+ 0.4
    gauss_cons = choicemap(((:y => i, gy[i]) for i = 1:n)...)

    ln = 150
    X = randn(svp_rng, 3, ln)
    ly = Float64.(rand(svp_rng, ln) .< 0.5)
    logi_cons = choicemap(((:y => i, ly[i]) for i = 1:ln)...)

    es_sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
    es_y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
    es_cons = choicemap(((:y => i, es_y[i]) for i = 1:8)...)

    iy = randn(svp_rng, 6)
    iid_cons = choicemap(((:y => i, iy[i]) for i = 1:6)...)

    @testset "bitwise identity to the public value path (generated + fallback)" begin
        cases = (
            ("gauss", svp_gauss, (n,), gauss_cons, 2),
            ("logistic", svp_logistic, (X, ln), logi_cons, 3),
            ("eight_schools_nc", svp_eight_nc, (es_sigma,), es_cons, 10),
            ("iid_latents", svp_iid_latents, (6,), iid_cons, 4),
        )
        for (name, model, args, cons, dim) in cases, seed = 1:4
            position = randn(MersenneTwister(7 * seed + dim), dim) .* 0.5
            target = svp_target(model, args, cons, position)
            reference = logjoint_unconstrained(model, position, args, cons; reject_invalid_parameters=true)
            value = svp_UT.target_logdensity(target, position)
            @test svp_same_bits(value, reference)
            # a second call must reuse the staged obs and stay identical (no drift)
            @test svp_same_bits(svp_UT.target_logdensity(target, position), reference)
        end

        # interpreter-fallback classes: value path must still match the public one
        marg_cons = choicemap((:y, 0.7))
        marg_target = svp_target(svp_marg, (), marg_cons, Float64[])
        @test svp_same_bits(
            svp_UT.target_logdensity(marg_target, Float64[]),
            logjoint_unconstrained(svp_marg, Float64[], (), marg_cons; reject_invalid_parameters=true),
        )

        bxs = collect(0.0:0.2:1.0)
        bcons = choicemap(:y => 1.1 .* bxs)
        bpos = [0.3, 0.5]
        b_target = svp_target(svp_broadcast, (bxs,), bcons, bpos)
        @test svp_same_bits(
            svp_UT.target_logdensity(b_target, bpos),
            logjoint_unconstrained(svp_broadcast, bpos, (bxs,), bcons; reject_invalid_parameters=true),
        )

        f32_cons = choicemap(((:y => i, Float32(gy[i])) for i = 1:n)...)
        fpos = [0.2]
        f_target = svp_target(svp_float32, (n,), f32_cons, fpos)
        @test svp_same_bits(
            svp_UT.target_logdensity(f_target, fpos),
            logjoint_unconstrained(svp_float32, fpos, (n,), f32_cons; reject_invalid_parameters=true),
        )
    end

    @testset "reject-mode: invalid parameters score -Inf, never throw" begin
        @tea static function svp_reject(n)
            mu ~ normal(0.0, 1.0)
            ls ~ normal(0.0, 1.0)
            for i = 1:n
                {:y => i} ~ normal(mu, exp(ls))
            end
            return mu
        end
        cons = choicemap(((:y => i, gy[i]) for i = 1:n)...)
        bad = [0.0, -800.0]  # exp(-800) underflows to 0 -> normal(mu, 0) invalid
        target = svp_target(svp_reject, (n,), cons, [0.0, 0.0])
        value = svp_UT.target_logdensity(target, bad)
        @test value == -Inf
        @test svp_same_bits(
            value,
            logjoint_unconstrained(svp_reject, bad, (n,), cons; reject_invalid_parameters=true),
        )
        # a good position on the same target still scores finite and matches
        good = [0.1, 0.2]
        @test svp_same_bits(
            svp_UT.target_logdensity(target, good),
            logjoint_unconstrained(svp_reject, good, (n,), cons; reject_invalid_parameters=true),
        )
    end

    @testset "in-place constraint mutation refreshes staged obs (gibbs path)" begin
        # A cache reused across an in-place observation mutation (as gibbs mutates
        # its merged ChoiceMap) must re-stage, not score stale data.
        cons = choicemap(((:y => i, gy[i]) for i = 1:n)...)
        position = [0.15, 0.05]
        target = svp_target(svp_gauss, (n,), cons, position)
        before = svp_UT.target_logdensity(target, position)
        @test svp_same_bits(
            before,
            logjoint_unconstrained(svp_gauss, position, (n,), cons; reject_invalid_parameters=true),
        )
        # mutate several observed values in place (bumps the ChoiceMap mutation
        # counter, invalidating the staged obs)
        for i = 1:n
            svp_UT._pushchoice!(cons, :y => i, gy[i] + 0.5)
        end
        after = svp_UT.target_logdensity(target, position)
        @test svp_same_bits(
            after,
            logjoint_unconstrained(svp_gauss, position, (n,), cons; reject_invalid_parameters=true),
        )
        @test !svp_same_bits(before, after)
    end
end
