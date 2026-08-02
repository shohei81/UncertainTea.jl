# User-defined distributions registered from outside the package (issue #17).
# The structs live at top level because Julia forbids struct definitions in
# local scope; everything else runs inside the named testset below.

struct CregSkewNormalDist{T<:Real} <: AbstractTeaDistribution
    location::T
    scale::T
    shape::T
    function CregSkewNormalDist(location::T, scale::T, shape::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("skewnormal requires scale > 0"))
        return new{T}(location, scale, shape)
    end
end

creg_skewnormal_builder(location, scale, shape) = CregSkewNormalDist(promote(location, scale, shape)...)

function UncertainTea.logpdf(dist::CregSkewNormalDist, x)
    z = (x - dist.location) / dist.scale
    return log(2) - z^2 / 2 - log(2 * pi) / 2 - log(dist.scale) + log(UncertainTea._std_normal_cdf(dist.shape * z))
end

function Random.rand(rng::AbstractRNG, dist::CregSkewNormalDist)
    delta = dist.shape / sqrt(1 + dist.shape^2)
    u0, v = randn(rng), randn(rng)
    z = delta * abs(u0) + sqrt(1 - delta^2) * v
    return dist.location + dist.scale * z
end

struct CregHalfCauchyDist{T<:Real} <: AbstractTeaDistribution
    scale::T
    function CregHalfCauchyDist(scale::T) where {T<:Real}
        scale > zero(T) || throw(ArgumentError("halfcauchy requires scale > 0"))
        return new{T}(scale)
    end
end

function UncertainTea.logpdf(dist::CregHalfCauchyDist, x)
    xx = float(x)
    xx > zero(xx) || return oftype(xx, -Inf)
    return log(2) - log(pi) - log(dist.scale) - log1p((xx / dist.scale)^2)
end

Random.rand(rng::AbstractRNG, dist::CregHalfCauchyDist) = dist.scale * tan(pi * rand(rng) / 2)

# Observation-only two-point distribution on {-1, +1}: registered without a
# transform, so a latent draw gets no parameter slot.
struct CregTwoPointDist{T<:Real} <: AbstractTeaDistribution
    p::T
    function CregTwoPointDist(p::T) where {T<:Real}
        zero(T) < p < one(T) || throw(ArgumentError("twopoint requires 0 < p < 1"))
        return new{T}(p)
    end
end

function UncertainTea.logpdf(dist::CregTwoPointDist, x)
    x == 1 && return log(dist.p)
    x == -1 && return log1p(-dist.p)
    return oftype(float(dist.p), -Inf)
end

Random.rand(rng::AbstractRNG, dist::CregTwoPointDist) = rand(rng) < dist.p ? 1.0 : -1.0

register_distribution(:cregskewnormal; builder=creg_skewnormal_builder, transform=IdentityTransform())
register_distribution(:creghalfcauchy; builder=CregHalfCauchyDist, transform=LogTransform())
register_distribution(:cregtwopoint; builder=CregTwoPointDist)

@tea static function creg_model()
    x ~ cregskewnormal(0.0, 1.0, 3.0)
    s ~ creghalfcauchy(1.5)
    {:y} ~ normal(x, s)
    return x
end

@tea static function creg_obs_model()
    x ~ normal(0.0, 1.0)
    {:y} ~ cregtwopoint(0.7)
    return x
end

@tea static function creg_latent_no_transform_model()
    z ~ cregtwopoint(0.3)
    x ~ normal(0.0, 1.0)
    {:y} ~ normal(x, 1.0)
    return z
end

# Re-registration capture semantics: models keep the builder/transform they
# were DEFINED with. These definitions must be sequential top-level statements
# (macro expansion inside one top-level block would see a single registry
# snapshot), so the swap sequence lives here rather than inside the testset.
creg_swap_builder_a(scale) = CregHalfCauchyDist(scale)
creg_swap_builder_b(scale) = CregSkewNormalDist(promote(0.0, scale, 0.0)...)

register_distribution(:cregswap; builder=creg_swap_builder_a, transform=LogTransform())

@tea static function creg_swap_before()
    x ~ cregswap(2.0)
    {:y} ~ normal(x, 1.0)
    return x
end

register_distribution(:cregswap; builder=creg_swap_builder_b, transform=IdentityTransform())

@tea static function creg_swap_after()
    x ~ cregswap(2.0)
    {:y} ~ normal(x, 1.0)
    return x
end

@testset "custom_distribution_registration" begin
    creg_constraints = choicemap((:y, 0.4))

    @testset "creg_api" begin
        @test :cregskewnormal in registered_distributions()
        @test :creghalfcauchy in registered_distributions()
        @test_throws ArgumentError register_distribution(:normal; builder=creg_skewnormal_builder)
        @test_throws ArgumentError register_distribution(:exp; builder=creg_skewnormal_builder)
        @test_throws ArgumentError register_distribution(:cregbad; builder=1.0)
        # re-registration overwrites
        first_registration = register_distribution(:cregtmp; builder=CregHalfCauchyDist)
        second_registration = register_distribution(:cregtmp; builder=CregHalfCauchyDist, transform=LogTransform())
        @test first_registration.transform === nothing
        @test second_registration.transform isa LogTransform
    end

    @testset "creg_runtime_and_layout" begin
        trace, logw = generate(creg_model, (), creg_constraints; rng=MersenneTwister(7))
        @test isfinite(logw)
        @test haskey(trace.choices, :x)
        @test trace[:s] > 0
        @test trace[:y] == 0.4

        layout = parameterlayout(creg_model)
        @test parametercount(layout) == 2
        @test layout.slots[1].transform isa IdentityTransform
        @test layout.slots[2].transform isa LogTransform

        unconstrained = transform_to_unconstrained(trace)
        reconstrained = transform_to_constrained(creg_model, unconstrained)
        @test reconstrained ≈ parameter_vector(trace) atol = 1e-12
        @test unconstrained[2] ≈ log(trace[:s]) atol = 1e-12
    end

    @testset "creg_scoring_and_gradient" begin
        trace, _ = generate(creg_model, (), creg_constraints; rng=MersenneTwister(9))
        params = parameter_vector(trace)
        manual =
            UncertainTea.logpdf(creg_skewnormal_builder(0.0, 1.0, 3.0), trace[:x]) +
            UncertainTea.logpdf(CregHalfCauchyDist(1.5), trace[:s]) +
            UncertainTea.logpdf(normal(trace[:x], trace[:s]), 0.4)
        @test logjoint(creg_model, params, (), creg_constraints) ≈ manual atol = 1e-8
        @test logjoint(creg_model, params, (), creg_constraints) ≈
              assess(creg_model, (), choicemap((:x, trace[:x]), (:s, trace[:s]), (:y, 0.4))) atol = 1e-8

        unconstrained = transform_to_unconstrained(trace)
        gradient = logjoint_gradient_unconstrained(creg_model, unconstrained, (), creg_constraints)
        for i in eachindex(unconstrained)
            h = cbrt(eps(Float64)) * max(1.0, abs(unconstrained[i]))
            up = copy(unconstrained)
            up[i] += h
            down = copy(unconstrained)
            down[i] -= h
            fd =
                (
                    logjoint_unconstrained(creg_model, up, (), creg_constraints) -
                    logjoint_unconstrained(creg_model, down, (), creg_constraints)
                ) / (2h)
            @test gradient[i] ≈ fd atol = 5e-6
        end
    end

    @testset "creg_backend_honesty" begin
        report = backend_report(creg_model)
        @test report.supported == false
        @test any(occursin("cregskewnormal", issue) for issue in report.issues)

        points = reshape([0.1 0.4; -0.2 0.2], 2, 2)
        cache = BatchedLogjointGradientCache(creg_model, points, (), creg_constraints)
        @test isnothing(cache.backend_cache)
        @test !isempty(cache.column_caches)
        batched = batched_logjoint_unconstrained(creg_model, points, (), creg_constraints)
        reference = [logjoint_unconstrained(creg_model, points[:, i], (), creg_constraints) for i = 1:2]
        @test batched ≈ reference atol = 1e-8
    end

    @testset "creg_nuts_smoke" begin
        chain = first(nuts(creg_model, (), creg_constraints; num_samples=40, num_warmup=40, rng=MersenneTwister(3)))
        @test all(isfinite, chain.constrained_samples)
        @test all(chain.constrained_samples[2, :] .> 0)
    end

    @testset "creg_reregistration_capture" begin
        # the model defined before the swap keeps the half-Cauchy builder and
        # its LogTransform; the model defined after uses the new builder
        @test parameterlayout(creg_swap_before).slots[1].transform isa LogTransform
        @test parameterlayout(creg_swap_after).slots[1].transform isa IdentityTransform
        swap_constraints = choicemap((:y, 0.5))
        @test logjoint(creg_swap_before, [-1.0], (), swap_constraints) == -Inf
        @test isfinite(logjoint(creg_swap_after, [-1.0], (), swap_constraints))
        before_trace, _ = generate(creg_swap_before, (), swap_constraints; rng=MersenneTwister(11))
        @test before_trace[:x] > 0
    end

    @testset "creg_observation_only" begin
        obs_trace, obs_logw = generate(creg_obs_model, (), choicemap((:y, 1.0)); rng=MersenneTwister(4))
        @test isfinite(obs_logw)
        @test obs_trace[:y] == 1.0
        params = parameter_vector(obs_trace)
        expected = UncertainTea.logpdf(normal(0.0, 1.0), obs_trace[:x]) + log(0.7)
        @test logjoint(creg_obs_model, params, (), choicemap((:y, 1.0))) ≈ expected atol = 1e-8

        # a latent draw from a transform-less family gets no parameter slot
        @test parametercount(parameterlayout(creg_latent_no_transform_model)) == 1
        latent_trace, _ = generate(creg_latent_no_transform_model, (), choicemap((:y, 0.2)); rng=MersenneTwister(5))
        @test latent_trace[:z] in (-1.0, 1.0)
    end
end
