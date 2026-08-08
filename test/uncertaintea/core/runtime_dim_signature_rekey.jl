# (signature, dims)-keyed resolution (issue #289, PR-1).
#
# The signature cache is keyed by `(Set{Address}, dims::Tuple{Vararg{Int}})`,
# where `dims` comes from `_resolve_runtime_dims(model, signature, args)`.
# PR-1 only reserves the seam: dims-free models (no runtime-dimension
# candidates) always resolve `()` -- one boolean test, one cache entry per
# signature, identical ResolvedSignaturePlan across calls with different
# argument VALUES -- while a LATENT runtime-dim candidate (an mv-family choice
# whose length is only knowable from the model arguments) throws an early,
# informative ArgumentError instead of today's late no-parameter-slot failure.
# An OBSERVED candidate keeps working: dynamic-size observations need no slot.

@tea static function rdsk_dims_free_model(x)
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal(mu + x, 1.0)
end

@tea static function rdsk_runtime_latent_model(n)
    theta ~ mvnormal(zeros(n), ones(n))
    {:y} ~ normal(sum(theta), 1.0)
end

@testset "runtime_dim_signature_rekey" begin
    U = UncertainTea

    @testset "dims-free model resolves identically across arg values" begin
        cons = choicemap((:y, 1.5))
        resolved_a = U._resolve_signature_plan(rdsk_dims_free_model, cons, (0.5,))
        resolved_b = U._resolve_signature_plan(rdsk_dims_free_model, cons, (123.75,))
        # different argument VALUES must not re-resolve: same signature, same
        # dims (`()`), thus the IDENTICAL memoized ResolvedSignaturePlan object
        @test resolved_a === resolved_b

        # dims-free detection: no runtime-dimension candidates on the plan
        @test isempty(U.executionplan(rdsk_dims_free_model).runtime_dim_candidates)
        @test U._resolve_runtime_dims(rdsk_dims_free_model, U.Set{U.Address}(), (0.5,)) === ()
    end

    @testset "signature-cache size stays 1 across repeated calls" begin
        @tea static function rdsk_cache_size_model(x)
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu * x, 1.0)
        end
        cons = choicemap((:y, 0.25))
        for x in (1.0, 2.0, 3.5, -4.0)
            logjoint(rdsk_cache_size_model, [0.1], (x,), cons)
        end
        cache = rdsk_cache_size_model.signature_cache[]
        @test cache isa Dict{U._SignatureCacheKey,U.ResolvedSignaturePlan}
        @test length(cache) == 1
        key = only(keys(cache))
        @test key[1] == Set{U.Address}([(:y,)])
        @test key[2] === ()
    end

    @testset "latent runtime-length mvnormal errors early and informatively" begin
        # the candidate is detected at plan build ...
        candidates = U.executionplan(rdsk_runtime_latent_model).runtime_dim_candidates
        @test length(candidates) == 1
        @test candidates[1].family === :mvnormal
        @test candidates[1].address == (:theta,)

        # ... and a latent `theta` throws the early error at logjoint time,
        # naming the address and the issue
        err = try
            logjoint(rdsk_runtime_latent_model, zeros(3), (3,), choicemap((:y, 0.5)))
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("latent `mvnormal`", err.msg)
        @test occursin("(:theta,)", err.msg)
        @test occursin("issue #289", err.msg)
        @test occursin("literal vector/tuple", err.msg)

        # nuts hits the same early error (resolution is shared by the samplers)
        @test_throws ArgumentError nuts(
            rdsk_runtime_latent_model, (3,), choicemap((:y, 0.5));
            num_samples=5, num_warmup=5,
        )

        # OBSERVING the candidate keeps today's dynamic-size observation path:
        # both mv choice and y constrained -> no latent slot needed, scores fine
        full = choicemap((:theta, [0.1, -0.2, 0.3]), (:y, 0.5))
        expected =
            U.logpdf(mvnormal(zeros(3), ones(3)), [0.1, -0.2, 0.3]) +
            U.logpdf(normal(0.1 - 0.2 + 0.3, 1.0), 0.5)
        @test logjoint(rdsk_runtime_latent_model, Float64[], (3,), full) ≈ expected atol = 1e-10
    end
end
