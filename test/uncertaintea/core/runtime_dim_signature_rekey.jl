# (signature, dims)-keyed resolution (issue #289, PR-1).
#
# The signature cache is keyed by `(Set{Address}, dims::Tuple{Vararg{Int}})`,
# where `dims` comes from `_resolve_runtime_dims(model, signature, args)`.
# Dims-free models (no runtime-dimension candidates) always resolve `()` --
# one boolean test, one cache entry per signature, identical
# ResolvedSignaturePlan across calls with different argument VALUES. A LATENT
# candidate of any runtime-dim family (mvnormal/mvnormaldense as of PR-2,
# mvstudentt/mvstudenttdense/dirichlet as of PR-3) resolves its dims from the
# arguments; runtime_dim_latents.jl owns the capability tests. An OBSERVED
# candidate keeps working: dynamic-size observations need no slot.

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

    @testset "runtime-length candidates: detection, resolution, observed path" begin
        # the mvnormal candidate is detected at plan build ...
        candidates = U.executionplan(rdsk_runtime_latent_model).runtime_dim_candidates
        @test length(candidates) == 1
        @test candidates[1].family === :mvnormal
        @test candidates[1].address == (:theta,)

        # ... and as of PR-2 a latent `theta` RESOLVES (dims from the args)
        # instead of erroring; runtime_dim_latents.jl owns the capability tests
        @test U._resolve_runtime_dims(
            rdsk_runtime_latent_model, U.Set{U.Address}([(:y,)]), (3,),
        ) === (3,)

        # a PR-3 family (dirichlet) resolves too: the dims entry is the VALUE
        # length of the concentration vector (the slot is n-1 wide behind a
        # late SimplexTransform; runtime_dim_latents.jl owns the capability
        # tests)
        @tea static function rdsk_runtime_dirichlet(alpha)
            w ~ dirichlet(alpha)
            {:y} ~ normal(w[1], 1.0)
        end
        @test U._resolve_runtime_dims(
            rdsk_runtime_dirichlet, U.Set{U.Address}([(:y,)]), ([1.0, 1.0, 1.0],),
        ) === (3,)
        # logjoint takes the CONSTRAINED value (a length-3 simplex); the
        # unconstrained slot behind it is 2 wide (SimplexTransform(3))
        @test isfinite(
            logjoint(rdsk_runtime_dirichlet, [0.2, 0.3, 0.5], ([1.0, 1.0, 1.0],), choicemap((:y, 0.5))),
        )
        @test isfinite(
            logjoint_unconstrained(
                rdsk_runtime_dirichlet, zeros(2), ([1.0, 1.0, 1.0],), choicemap((:y, 0.5)),
            ),
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
