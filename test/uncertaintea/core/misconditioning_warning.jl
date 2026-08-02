# Misconditioning guard (issue #310): a constraint address that matches no
# model choice is silently dropped by the conditioning rule, so the intended
# observation stays a latent and sampling quietly targets the prior — the
# worst silent failure a typo can produce. The guard warns on the FIRST
# encounter of a conditioning signature (static template matching on the
# signature-cache miss; zero hot-path cost).

using Logging

@tea static function miscond_model(n)
    mu ~ normal(0.0, 1.0)
    {:z} ~ normal(mu, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 0.5)
    end
    return mu
end

miscond_capture(f) = begin
    buffer = IOBuffer()
    with_logger(SimpleLogger(buffer, Logging.Warn)) do
        f()
    end
    String(take!(buffer))
end

@testset "misconditioning_warning" begin
    miscond_full = choicemap((:z, 0.5), (:y => 1, 0.1), (:y => 2, 0.2), (:y => 3, 0.3))

    @testset "a typo'd address warns with the ignored address named" begin
        typo_cm = choicemap((:z, 0.5), (:zz, 0.3), (:y => 1, 0.1), (:y => 2, 0.2), (:y => 3, 0.3))
        out = miscond_capture() do
            logjoint(miscond_model, [0.1], (3,), typo_cm)
        end
        @test occursin("zz", out)
        @test occursin("IGNORED", out)
    end

    @testset "valid static / loop / latent-conditioning addresses stay silent" begin
        out = miscond_capture() do
            logjoint(miscond_model, [0.1], (3,), miscond_full)
            merged = choicemap(
                (:mu, 0.1), (:z, 0.5), (:y => 1, 0.1), (:y => 2, 0.2), (:y => 3, 0.3),
            )
            logjoint(miscond_model, Float64[], (3,), merged)
        end
        @test isempty(out)
    end

    @testset "the guard runs only on the first signature encounter" begin
        fresh = @tea static function miscond_fresh()
            mu ~ normal(0.0, 1.0)
            {:y} ~ normal(mu, 1.0)
            return mu
        end
        bad = choicemap((:y, 0.3), (:oops, 1.0))
        first_out = miscond_capture() do
            logjoint(fresh, [0.1], (), bad)
        end
        @test occursin("oops", first_out)
        # same signature again: the memoized plan short-circuits the guard
        second_out = miscond_capture() do
            logjoint(fresh, [0.2], (), bad)
        end
        @test isempty(second_out)
    end
end
