# NaN observation guard (issue #346): a NaN inside a constrained observation
# value used to make `logjoint` (and every gradient) silently NaN, flowing
# unnoticed into waic/psis_loo-style scoring. The guard scans constraint
# values at constraint-resolution time and throws an ArgumentError naming the
# offending address. The scan is stamped on the ChoiceMap per mutation_count
# (`nan_checked_mutation_count`), so the warm path pays one Int comparison.
# Inf stays allowed: it is a legitimate value wherever the support admits it
# (it merely scores -Inf); only NaN is always a data bug.

@tea static function nanobs_vec_model(xs)
    mu ~ normal(0.0, 1.0)
    {:y} ~ normal.(mu .+ xs, 1.0)
    return mu
end

@tea static function nanobs_loop_model(n)
    mu ~ normal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, 1.0)
    end
    return mu
end

nanobs_caught(f) = begin
    err = try
        f()
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    sprint(showerror, err)
end

@testset "nan_observation_guard" begin
    nanobs_xs = [0.5, -0.25, 1.0, 0.0]

    @testset "NaN inside a vector observation throws, naming the address" begin
        bad = choicemap((:y, [0.1, NaN, -0.2, 0.4]))
        msg = nanobs_caught() do
            logjoint(nanobs_vec_model, [0.1], (nanobs_xs,), bad)
        end
        @test occursin("(:y,)", msg)
        @test occursin("NaN", msg)
        @test occursin("choicemap", msg)
    end

    @testset "NaN in a loop-indexed constraint throws, naming the index" begin
        bad = choicemap((:y => 1, 0.1), (:y => 2, 0.2), (:y => 3, NaN))
        msg = nanobs_caught() do
            logjoint(nanobs_loop_model, [0.1], (3,), bad)
        end
        @test occursin("(:y, 3)", msg)
        @test occursin("NaN", msg)
    end

    @testset "valid data is unchanged and the warm path stays cheap" begin
        cm = choicemap((:y, [0.1, -0.3, 0.7, 0.2]))
        cold = logjoint(nanobs_vec_model, [0.1], (nanobs_xs,), cm)
        warm = logjoint(nanobs_vec_model, [0.1], (nanobs_xs,), cm)
        @test cold == warm
        @test isfinite(cold)
        # the scan ran once and stamped the map: warm calls skip it entirely
        @test cm.nan_checked_mutation_count == cm.mutation_count
        # gross-regression guard only: the warm value path must not gain a
        # per-call rescan or per-observation boxing (exact counts are pinned
        # elsewhere; this bound is deliberately loose)
        nanobs_warm() = logjoint(nanobs_vec_model, [0.1], (nanobs_xs,), cm)
        nanobs_warm()
        @test (@allocated nanobs_warm()) < 100_000
    end

    @testset "mutating a scanned map re-arms the guard" begin
        cm = choicemap((:y => 1, 0.1), (:y => 2, 0.2), (:y => 3, 0.3))
        @test isfinite(logjoint(nanobs_loop_model, [0.1], (3,), cm))
        UncertainTea._pushchoice!(cm, :y => 2, NaN)
        msg = nanobs_caught() do
            logjoint(nanobs_loop_model, [0.1], (3,), cm)
        end
        @test occursin("(:y, 2)", msg)
    end

    @testset "Inf observations still score (no throw; -Inf is legitimate)" begin
        inf_vec = choicemap((:y, [0.1, Inf, -0.2, 0.4]))
        lj_vec = logjoint(nanobs_vec_model, [0.1], (nanobs_xs,), inf_vec)
        @test !isnan(lj_vec)
        @test lj_vec == -Inf

        inf_loop = choicemap((:y => 1, 0.1), (:y => 2, -Inf), (:y => 3, 0.3))
        lj_loop = logjoint(nanobs_loop_model, [0.1], (3,), inf_loop)
        @test !isnan(lj_loop)
        @test lj_loop == -Inf
    end
end
