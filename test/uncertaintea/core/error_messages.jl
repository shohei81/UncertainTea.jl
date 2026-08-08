# Actionable error messages (issue #316): the worst live-tested offenders now
# name the problem in user terms and say what to do instead. Macro-expansion
# errors under @eval arrive wrapped in LoadError; the message assertions
# unwrap to the underlying ArgumentError.

_emsg_unwrap(err) = err isa LoadError ? _emsg_unwrap(err.error) : err

function _emsg_thrown(f)
    try
        f()
    catch err
        return _emsg_unwrap(err)
    end
    return nothing
end

@testset "error_messages" begin
    @testset "iid with a runtime count says why and what to do" begin
        err = _emsg_thrown() do
            @eval @tea static function emsg_iid_runtime(n)
                mu ~ normal(0.0, 1.0)
                {:y} ~ iid(normal(mu, 1.0), n)
                return mu
            end
        end
        @test err isa ArgumentError
        # no self-parody: the message must not read "got `n`" without context
        @test occursin("literal integer", err.msg)
        @test occursin("static `for` loop", err.msg)   # points at the loop alternative
        @test occursin("model argument", err.msg)
        # ... and at the runtime-length mvnormal spelling for the iid-normal case
        @test occursin("mvnormal(zeros(n), ones(n))", err.msg)
        @test occursin("issue #289", err.msg)
    end

    @testset "unsupported ~ left-hand side lists the valid spellings" begin
        err = _emsg_thrown() do
            @eval @tea static function emsg_bad_lhs()
                [1, 2] ~ normal(0.0, 1.0)
                return 0.0
            end
        end
        @test err isa ArgumentError
        @test occursin("Supported", err.msg)
        @test occursin("{:addr} ~ dist(...)", err.msg)
        @test occursin("{:addr => i} ~ dist(...)", err.msg)
    end

    @testset "capitalized Distributions.jl spellings get a did-you-mean" begin
        for (bad, good) in ((:Normal, :normal), (:Poisson, :poisson), (:Beta, :beta))
            err = _emsg_thrown() do
                @eval @tea static function $(Symbol(:emsg_cap_, good))()
                    mu ~ $bad(0.0, 1.0)
                    return mu
                end
            end
            @test err isa ArgumentError
            @test occursin("did you mean", err.msg)
            @test occursin("`$(good)`", err.msg)
        end

        # dynamic mode hits the same check on its rewrite path
        err = _emsg_thrown() do
            @eval @tea function emsg_cap_dynamic()
                mu ~ Normal(0.0, 1.0)
                return mu
            end
        end
        @test err isa ArgumentError
        @test occursin("did you mean", err.msg)

        # a capitalized call whose lowercase is NOT a known family does not
        # trigger the did-you-mean: it keeps its existing meaning (a generative
        # subcall, which requires a TeaModel callee)
        @eval MyDist(mu) = UncertainTea.normal(mu, 1.0)
        err = _emsg_thrown() do
            @eval @tea function emsg_userfun()
                mu ~ MyDist(0.0)
                return mu
            end
        end
        @test err isa ArgumentError
        @test !occursin("did you mean", err.msg)
        @test occursin("TeaModel", err.msg)
    end

    @testset "partially constrained repeated site names the remedy" begin
        @tea static function emsg_loop_obs(n)
            mu ~ normal(0.0, 1.0)
            for i = 1:n
                {:y => i} ~ normal(mu, 1.0)
            end
            return mu
        end
        # constraining index 1 but not 2/3 classifies the whole site observed
        partial = choicemap((:y => 1, 0.5))
        err = _emsg_thrown() do
            logjoint(emsg_loop_obs, [0.0], (3,), partial)
        end
        @test err isa ArgumentError
        @test occursin("every index needs a value", err.msg)
    end
end
