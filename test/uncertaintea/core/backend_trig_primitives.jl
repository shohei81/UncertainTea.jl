# Trig/hyperbolic/expm1 backend primitives (issue #286): sin, cos, tan, tanh,
# sinh, cosh, atan, expm1 in deterministic bindings now stay on the fast path --
# backend-native scoring, hand-derived analytic batched gradients, and device
# lowering -- instead of dropping the whole model to the interpreter +
# ForwardDiff column fallback (previously the most common reason a realistic
# model with a link function or periodic feature lost the acceleration).

using KernelAbstractions: CPU

# every new primitive appears in a deterministic expression feeding a likelihood
@tea static function trig_prim_model(n)
    mu ~ normal(0.0, 1.0)
    logs ~ normal(-0.5, 0.5)
    loc =
        tanh(mu) + sin(0.5 * mu) - cos(mu) + 0.1 * tan(0.4 * mu) +
        0.05 * sinh(0.5 * mu) + 0.05 * cosh(0.3 * mu) + atan(mu) + 0.1 * expm1(0.2 * mu)
    for i = 1:n
        {:y => i} ~ normal(loc, exp(logs))
    end
    return mu
end

@testset "backend_trig_primitives" begin
    trig_cm = choicemap([(:y => i, 0.3 + 0.1 * i) for i = 1:6])

    @testset "the model stays on the fast paths" begin
        @test UncertainTea.backend_report(trig_prim_model).supported
        @test device_lowering_report(trig_prim_model)[1]
    end

    @testset "analytic batched gradient matches ForwardDiff per op mix" begin
        trig_params = reshape([0.3, -0.2, 0.7, 0.1, -1.1, 0.4], 2, 3)
        gb = batched_logjoint_gradient_unconstrained(trig_prim_model, trig_params, (6,), trig_cm)
        for col = 1:3
            gref = logjoint_gradient_unconstrained(trig_prim_model, trig_params[:, col], (6,), trig_cm)
            @test isapprox(gb[:, col], gref; rtol=1e-9, atol=1e-12)
        end
    end

    @testset "device lowering matches the analytic gradient at Float64" begin
        trig_params = reshape([0.3, -0.2, 0.7, 0.1], 2, 2)
        gb = batched_logjoint_gradient_unconstrained(trig_prim_model, trig_params, (6,), trig_cm)
        vals, grads = device_batched_logjoint_gradient(
            trig_prim_model, trig_params, (6,), trig_cm; backend=CPU(), precision=Float64,
        )
        @test isapprox(Array(grads), gb; rtol=1e-10)
        vref = batched_logjoint_unconstrained(trig_prim_model, trig_params, (6,), trig_cm)
        @test isapprox(Array(vals), vref; rtol=1e-10)
    end

    @testset "each unary op differentiates correctly in isolation" begin
        # one tiny model per op, analytic vs the single-chain ForwardDiff reference
        for (label, model) in (
            ("sin", @tea static function trig_sin(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(sin(mu), 0.5)
                end
                return mu
            end),
            ("cos", @tea static function trig_cos(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(cos(mu), 0.5)
                end
                return mu
            end),
            ("tan", @tea static function trig_tan(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(tan(0.3 * mu), 0.5)
                end
                return mu
            end),
            ("tanh", @tea static function trig_tanh(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(tanh(mu), 0.5)
                end
                return mu
            end),
            ("sinh", @tea static function trig_sinh(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(sinh(0.5 * mu), 0.5)
                end
                return mu
            end),
            ("cosh", @tea static function trig_cosh(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(cosh(0.5 * mu), 0.5)
                end
                return mu
            end),
            ("atan", @tea static function trig_atan(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(atan(mu), 0.5)
                end
                return mu
            end),
            ("expm1", @tea static function trig_expm1(n)
                mu ~ normal(0.0, 1.0)
                for i = 1:n
                    {:y => i} ~ normal(expm1(0.4 * mu), 0.5)
                end
                return mu
            end),
        )
            op_cm = choicemap([(:y => i, 0.1 * i) for i = 1:3])
            @test UncertainTea.backend_report(model).supported
            op_params = reshape([0.6, -0.8], 1, 2)
            gb = batched_logjoint_gradient_unconstrained(model, op_params, (3,), op_cm)
            for col = 1:2
                gref = logjoint_gradient_unconstrained(model, op_params[:, col], (3,), op_cm)
                @test isapprox(gb[:, col], gref; rtol=1e-9, atol=1e-12)
            end
        end
    end
end
