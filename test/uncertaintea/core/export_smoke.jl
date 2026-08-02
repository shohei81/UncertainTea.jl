# Export smoke test (issue #334): every name exported from the top-level module
# and the three facade submodules (Inference / Diagnostics / Device) must
# actually resolve. This catches exports whose definition was deleted or
# renamed (the cycle-3 audit found 17 exported names no test ever touched:
# TeaTrace, ADVIResult, WAICResult, LOOResult, MAPResult, LaplaceResult,
# EllipticalSliceResult, SMCStageSummary, HMCParameterSummary, ...). It only
# asserts resolution — construct-or-call coverage stays with the topical tests.

@testset "export smoke: every exported name resolves (issue #334)" begin
    modules = (
        UncertainTea,
        UncertainTea.Inference,
        UncertainTea.Diagnostics,
        UncertainTea.Device,
    )
    for mod in modules
        exported = setdiff(names(mod), [nameof(mod)])
        @testset "$(mod)" begin
            for name in exported
                @test isdefined(mod, name)
                binding = Base.Docs.Binding(mod, name)
                @test Base.Docs.defined(binding)
                @test Base.Docs.resolve(binding) === getfield(mod, name)
            end
        end
    end
end
