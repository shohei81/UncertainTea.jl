# Guards the per-model plan-BUILD de-specialization (issue #155 part 2).
#
# The compiled-plan builders (`evaluator.jl`) and backend lowering
# (`backend/lowering/*.jl`) run ONCE per (model, signature) -- their results are
# memoized on the model -- but used to recompile for every distinct
# `TeaModel{M,F,S}` because the `impl` type `F` is unique per `@tea` model,
# re-paying ~7-11 s of JIT per new model (the tax #155 measured). We mark the
# `model` argument `@nospecialize` on those builders so they compile a SINGLE
# time against the abstract `TeaModel` and are reused across models, without
# changing the emitted plan or any numeric result (the generated scorer #144 and
# the hot scoring/gradient kernels still specialize on the CONCRETE plan).
#
# This test asserts the `@nospecialize` annotation is present on the key
# builders. It intentionally checks the annotation, not a wall-time: TTFX is
# environment-sensitive, but a dropped `@nospecialize` would silently reintroduce
# the per-model recompilation this change removed. `Method.nospecialize` is a
# per-argument bitmask (bit i set == argument i is `@nospecialize`d).
@testset "plan_build_nospecialize" begin
    U = UncertainTea

    # (function, positional-arg types, 1-based index of the `model` argument)
    build_entries = [
        (U._compile_plan_expr, Tuple{U.TeaModel,U.EnvironmentLayout,Any}, 1),
        (U._compile_execution_plan, Tuple{U.TeaModel,U.ExecutionPlan}, 1),
        (U._compile_address, Tuple{U.EnvironmentLayout,U.TeaModel,U.AddressSpec}, 2),
        (U._resolve_signature_plan, Tuple{U.TeaModel,Set{U.Address}}, 1),
        (U._backend_lower_expr, Tuple{U.TeaModel,U.EnvironmentLayout,Any,Vector{String},String}, 1),
        (U._backend_lower_address, Tuple{U.TeaModel,U.EnvironmentLayout,U.AddressSpec,Vector{String}}, 1),
        (U._lower_backend_execution_plan, Tuple{U.TeaModel,U.ExecutionPlan}, 1),
    ]

    for (fn, sig, model_arg) in build_entries
        m = which(fn, sig)
        # bit `model_arg` of the nospecialize mask must be set
        @test (m.nospecialize & (1 << (model_arg - 1))) != 0
    end
end
