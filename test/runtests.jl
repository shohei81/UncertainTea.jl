using Test
using Random
using UncertainTea

# Internal (unexported) names used across the suite (issue #283). fixtures.jl
# repeats this for standalone core runs; sampling.jl relies on this import via the
# shared include scope.
using UncertainTea:
    AddressSpec, ChoiceSpec, ModelSpec, AddressLiteralPart, AddressDynamicPart,
    DistributionSpec, GenerativeCallSpec, RawChoiceRhsSpec, BroadcastDistributionSpec,
    LoopScopeSpec, ParameterLayout, ParameterSlotSpec, ExecutionPlan, ChoicePlanStep,
    DeterministicPlanStep, LoopPlanStep, executionplan, IdentityTransform,
    VectorIdentityTransform, LogTransform, LogitTransform, SimplexTransform,
    CholeskyCorrTransform, CholeskyCovTransform, VectorLogTransform, VectorLogitTransform,
    BoundedTransform, LowerBoundedTransform, UpperBoundedTransform, StaticMode, DynamicMode,
    modelspec, isstaticaddress, isaddresstemplate, isrepeatedchoice, hasrepeatedchoices,
    parameterlayout, parametercount, parametervaluecount, BackendExecutionPlan,
    BackendLoweringReport, backend_report, backend_execution_plan

@testset "UncertainTea" begin
    include("uncertaintea/core.jl")
    if get(ENV, "UNCERTAINTEA_TEST_GROUP", "all") in ("all", "sampling")
        include("uncertaintea/sampling.jl")
    end
end
