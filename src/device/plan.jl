# Device execution plan: an `isbits` mirror of the second-stage BackendExecutionPlan.
#
# Design notes
# ------------
# * Addresses are fully ERASED. Observed choice values are resolved on the host
#   (staging.jl) into a dense `observed[row, col]` matrix; the kernel walks the
#   plan maintaining an observation cursor that advances identically to staging.
# * Primitive operators are lifted to a type parameter (`DevicePrimitiveExpr{Op}`,
#   `Op::Symbol`) so the kernel dispatches at compile time with no runtime branch.
# * `Union{Nothing,Int}` is avoided everywhere (it is not an isbits layout). Slots
#   use `Int32` with `0` meaning "no binding". A choice step's `value_source::Int32`
#   encodes the value origin: `> 0` -> unconstrained parameter row (latent);
#   `< 0` -> observed (value comes from the staged observation cursor).
# * Loops are non-nested and carry a static `loop_id::Int32`; the trip count and
#   start index are staged per loop into device Int32 buffers.

include("plan/exprs.jl")
include("plan/steps.jl")
include("plan/lowering.jl")
