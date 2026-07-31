module UncertainTeaEnzymeExt

using UncertainTea
using Enzyme

# Add the reverse-mode method to the core `reverse_mode_gradient` function (which
# is declared method-less in UncertainTea). Issue #268 (follow-up to RFC #263).
#
# The prototype (bench/reverse_mode/) established the two incantations this needs:
#   * `Const(f)` -- the objective closes over constant data (inputs, observations)
#     and does not itself carry derivative data, so it must be passed as Const or
#     Enzyme cannot prove it readonly.
#   * `set_runtime_activity(Reverse)` -- tolerates the residual type-instability in
#     UncertainTea's scalar `logpdf` paths (e.g. the GP dense-Cholesky marginal).
# Encapsulating them here means callers just write `reverse_mode_gradient(f, x)`.
function UncertainTea.reverse_mode_gradient(f, x::AbstractVector)
    return only(Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), x))
end

# Value AND gradient in a single reverse pass (ReverseWithPrimal), for the batched
# per-column gradient path (issue #268, part A2): the sampler needs both the
# logjoint and its gradient every leapfrog leaf, and one primal+cotangent pass is
# cheaper than a separate value evaluation.
function UncertainTea.reverse_mode_value_and_gradient(f, x::AbstractVector)
    result = Enzyme.gradient(set_runtime_activity(Enzyme.ReverseWithPrimal), Enzyme.Const(f), x)
    return result.val, only(result.derivs)
end

end # module
