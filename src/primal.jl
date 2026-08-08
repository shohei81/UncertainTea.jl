# Primal-value extraction for boundary comparisons (ForwardDiff 1.x compat).
#
# ForwardDiff 1.x compares Duals LEXICOGRAPHICALLY: at a value tie the partials
# break the tie, so `Dual(1.0, 8.5e-17) <= 1` is false (ForwardDiff 0.10
# compared values only). UncertainTea's saturation machinery (issues #343/#354)
# deliberately produces values EXACTLY on support boundaries with nonzero
# partials -- sigmoid(theta) rounds to 1.0 for theta >~ 36.74 while its
# derivative (~8.5e-17) is still representable -- so boundary-inclusive
# validations like `0 <= p <= 1` and support checks like `x > 0` must judge the
# PRIMAL value: a perturbation direction must never flip support membership or
# parameter validity. Every such comparison site routes its Dual-reachable
# operands through `_primal`. Under ForwardDiff 0.10 primal comparison is
# exactly what `<`/`==` already did, so this changes nothing there.
_primal(x::Real) = x
_primal(d::ForwardDiff.Dual) = _primal(ForwardDiff.value(d))
