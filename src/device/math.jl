# Device-safe, T-generic scalar log-densities used by the fused KernelAbstractions
# logjoint kernel. Every function here must be callable inside a GPU kernel:
#   * no exceptions / no error paths (out-of-support returns -Inf through ifelse),
#   * no allocations,
#   * no dependency on SpecialFunctions (a pure-Julia Lanczos loggamma is provided).
# Non-finite parameters (e.g. sigma <= 0) propagate as NaN naturally, matching the
# "garbage in -> NaN out" contract; the authoritative CPU backend still validates.

# Lanczos approximation coefficients (g = 7, n = 9). Stored as Float64 and cast to
# the working precision T on demand.
const _DEVICE_LANCZOS_G = 7.0
const _DEVICE_LANCZOS_COEFS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

"""
    _device_loggamma(x::T) -> T

Pure-Julia log-gamma via the Lanczos approximation, valid on device. Uses the
reflection formula for `x < 0.5`. Coefficients are Float64 constants cast to `T`.
"""
@inline function _device_loggamma(x::T) where {T}
    if x < T(0.5)
        # reflection: loggamma(x) = log(pi / sin(pi x)) - loggamma(1 - x)
        return T(log(pi)) - log(abs(sin(T(pi) * x))) - _device_loggamma(one(T) - x)
    end
    xm1 = x - one(T)
    a = T(_DEVICE_LANCZOS_COEFS[1])
    t = xm1 + T(_DEVICE_LANCZOS_G) + T(0.5)
    a += T(_DEVICE_LANCZOS_COEFS[2]) / (xm1 + one(T))
    a += T(_DEVICE_LANCZOS_COEFS[3]) / (xm1 + T(2))
    a += T(_DEVICE_LANCZOS_COEFS[4]) / (xm1 + T(3))
    a += T(_DEVICE_LANCZOS_COEFS[5]) / (xm1 + T(4))
    a += T(_DEVICE_LANCZOS_COEFS[6]) / (xm1 + T(5))
    a += T(_DEVICE_LANCZOS_COEFS[7]) / (xm1 + T(6))
    a += T(_DEVICE_LANCZOS_COEFS[8]) / (xm1 + T(7))
    a += T(_DEVICE_LANCZOS_COEFS[9]) / (xm1 + T(8))
    half_log_2pi = T(0.9189385332046727) # 0.5*log(2*pi)
    return half_log_2pi + (xm1 + T(0.5)) * log(t) - t + log(a)
end

@inline _device_neginf(::Type{T}) where {T} = T(-Inf)

# log Gamma(x + 1/2) - log Gamma(x), computed WITHOUT differencing two large
# loggammas: at large x each loggamma is ~ x log x, so their O(log x) difference
# drowns in rounding (0.06 absolute at Float32 for x ~ 5e4, visible in any
# large-nu Student-t density or CDF). Below the precision-dependent crossover
# the direct difference is more accurate; above it, the Stirling-series form
#   x log1p(1/(2x)) + log(x)/2 - 1/2 + (1/(x+1/2) - 1/x)/12
# carries relative error ~1/(360 x^3).
@inline _device_loggamma_half_ratio_crossover(::Type{Float32}) = 12.0f0
@inline _device_loggamma_half_ratio_crossover(::Type{T}) where {T} = T(1200)
@inline _device_loggamma_half_ratio_crossover(::Type{DeviceDual{T}}) where {T} =
    DeviceDual{T}(_device_loggamma_half_ratio_crossover(T), zero(T))

@inline function _device_loggamma_half_ratio(x::T) where {T}
    if x > _device_loggamma_half_ratio_crossover(T)
        return x * log1p(one(T) / (T(2) * x)) + log(x) / T(2) - T(0.5) +
               (one(T) / (x + T(0.5)) - one(T) / x) / T(12)
    end
    return _device_loggamma(x + T(0.5)) - _device_loggamma(x)
end

@inline function _device_normal_logpdf(mu::T, sigma::T, x::T) where {T}
    z = (x - mu) / sigma
    return -log(sigma) - T(0.9189385332046727) - z * z / T(2)
end

@inline function _device_lognormal_logpdf(mu::T, sigma::T, x::T) where {T}
    lx = log(x)
    base = -log(sigma) - T(0.9189385332046727) - ((lx - mu) / sigma)^2 / T(2) - lx
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_exponential_logpdf(rate::T, x::T) where {T}
    return ifelse(x >= zero(T), log(rate) - rate * x, _device_neginf(T))
end

@inline function _device_gamma_logpdf(shape::T, rate::T, x::T) where {T}
    base = shape * log(rate) - _device_loggamma(shape) +
           (shape - one(T)) * log(x) - rate * x
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_laplace_logpdf(loc::T, scale::T, x::T) where {T}
    return -log(T(2) * scale) - abs(x - loc) / scale
end

# scalar prior families (issue #229). Constants: log(pi) = 1.1447298858494002,
# log(2) = 0.6931471805599453, log(2pi)/2 = 0.9189385332046727.
@inline function _device_cauchy_logpdf(mu::T, sigma::T, x::T) where {T}
    z = (x - mu) / sigma
    return -T(1.1447298858494002) - log(sigma) - log1p(z * z)
end

@inline function _device_halfnormal_logpdf(sigma::T, x::T) where {T}
    z = x / sigma
    base = T(0.6931471805599453) - log(sigma) - T(0.9189385332046727) - z * z / T(2)
    return ifelse(x >= zero(T), base, _device_neginf(T))
end

@inline function _device_halfcauchy_logpdf(scale::T, x::T) where {T}
    z = x / scale
    base = T(0.6931471805599453) - T(1.1447298858494002) - log(scale) - log1p(z * z)
    return ifelse(x >= zero(T), base, _device_neginf(T))
end

@inline function _device_logistic_logpdf(mu::T, scale::T, x::T) where {T}
    z = (x - mu) / scale
    az = abs(z)
    return -log(scale) - az - T(2) * log1p(exp(-az))
end

@inline function _device_gumbel_logpdf(mu::T, scale::T, x::T) where {T}
    z = (x - mu) / scale
    return -log(scale) - z - exp(-z)
end

# positive-support / heavy-tail families (issue #230). log(2pi)/2 = 0.9189385332046727.
@inline function _device_frechet_logpdf(shape::T, scale::T, x::T) where {T}
    logz = log(x) - log(scale)
    base = log(shape) - log(scale) - (one(T) + shape) * logz - exp(-shape * logz)
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_rayleigh_logpdf(scale::T, x::T) where {T}
    base = log(x) - T(2) * log(scale) - x * x / (T(2) * scale * scale)
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_inversegaussian_logpdf(mu::T, lambda::T, x::T) where {T}
    d = x - mu
    base = T(0.5) * log(lambda) - T(0.9189385332046727) - T(1.5) * log(x) - lambda * d * d / (T(2) * mu * mu * x)
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_beta_logpdf(alpha::T, beta::T, x::T) where {T}
    logbeta = _device_loggamma(alpha) + _device_loggamma(beta) - _device_loggamma(alpha + beta)
    base = (alpha - one(T)) * log(x) + (beta - one(T)) * log1p(-x) - logbeta
    return ifelse((x > zero(T)) & (x < one(T)), base, _device_neginf(T))
end

@inline function _device_studentt_logpdf(nu::T, mu::T, sigma::T, x::T) where {T}
    z = (x - mu) / sigma
    # loggamma((nu+1)/2) - loggamma(nu/2) via the half-ratio (large-nu safe)
    return _device_loggamma_half_ratio(nu / T(2)) -
           (log(nu) + T(1.1447298858494002)) / T(2) - log(sigma) - # log(pi)
           (nu + one(T)) * log1p(z * z / nu) / T(2)
end

@inline function _device_inversegamma_logpdf(shape::T, scale::T, x::T) where {T}
    base = shape * log(scale) - _device_loggamma(shape) -
           (shape + one(T)) * log(x) - scale / x
    return ifelse(x > zero(T), base, _device_neginf(T))
end

@inline function _device_weibull_logpdf(shape::T, scale::T, x::T) where {T}
    base = log(shape) + (shape - one(T)) * log(x) -
           shape * log(scale) - exp(shape * (log(x) - log(scale)))
    at_zero = ifelse(shape < one(T), T(Inf), ifelse(shape == one(T), -log(scale), _device_neginf(T)))
    return ifelse(x > zero(T), base, ifelse(x == zero(T), at_zero, _device_neginf(T)))
end

# Support mirrors the CPU `_bernoulli_value` normalization (issue #85): only
# values equal to 0 or 1 are in support; anything else scores -Inf instead of
# throwing (exception-free device contract).
@inline function _device_bernoulli_logpdf(p::T, x::T) where {T}
    base = ifelse(x > T(0.5), log(p), log1p(-p))
    in_support = (x == zero(T)) | (x == one(T))
    return ifelse(in_support, base, _device_neginf(T))
end

# Numerically stable `log(1 + exp(eta))` (issues #149/#150), mirroring the CPU
# `_bernoullilogit_log1p_exp`: the branch keeps the exponentiated argument
# non-positive so it never overflows (finite at |eta| = 90). Exception-free
# (`ifelse`) for the device; both branches are evaluated but only one selected,
# and neither can produce a NaN on a finite eta.
@inline _device_log1p_exp(eta::T) where {T} =
    ifelse(eta > zero(T), eta + log1p(exp(-eta)), log1p(exp(eta)))

# Logit-parameterized Bernoulli logpdf `x*eta - log1p(exp(eta))` (issues
# #149/#150), mirroring the CPU `_backend_bernoullilogit_logpdf`. Support matches
# `_device_bernoulli_logpdf` (only 0/1 in support; anything else scores -Inf under
# the exception-free device contract). The well-conditioned gradient
# `d/d_eta = x - logistic(eta)` falls out of forward-mode through this closed form
# (log1p_exp differentiates to the logistic), so the gradient kernel needs no
# special seeding beyond the duals that already flow through `eta`.
@inline function _device_bernoullilogit_logpdf(eta::T, x::T) where {T}
    log_normalizer = _device_log1p_exp(eta)
    base = x * eta - log_normalizer
    in_support = (x == zero(T)) | (x == one(T))
    return ifelse(in_support, base, _device_neginf(T))
end

# Nonnegative exact-integer support check shared by the count families; mirrors the
# CPU `_poisson_count` acceptance (staged integer counts are exact in Float32/64).
@inline _device_count_ok(x::T, k::T) where {T} = (x >= zero(T)) & (x == k)

@inline function _device_poisson_logpdf(lambda::T, x::T) where {T}
    k = round(x)
    base = k * log(lambda) - lambda - _device_loggamma(k + one(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

# Base real type behind a plain real or a DeviceDual (the gradient kernel's p).
@inline _device_real_basetype(::Type{DeviceDual{T}}) where {T} = T
@inline _device_real_basetype(::Type{T}) where {T<:Real} = T

# Exception-free count round: a non-finite / out-of-Int64-range value maps to -1
# (out of support -> -Inf) instead of throwing an InexactError inside the kernel.
@inline function _device_count_int(x::Real)
    (isfinite(x) & (abs(x) < 9.0e15)) || return Int64(-1)
    return round(Int64, x)
end

# Exact-integer log binomial coefficient computed in Float64 (integers <= 2^53 are
# exact), so the n-vs-k distinction survives regardless of the compute precision
# T (issue #71).
@inline function _device_log_binom_coeff(n::Integer, k::Integer)
    return _device_loggamma(Float64(n) + 1.0) - _device_loggamma(Float64(k) + 1.0) -
           _device_loggamma(Float64(n - k) + 1.0)
end

# `n` (trials) and `k` (count) are EXACT integers threaded through the integer
# observation buffer (issue #71); `p` is a plain real (logjoint kernel) or a
# `DeviceDual` (gradient kernel). The p-independent combinatorial term is formed
# in Float64 and folded in as a zero-derivative constant, so the difference
# `n - k` stays exact and the gradient flows only through the p-dependent terms
# whose integer coefficients are exact.
@inline function _device_binomial_logpdf(n::Integer, k::Integer, p::P) where {P}
    T = _device_real_basetype(P)
    in_support = (k >= 0) & (n >= 0) & (k <= n)
    log_combination = convert(P, T(_device_log_binom_coeff(n, k)))
    # guard the k == 0 / k == n corners so p in {0, 1} cannot produce 0 * -Inf
    base = log_combination + ifelse(k > 0, k * log(p), zero(p)) + ifelse(k < n, (n - k) * log1p(-p), zero(p))
    return ifelse(in_support, base, oftype(base, -Inf))
end

@inline function _device_geometric_logpdf(p::T, x::T) where {T}
    k = round(x)
    base = log(p) + ifelse(k > zero(T), k * log1p(-p), zero(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

@inline function _device_negativebinomial_logpdf(successes::T, p::T, x::T) where {T}
    k = round(x)
    base =
        _device_loggamma(k + successes) - _device_loggamma(successes) -
        _device_loggamma(k + one(T)) + successes * log(p) +
        ifelse(k > zero(T), k * log1p(-p), zero(T))
    return ifelse(_device_count_ok(x, k), base, _device_neginf(T))
end

# Compile-time unrolled selection of log(probabilities[k]); unmatched entries
# contribute an exact zero so heterogeneous (literal/dual) tuples promote safely.
@inline _device_categorical_pick(::Tuple{}, k, index::Int32, acc) = acc
@inline function _device_categorical_pick(probabilities::Tuple, k, index::Int32, acc)
    logp = log(first(probabilities))
    term = ifelse(k == index, logp, zero(logp))
    return _device_categorical_pick(Base.tail(probabilities), k, index + Int32(1), acc + term)
end

# Value-channel probability of a tuple entry (plain-`T` in the logjoint kernel,
# `DeviceDual{T}` in the gradient kernel); validation reads only the value.
@inline _device_prob_value(x::DeviceDual) = x.value
@inline _device_prob_value(x) = x

# Fold the probability vector into (all entries in [0, 1], running sum), reading
# the value channel so duals and literals validate identically.
@inline _device_categorical_validate(::Tuple{}, acc::T, ok::Bool) where {T} = (ok, acc)
@inline function _device_categorical_validate(probabilities::Tuple, acc::T, ok::Bool) where {T}
    p = T(_device_prob_value(first(probabilities)))
    ok2 = ok & (p >= zero(T)) & (p <= one(T))
    return _device_categorical_validate(Base.tail(probabilities), acc + p, ok2)
end

# Mirrors the CPU `_backend_categorical_logpdf` (src/backend/scoring/discrete.jl):
# besides the observed-category range check, each probability must lie in [0, 1]
# and the vector must sum to 1 (within tolerance). Invalid vectors map to -Inf per
# the exception-free device contract instead of throwing (issue #84).
@inline function _device_categorical_logpdf(probabilities::Tuple, x::T) where {T}
    # validation runs on the base real type (`eps`/comparisons are value-channel),
    # so it works whether `x`/the probabilities are plain reals or `DeviceDual`s.
    TB = _device_real_basetype(T)
    k = round(x)
    in_support = (x >= one(T)) & (x <= T(length(probabilities))) & (x == k)
    in_range, total_prob = _device_categorical_validate(probabilities, zero(TB), true)
    tolerance = sqrt(eps(one(TB))) * TB(max(length(probabilities), 1) * 8)
    normalized = abs(total_prob - one(TB)) <= tolerance
    total = _device_categorical_pick(probabilities, k, Int32(1), zero(T))
    return ifelse(in_support & in_range & normalized, total, oftype(total, -Inf))
end

# ---- erf / erfc (W. J. Cody, SPECFUN CALERF rational approximations) ------------
# Three-interval rational forms, |relative error| ~ 1e-16 at Float64, built from
# branches and polynomials only (device-safe). The DeviceDual overloads below
# differentiate through the ANALYTIC derivative (2/sqrt(pi) * exp(-x^2)), not the
# approximation, so device gradients match the CPU analytic-gradient formulas.

const _DEVICE_ERF_A = (
    3.16112374387056560e0,
    1.13864154151050156e2,
    3.77485237685302021e2,
    3.20937758913846947e3,
    1.85777706184603153e-1,
)
const _DEVICE_ERF_B = (2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3, 2.84423683343917062e3)
const _DEVICE_ERF_C = (
    5.64188496988670089e-1,
    8.88314979438837594e0,
    6.61191906371416295e1,
    2.98635138197400131e2,
    8.81952221241769090e2,
    1.71204761263407058e3,
    2.05107837782607147e3,
    1.23033935479799725e3,
    2.15311535474403846e-8,
)
const _DEVICE_ERF_D = (
    1.57449261107098347e1,
    1.17693950891312499e2,
    5.37181101862009858e2,
    1.62138957456669019e3,
    3.29079923573345963e3,
    4.36261909014324716e3,
    3.43936767414372164e3,
    1.23033935480374942e3,
)
const _DEVICE_ERF_P = (
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
)
const _DEVICE_ERF_Q = (
    2.56852019228982242e0,
    1.87295284992346047e0,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
)

# erfc(y) * exp(y^2) for y > 0.46875 (the two outer Cody intervals), without the
# final exp(-y^2) scaling.
@inline function _device_erfcx_tail(y::T) where {T}
    if y <= T(4)
        xnum = T(_DEVICE_ERF_C[9]) * y
        xden = y
        xnum = (xnum + T(_DEVICE_ERF_C[1])) * y
        xden = (xden + T(_DEVICE_ERF_D[1])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[2])) * y
        xden = (xden + T(_DEVICE_ERF_D[2])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[3])) * y
        xden = (xden + T(_DEVICE_ERF_D[3])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[4])) * y
        xden = (xden + T(_DEVICE_ERF_D[4])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[5])) * y
        xden = (xden + T(_DEVICE_ERF_D[5])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[6])) * y
        xden = (xden + T(_DEVICE_ERF_D[6])) * y
        xnum = (xnum + T(_DEVICE_ERF_C[7])) * y
        xden = (xden + T(_DEVICE_ERF_D[7])) * y
        return (xnum + T(_DEVICE_ERF_C[8])) / (xden + T(_DEVICE_ERF_D[8]))
    end
    inv_y2 = one(T) / (y * y)
    xnum = T(_DEVICE_ERF_P[6]) * inv_y2
    xden = inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[1])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[1])) * inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[2])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[2])) * inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[3])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[3])) * inv_y2
    xnum = (xnum + T(_DEVICE_ERF_P[4])) * inv_y2
    xden = (xden + T(_DEVICE_ERF_Q[4])) * inv_y2
    result = inv_y2 * (xnum + T(_DEVICE_ERF_P[5])) / (xden + T(_DEVICE_ERF_Q[5]))
    return (T(5.6418958354775628695e-1) - result) / y # 1/sqrt(pi)
end

@inline function _device_erfc_positive(y::T) where {T}
    # split exp(-y^2) as exp(-ysq^2) * exp(-del) with ysq = trunc(16y)/16 (Cody)
    ysq = trunc(y * T(16)) / T(16)
    del = (y - ysq) * (y + ysq)
    return exp(-ysq * ysq) * exp(-del) * _device_erfcx_tail(y)
end

@inline function _device_erf(x::T) where {T}
    y = abs(x)
    if y <= T(0.46875)
        ysq = ifelse(y > T(1.11e-16), y * y, zero(T))
        xnum = T(_DEVICE_ERF_A[5]) * ysq
        xden = ysq
        xnum = (xnum + T(_DEVICE_ERF_A[1])) * ysq
        xden = (xden + T(_DEVICE_ERF_B[1])) * ysq
        xnum = (xnum + T(_DEVICE_ERF_A[2])) * ysq
        xden = (xden + T(_DEVICE_ERF_B[2])) * ysq
        xnum = (xnum + T(_DEVICE_ERF_A[3])) * ysq
        xden = (xden + T(_DEVICE_ERF_B[3])) * ysq
        return x * (xnum + T(_DEVICE_ERF_A[4])) / (xden + T(_DEVICE_ERF_B[4]))
    end
    y > T(26.5) && return ifelse(x > zero(T), one(T), -one(T))
    result = one(T) - _device_erfc_positive(y)
    return ifelse(x < zero(T), -result, result)
end

@inline function _device_erfc(x::T) where {T}
    y = abs(x)
    y <= T(0.46875) && return one(T) - _device_erf(x)
    # no explicit big-argument cutoff: exp(-ysq^2) underflows naturally, so
    # representable subnormal tails (e.g. erfc(26.51) ~ 1.3e-307 at Float64)
    # survive instead of collapsing to zero before the type does
    isinf(y) && return ifelse(x > zero(T), zero(T), T(2))
    result = _device_erfc_positive(y)
    return ifelse(x < zero(T), T(2) - result, result)
end

# Analytic-derivative duals: d/dx erf(x) = 2/sqrt(pi) exp(-x^2). An infinite
# value (an unbounded truncation side) is pinned to a zero derivative -- its
# incoming `deriv` can itself be infinite (Inf flowed through the standardizing
# division), and 0 * Inf would poison the gradient with NaN; this mirrors the
# CPU analytic gradient's isinf guards.
@inline function _device_erf(a::DeviceDual{T}) where {T}
    v = a.value
    deriv = ifelse(isinf(v), zero(T), T(1.1283791670955126) * exp(-v * v) * a.deriv)
    return DeviceDual{T}(_device_erf(v), deriv)
end
@inline function _device_erfc(a::DeviceDual{T}) where {T}
    v = a.value
    deriv = ifelse(isinf(v), zero(T), -T(1.1283791670955126) * exp(-v * v) * a.deriv)
    return DeviceDual{T}(_device_erfc(v), deriv)
end

# ---- regularized incomplete beta / Student-t CDF ---------------------------------
# Continued fraction (Lentz) for I_x(a, b), mirroring the Numerical Recipes form.
# Called with plain reals only: the DeviceDual overload of `_device_std_t_cdf`
# differentiates via the analytic d/dz = t-pdf(z) instead.

@inline function _device_betacf(a::T, b::T, x::T) where {T}
    qab = a + b
    qap = a + one(T)
    qam = a - one(T)
    fpmin = T(1e-30)
    c = one(T)
    d = one(T) - qab * x / qap
    abs(d) < fpmin && (d = fpmin)
    d = one(T) / d
    h = d
    for m = 1:200
        mf = T(m)
        m2 = 2 * mf
        aa = mf * (b - mf) * x / ((qam + m2) * (a + m2))
        d = one(T) + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = one(T) + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = one(T) / d
        h *= d * c
        aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
        d = one(T) + aa * d
        abs(d) < fpmin && (d = fpmin)
        c = one(T) + aa / c
        abs(c) < fpmin && (c = fpmin)
        d = one(T) / d
        delta = d * c
        h *= delta
        abs(delta - one(T)) <= T(4) * eps(T) && break
    end
    return h
end

# `x` and `y` are the caller-computed pair with x + y == 1 in exact arithmetic;
# passing both lets the caller keep each accurate near its own zero (recomputing
# `1 - x` at x ~ 1 destroys the low bits Float32 needs, e.g. the Student-t CDF
# at small |z|).
@inline function _device_beta_inc_reg_parts(a::T, b::T, x::T, y::T) where {T}
    x <= zero(T) && return zero(T)
    y <= zero(T) && return one(T)
    logx = ifelse(x < T(0.5), log(x), log1p(-y))
    logy = ifelse(y < T(0.5), log(y), log1p(-x))
    # the Student-t CDF calls this with b == 1/2, where loggamma(a+b) -
    # loggamma(a) is exactly the half-ratio; computing it as a ratio keeps
    # large-a (large-nu) accuracy that the difference of two huge loggammas
    # loses. Other b fall back to the direct difference.
    log_gamma_ratio = ifelse(
        b == T(0.5),
        _device_loggamma_half_ratio(a),
        _device_loggamma(a + b) - _device_loggamma(a),
    )
    front = exp(a * logx + b * logy + log_gamma_ratio - _device_loggamma(b))
    if x < (a + one(T)) / (a + b + T(2))
        return front * _device_betacf(a, b, x) / a
    end
    return one(T) - front * _device_betacf(b, a, y) / b
end

@inline _device_beta_inc_reg(a::T, b::T, x::T) where {T} = _device_beta_inc_reg_parts(a, b, x, one(T) - x)

# log I_x(a, b) computed IN LOG SPACE on the direct branch (x small), so a tail
# probability that underflows as a plain value (light Student-t tails at large
# nu, Float32) still yields a finite log. The symmetric branch is the
# complement of a small quantity and is safe as log1p.
@inline function _device_log_beta_inc_reg_parts(a::T, b::T, x::T, y::T) where {T}
    x <= zero(T) && return _device_neginf(T)
    y <= zero(T) && return zero(T)
    logx = ifelse(x < T(0.5), log(x), log1p(-y))
    logy = ifelse(y < T(0.5), log(y), log1p(-x))
    log_gamma_ratio = ifelse(
        b == T(0.5),
        _device_loggamma_half_ratio(a),
        _device_loggamma(a + b) - _device_loggamma(a),
    )
    log_front = a * logx + b * logy + log_gamma_ratio - _device_loggamma(b)
    if x < (a + one(T)) / (a + b + T(2))
        return log_front + log(_device_betacf(a, b, x)) - log(a)
    end
    return log1p(-exp(log_front) * _device_betacf(b, a, y) / b)
end

# log(T_cdf(z; nu)), staying in log space so light tails survive underflow.
# Both signs route through the SINGLE log-space incomplete beta (the z > 0
# complement exponentiates it, which is safe: I_x <= 1), keeping the number of
# inlined `_device_betacf` loop instantiations per kernel low -- duplicating
# the plain-value path here made Metal's shader compiler blow up for minutes
# on the fused truncated kernels.
@inline function _device_std_t_log_cdf(z::T, nu::T) where {T}
    isinf(z) && return ifelse(z > zero(T), zero(T), _device_neginf(T))
    z == zero(T) && return -log(T(2))
    denominator = nu + z * z
    x = nu / denominator
    y = z * z / denominator
    log_regularized = _device_log_beta_inc_reg_parts(nu / T(2), T(0.5), x, y)
    if z < zero(T)
        return log_regularized - log(T(2))
    end
    return log1p(-exp(log_regularized) / T(2))
end

# d/dz log cdf = exp(log pdf - log cdf), computed in log space so a tail whose
# plain cdf underflows still gets a finite derivative ratio.
@inline function _device_std_t_log_cdf(z::DeviceDual{T}, nu::Real) where {T}
    nu_value = convert(T, nu)
    v = z.value
    log_cdf = _device_std_t_log_cdf(v, nu_value)
    log_pdf = _device_studentt_logpdf(nu_value, zero(T), one(T), v)
    deriv = ifelse(isinf(v), zero(T), exp(log_pdf - log_cdf) * z.deriv)
    return DeviceDual{T}(log_cdf, deriv)
end
@inline _device_std_t_log_cdf(z::DeviceDual{T}, nu::DeviceDual{T}) where {T} = _device_std_t_log_cdf(z, nu.value)

# Standard Student-t CDF for a lowering-guaranteed constant nu; mirrors the CPU
# `_std_t_cdf` branch structure exactly.
@inline function _device_std_t_cdf(z::T, nu::T) where {T}
    isinf(z) && return ifelse(z > zero(T), one(T), zero(T))
    z == zero(T) && return T(0.5)
    denominator = nu + z * z
    x = nu / denominator
    y = z * z / denominator # accurate complement of x (see _device_beta_inc_reg_parts)
    regularized = _device_beta_inc_reg_parts(nu / T(2), T(0.5), x, y)
    return ifelse(z > zero(T), one(T) - regularized / T(2), regularized / T(2))
end

# Standard Student-t pdf with the infinite-argument guard used by the CPU
# analytic gradient (an unbounded truncation side contributes zero density).
@inline function _device_std_t_pdf(z::T, nu::T) where {T}
    isinf(z) && return zero(T)
    return exp(_device_studentt_logpdf(nu, zero(T), one(T), z))
end

# d/dz T_cdf(z; nu) = t-pdf(z; nu); nu is a literal (derivative-free), so the nu
# channel is deliberately ignored, matching the CPU analytic gradient's omitted
# (genuinely zero) d/dnu term.
@inline function _device_std_t_cdf(z::DeviceDual{T}, nu::Real) where {T}
    nu_value = convert(T, nu)
    v = z.value
    # pin the derivative to zero on an infinite argument (see the erf duals)
    deriv = ifelse(isinf(v), zero(T), _device_std_t_pdf(v, nu_value) * z.deriv)
    return DeviceDual{T}(_device_std_t_cdf(v, nu_value), deriv)
end
# nu is a lowering-guaranteed literal, so a promoted dual nu carries derivative
# zero; only its value participates (disambiguates against the plain method).
@inline _device_std_t_cdf(z::DeviceDual{T}, nu::DeviceDual{T}) where {T} = _device_std_t_cdf(z, nu.value)

# ---- truncated families -----------------------------------------------------------

# Standard normal pdf with the infinite-bound guard (matches `_std_normal_pdf`).
@inline function _device_std_normal_pdf(z::T) where {T}
    isinf(z) && return zero(T)
    return exp(-z * z / T(2)) * T(0.3989422804014327) # 1/sqrt(2*pi)
end

# log(erfc(x)) with an asymptotic fallback once erfc underflows (Float32 loses
# erfc around x ~ 9.3, far before the CPU Float64 reference):
# erfc(x) ~ exp(-x^2)/(x sqrt(pi)) * (1 - u/2 + 3u^2/4 - 15u^3/8), u = 1/x^2.
# At the smallest engaging x the truncated series is accurate to ~1e-7, well
# inside the Float32 tolerance contract; the derivative matches to the same
# order.
@inline function _device_log_erfc(x::T) where {T}
    e = _device_erfc(x)
    u = one(T) / (x * x)
    # clamp: the series is only SELECTED for large x (u <= ~0.012, argument
    # ~ -0.006), but ifelse evaluates it everywhere and small x would push the
    # log1p argument below -1 (a DomainError on CPU and GPU alike)
    series = log1p(max(-u / T(2) + T(0.75) * u * u - T(1.875) * u * u * u, -T(0.5)))
    # log(abs(x)): the asymptote is only SELECTED for large positive x (erfc
    # underflow), but ifelse evaluates both branches and a GPU log() throws a
    # DomainError on negative arguments instead of returning NaN
    asymptotic = -x * x - log(abs(x)) - T(0.5723649429247001) + series # log(sqrt(pi))
    return ifelse(e > zero(T), log(e), asymptotic)
end

# Scaled complementary error function erfcx(y) = erfc(y) exp(y^2) for y >= 0:
# the Cody tail intervals compute it natively; the central interval scales the
# plain erfc (both factors representable there). The unselected branch is
# domain-safe: exp(y^2) can only overflow where erfc is exactly 0.
@inline _device_erfcx(y::T) where {T} =
    ifelse(y > T(0.46875), _device_erfcx_tail(y), _device_erfc(y) * exp(y * y))

# log(erfc(a) - erfc(b)) for 0 < a < b, cancellation- and underflow-free:
# erfc(a) - erfc(b) = e^{-a^2} erfcx(a) (1 - e^{a^2-b^2} erfcx(b)/erfcx(a)),
# so the tail magnitude lives in the -a^2 log term and the parenthesis is
# -expm1(s) with s < 0. `s` carries ~eps-level absolute error from the erfcx
# ratio, so once |s| sinks under sqrt(eps) the expm1 form has lost half its
# digits -- there the caller-supplied log-space midpoint form takes over (its
# own error is O(s^2), negligible exactly where it is selected).
@inline function _device_log_erfc_diff(a::T, b::T, log_fallback::T) where {T}
    ea = _device_erfcx(a)
    eb = _device_erfcx(b)
    s = (a - b) * (a + b) + log(eb / ea)
    return ifelse(s < -sqrt(eps(T)), -a * a + log(ea) + log(abs(expm1(s))), log_fallback)
end

# log(Phi(zb) - Phi(za)), same branch structure as the CPU
# `_log_normal_cdf_diff` but with each branch computed in log space: one-sided
# tails via log(erfc) (asymptotic past underflow), same-sign two-sided
# intervals via the erfcx difference identity (exact for narrow and far-tail
# intervals where plain erfc differences cancel or underflow), and the
# straddling branch directly (an addition -- it cannot cancel).
@inline function _device_log_normal_cdf_diff(za::T, zb::T) where {T}
    root2 = sqrt(T(2))
    log2 = log(T(2))
    lower_infinite = isinf(za) & (za < zero(T))
    upper_infinite = isinf(zb) & (zb > zero(T))
    if lower_infinite && upper_infinite
        return zero(T)
    elseif lower_infinite
        return _device_log_erfc(-zb / root2) - log2
    elseif upper_infinite
        return _device_log_erfc(za / root2) - log2
    end
    # log-space midpoint form for ulp-narrow same-sign intervals (mid^2 is
    # symmetric under the za <-> -zb mirror, so both tails share it); the +log2
    # aligns it with the erfc-difference scale the branches subtract log2 from
    mid = (za + zb) / T(2)
    log_fallback = -mid * mid / T(2) - T(0.9189385332046727) + log(abs(zb - za)) + log2
    if za > zero(T)
        return _device_log_erfc_diff(za / root2, zb / root2, log_fallback) - log2
    elseif zb < zero(T)
        return _device_log_erfc_diff(-zb / root2, -za / root2, log_fallback) - log2
    end
    return log(_device_erf(zb / root2) - _device_erf(za / root2)) - log2
end

@inline function _device_truncatednormal_logpdf(mu::T, sigma::T, lower::T, upper::T, x::T) where {T}
    base = _device_normal_logpdf(mu, sigma, x)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    # collapsed/cancelled differences fall back to the log-space midpoint
    # approximation inside `_device_log_normal_cdf_diff` itself
    log_z = _device_log_normal_cdf_diff(za, zb)
    # degrade the degenerate lower >= upper (zero-width) case to -Inf; the
    # exception-free device path must never accept it as an infinite density
    in_support = (x >= lower) & (x <= upper) & (lower < upper)
    return ifelse(in_support, base - log_z, _device_neginf(T))
end

# log(T_cdf(zb) - T_cdf(za)) for a constant nu. One-sided normalizers use the
# symmetry S(z) = cdf(-z) so a tail probability is computed DIRECTLY (the
# regularized incomplete beta is small there and keeps relative accuracy) rather
# than as 1 - cdf, which cancels at Float32. Two-sided same-sign intervals use
# the tail-side representation for the same reason, with the midpoint fallback
# guarding what cancellation remains; the straddling branch spans ~0.5 of mass
# on each side, so only a narrow straddle can cancel (also caught by the guard,
# and its midpoint is finite by construction).
@inline function _device_t_log_normalizer(nu::T, za::T, zb::T) where {T}
    lower_infinite = isinf(za) & (za < zero(T))
    upper_infinite = isinf(zb) & (zb > zero(T))
    if lower_infinite && upper_infinite
        return zero(T)
    elseif lower_infinite
        return _device_std_t_log_cdf(zb, nu)
    elseif upper_infinite
        return _device_std_t_log_cdf(-za, nu)
    end
    # both finite: difference of LOG CDFs via expm1 -- log Z = L_big +
    # log(-expm1(L_small - L_big)) -- which stays exact when the plain CDFs
    # underflow (deep light tails) AND when the interval is narrow (the log
    # difference has relative resolution where the plain difference cancels)
    if za > zero(T) # right tail: S(za) - S(zb) = cdf(-za) - cdf(-zb)
        log_big = _device_std_t_log_cdf(-za, nu)
        log_small = _device_std_t_log_cdf(-zb, nu)
    else # left tail and straddling: cdf(zb) - cdf(za)
        log_big = _device_std_t_log_cdf(zb, nu)
        log_small = _device_std_t_log_cdf(za, nu)
    end
    s = log_small - log_big
    # under sqrt(eps) the log-CDF difference is rounding noise; the log-space
    # midpoint form is exact exactly there (its error is O(s^2)), and
    # log(abs(expm1(s))) keeps the unselected branch GPU-safe
    log_fallback = _device_studentt_logpdf(nu, zero(T), one(T), (za + zb) / T(2)) + log(abs(zb - za))
    return ifelse(s < -sqrt(eps(T)), log_big + log(abs(expm1(s))), log_fallback)
end

@inline function _device_truncatedstudentt_logpdf(nu::T, mu::T, sigma::T, lower::T, upper::T, x::T) where {T}
    base = _device_studentt_logpdf(nu, mu, sigma, x)
    za = (lower - mu) / sigma
    zb = (upper - mu) / sigma
    log_z = _device_t_log_normalizer(nu, za, zb)
    # the CPU reference REJECTS lower >= upper (ArgumentError); the exception-free
    # device contract degrades that to -Inf instead of an infinite density
    in_support = (x >= lower) & (x <= upper) & (lower < upper)
    return ifelse(in_support, base - log_z, _device_neginf(T))
end

# ---- simplex (dirichlet) ----------------------------------------------------------

# Compile-time tuple folds (dual-safe: comparisons select on the value channel,
# arithmetic promotes per element).
@inline _device_tuple_max(t::Tuple{A}) where {A} = t[1]
@inline _device_tuple_max(t::Tuple) = max(first(t), _device_tuple_max(Base.tail(t)))
@inline _device_tuple_sum(t::Tuple{A}) where {A} = t[1]
@inline _device_tuple_sum(t::Tuple) = first(t) + _device_tuple_sum(Base.tail(t))
@inline _device_tuple_log_sum(t::Tuple{A}) where {A} = log(t[1])
@inline _device_tuple_log_sum(t::Tuple) = log(first(t)) + _device_tuple_log_sum(Base.tail(t))

# Shifted-softmax simplex constrain, mirroring the CPU `_to_constrained_simplex!`
# exactly: K-1 unconstrained logits plus an implicit K-th logit of 0, max-shifted
# for stability. Returns the K-tuple and the log-abs-det `sum(log.(p))` over all
# K constrained entries (`_simplex_logabsdet`).
@inline function _device_simplex_constrain(z::NTuple{Km1,T}) where {Km1,T}
    shift = max(zero(T), _device_tuple_max(z))
    exponentials = ntuple(i -> exp(z[i] - shift), Val(Km1))
    implicit_last = exp(-shift)
    total = implicit_last + _device_tuple_sum(exponentials)
    constrained = (ntuple(i -> exponentials[i] / total, Val(Km1))..., implicit_last / total)
    return (constrained, _device_tuple_log_sum(constrained))
end

# Dirichlet logpdf over compile-time tuples; per-component promotion keeps
# heterogeneous (literal/dual) alpha tuples safe. The support check mirrors the
# CPU `_backend_dirichlet_logpdf` acceptance (all components positive, sum
# within sqrt(eps)*K*16 of one); a constrained latent satisfies it by
# construction, observations may not.
# log(abs(v)): a negative observed component never SELECTS this branch (the
# in_support ifelse rejects it), but the fold is evaluated eagerly and log()
# throws a DomainError on negative arguments (CPU KernelAbstractions backend;
# GPU too) instead of the reference's -Inf.
@inline _device_dirichlet_fold(alpha::Tuple{A}, value::Tuple{B}) where {A,B} =
    (alpha[1] - 1) * log(abs(value[1]))
@inline _device_dirichlet_fold(alpha::Tuple, value::Tuple) =
    (first(alpha) - 1) * log(abs(first(value))) + _device_dirichlet_fold(Base.tail(alpha), Base.tail(value))
@inline _device_dirichlet_loggamma_sum(t::Tuple{A}) where {A} = _device_loggamma(t[1])
@inline _device_dirichlet_loggamma_sum(t::Tuple) =
    _device_loggamma(first(t)) + _device_dirichlet_loggamma_sum(Base.tail(t))
@inline _device_tuple_all_positive(t::Tuple{A}) where {A} = t[1] > zero(t[1])
@inline _device_tuple_all_positive(t::Tuple) = (first(t) > zero(first(t))) & _device_tuple_all_positive(Base.tail(t))

@inline function _device_dirichlet_logpdf(alpha::Tuple, value::Tuple)
    total_alpha = _device_tuple_sum(alpha)
    accumulator =
        _device_loggamma(total_alpha) - _device_dirichlet_loggamma_sum(alpha) +
        _device_dirichlet_fold(alpha, value)
    total_value = _device_tuple_sum(value)
    tolerance = sqrt(eps(typeof(total_value))) * length(value) * 16
    in_support = _device_tuple_all_positive(value) & (abs(total_value - one(total_value)) <= tolerance)
    return ifelse(in_support, accumulator, oftype(accumulator, -Inf))
end

# ---- finite mixture of normals -----------------------------------------------

# Per-component log(w) + normal logpdf. log(abs(w)) keeps the eagerly-evaluated
# branch GPU-safe; a nonpositive weight degrades its component to -Inf (the CPU
# reference throws on negative weights, scores log(0) = -Inf on zero).
@inline function _device_mixture_term(w, mu, sigma, x)
    ww, m, s, v = promote(w, mu, sigma, x)
    term = log(abs(ww)) + _device_normal_logpdf(m, s, v)
    return ifelse(ww > zero(ww), term, oftype(term, -Inf))
end

@inline _device_mixture_terms(weights::Tuple{A}, mus::Tuple{B}, sigmas::Tuple{C}, x) where {A,B,C} =
    (_device_mixture_term(weights[1], mus[1], sigmas[1], x),)
@inline _device_mixture_terms(weights::Tuple, mus::Tuple, sigmas::Tuple, x) = (
    _device_mixture_term(first(weights), first(mus), first(sigmas), x),
    _device_mixture_terms(Base.tail(weights), Base.tail(mus), Base.tail(sigmas), x)...,
)

@inline _device_exp_shift_sum(t::Tuple{A}, shift) where {A} = exp(t[1] - shift)
@inline _device_exp_shift_sum(t::Tuple, shift) = exp(first(t) - shift) + _device_exp_shift_sum(Base.tail(t), shift)

@inline _device_tuple_all_nonneg(t::Tuple{A}) where {A} = t[1] >= zero(t[1])
@inline _device_tuple_all_nonneg(t::Tuple) =
    (first(t) >= zero(first(t))) & _device_tuple_all_nonneg(Base.tail(t))

# Max-shifted log-sum-exp over compile-time component tuples, mirroring the CPU
# `_backend_mixture_normal_logpdf` (all components -Inf -> -Inf). The weight
# validation the CPU scorer enforces by throwing (nonnegative, summing to 1
# within 1e-8, issue #76) degrades to -Inf under the exception-free device
# contract.
@inline function _device_mixture_normal_logpdf(weights::Tuple, mus::Tuple, sigmas::Tuple, x)
    terms = _device_mixture_terms(weights, mus, sigmas, x)
    shift = _device_tuple_max(terms)
    result = shift + log(_device_exp_shift_sum(terms, shift))
    total = _device_tuple_sum(weights)
    weights_ok =
        _device_tuple_all_nonneg(weights) & (abs(total - one(total)) <= oftype(total, 1e-8))
    return ifelse(weights_ok & isfinite(shift), result, oftype(result, -Inf))
end

# ---- marginalize=:enumerate fold (issue #67) ----------------------------------
#
# Per-branch log pmf of the enumerated family. bernoulli takes its single
# probability and the branch value (0/1); categorical picks log(probabilities[k]).
# Both reuse the exception-free device logpdfs (out-of-support / invalid vectors
# degrade to -Inf), so a zero-mass branch value naturally scores -Inf.
@inline _device_marg_logpmf(::Val{:bernoulli}, probabilities::Tuple, value) =
    _device_bernoulli_logpdf(promote(first(probabilities), value)...)
@inline _device_marg_logpmf(::Val{:categorical}, probabilities::Tuple, value) =
    _device_categorical_logpdf(probabilities, value)

# A -Inf with a ZERO derivative, in the working type (plain `T` in the logjoint
# kernel, `DeviceDual{T}` in the gradient kernel). Used to exclude a branch by its
# FULL term (issue #62 lesson) BEFORE its degenerate gradient buffer can reach the
# fold: a raw -Inf branch dual can carry an Inf/NaN derivative, and exp() of it
# would produce 0 * NaN = NaN; substituting a clean (-Inf, 0) dual makes the
# excluded branch contribute exactly zero to both the value and the derivative.
@inline _device_marg_neginf(::Type{T}) where {T<:Real} = T(-Inf)
@inline _device_marg_neginf(::Type{DeviceDual{T}}) where {T} = DeviceDual{T}(T(-Inf), zero(T))

# Exclude a branch whose FULL term (log pmf + suffix) is non-finite (zero-mass or
# an impossible/unevaluable suffix): its value channel already contributes ~0 via
# exp(-Inf), and this scrubs any Inf/NaN derivative it might carry.
@inline _device_marg_sanitize(t) =
    ifelse(isfinite(_device_dual_value(t)), t, _device_marg_neginf(typeof(t)))
@inline _device_marg_sanitize_all(::Tuple{}) = ()
@inline _device_marg_sanitize_all(t::Tuple) =
    (_device_marg_sanitize(first(t)), _device_marg_sanitize_all(Base.tail(t))...)

# Compile-time unrolled masked pick of branch `selector` (1-based) for the
# CONDITIONED single-branch case; unmatched branches contribute zero, so duals
# promote safely and only the matched branch's term (and its derivative) survives.
# The selector is a plain float (the staged selector row is a small exact integer,
# exact even in Float32), so the branch index is compared in float space -- no
# in-kernel integer conversion (exception-free device contract).
@inline _device_marg_select(::Tuple{}, ::Int, selector, acc) = acc
@inline function _device_marg_select(totals::Tuple, index::Int, selector, acc)
    t = first(totals)
    chosen = ifelse(oftype(selector, index) == selector, t, zero(t))
    return _device_marg_select(Base.tail(totals), index + 1, selector, acc + chosen)
end

# Combine the per-branch full terms `t_v = log pmf(v) + suffix(v)` into the step's
# contribution, given the staged branch `selector` (0 == marginalize; `v` in
# `1..S` == condition on branch `v`; anything else == -Inf). `selector` is read
# from the float observation buffer (never the count-only integer mirror, which is
# dropped to a dummy when no count family reads it). Marginalization is a
# max-shifted log-sum-exp over the sanitized terms (the mixture-normal template);
# the dual arithmetic through `shift + log(sum exp(t - shift))` yields exactly the
# responsibility-weighted CPU gradient `sum_v r_v * d t_v`. `ifelse` (not control
# flow) keeps the kernel divergence-free.
@inline function _device_marg_combine(totals::Tuple, selector)
    sanitized = _device_marg_sanitize_all(totals)
    shift = _device_tuple_max(sanitized)
    marginal = shift + log(_device_exp_shift_sum(sanitized, shift))
    marginal = ifelse(isfinite(_device_dual_value(shift)), marginal, _device_marg_neginf(typeof(shift)))
    chosen = _device_marg_select(totals, 1, selector, zero(first(totals)))
    in_range = (selector >= oftype(selector, 1)) & (selector <= oftype(selector, length(totals)))
    conditioned = ifelse(in_range, chosen, _device_marg_neginf(typeof(first(totals))))
    return ifelse(selector == zero(selector), marginal, conditioned)
end

# ---- dense multivariate normal ------------------------------------------------

# Forward substitution solving L z = x - mu over a COLUMN-MAJOR PACKED lower
# triangle (the CPU `_packed_lower_index` order), fully unrolled at compile
# time from the dimension in the type. Returns (log_det, quadratic, diag_ok);
# `log(abs(...))` keeps the unselected branch GPU-safe and `diag_ok` lets the
# caller degrade a non-positive diagonal to -Inf (the CPU reference would
# throw a DomainError there; the device contract is exception-free).
@generated function _device_mvnormaldense_solve(scale_packed::NTuple{P,TL}, mu::Tuple, x::NTuple{D,TX}) where {P,TL,D,TX}
    packed_index(row, col) = (col - 1) * D - ((col - 1) * (col - 2)) ÷ 2 + (row - col + 1)
    z_symbols = [Symbol(:z_, i) for i = 1:D]
    body = Expr[]
    push!(body, :(diagonal = scale_packed[$(packed_index(1, 1))]))
    push!(body, :(diag_ok = diagonal > zero(diagonal)))
    push!(body, :(log_det = log(abs(diagonal))))
    push!(body, :($(z_symbols[1]) = (x[1] - mu[1]) / diagonal))
    push!(body, :(quadratic = $(z_symbols[1]) * $(z_symbols[1])))
    for row = 2:D
        residual = :(x[$row] - mu[$row])
        for col = 1:(row-1)
            residual = :($residual - scale_packed[$(packed_index(row, col))] * $(z_symbols[col]))
        end
        push!(body, :(diagonal = scale_packed[$(packed_index(row, row))]))
        push!(body, :(diag_ok &= diagonal > zero(diagonal)))
        push!(body, :($(z_symbols[row]) = ($residual) / diagonal))
        push!(body, :(log_det += log(abs(diagonal))))
        push!(body, :(quadratic += $(z_symbols[row]) * $(z_symbols[row])))
    end
    return quote
        $(body...)
        (log_det, quadratic, diag_ok)
    end
end

@inline function _device_mvnormaldense_logpdf(scale_packed::NTuple{P,TL}, mu::Tuple, x::NTuple{D,TX}) where {P,TL,D,TX}
    log_det, quadratic, diag_ok = _device_mvnormaldense_solve(scale_packed, mu, x)
    base = -log_det - quadratic / 2 - D * TX(0.9189385332046727) # 0.5*log(2*pi) per dimension
    return ifelse(diag_ok, base, oftype(base, -Inf))
end

# ---- multivariate Student-t ---------------------------------------------------

# Shared multivariate-t log density from the Cholesky log-determinant
# `log_det = sum_i log(scale diagonal)` and the Mahalanobis quadratic `q`,
# mirroring the CPU `_mvstudentt_log_density`. `d` is the compile-time dimension.
@inline function _device_mvstudentt_density(nu, d::Integer, log_det, quadratic)
    half = (nu + d) / 2
    cst = _device_loggamma(half) - _device_loggamma(nu / 2)
    return cst - (d / 2) * log(nu * oftype(nu, pi)) - log_det - half * log1p(quadratic / nu)
end

# Compile-time-unrolled reduction of a diagonal-scale multivariate-t into
# (log_det, quadratic); per-component `promote` keeps heterogeneous
# (literal/dual) argument tuples safe, mirroring `_device_mvnormal_logpdf_fold`.
@inline function _device_mvt_diag_reduce(mu::Tuple{A}, sigma::Tuple{B}, value::Tuple{C}) where {A,B,C}
    m, s, v = promote(mu[1], sigma[1], value[1])
    z = (v - m) / s
    return (log(s), z * z)
end
@inline function _device_mvt_diag_reduce(mu::Tuple, sigma::Tuple, value::Tuple)
    m, s, v = promote(first(mu), first(sigma), first(value))
    z = (v - m) / s
    ld, q = _device_mvt_diag_reduce(Base.tail(mu), Base.tail(sigma), Base.tail(value))
    return (log(s) + ld, z * z + q)
end

@inline function _device_mvstudentt_diag_logpdf(nu, mu::Tuple, sigma::Tuple, value::Tuple)
    log_det, quadratic = _device_mvt_diag_reduce(mu, sigma, value)
    return _device_mvstudentt_density(nu, length(mu), log_det, quadratic)
end

# Dense-scale multivariate-t: reuse the mvnormaldense forward substitution for
# (log_det, quadratic, diag_ok), then wrap the quadratic in the Student-t tail.
@inline function _device_mvstudentt_dense_logpdf(
    nu,
    scale_packed::NTuple{P,TL},
    mu::Tuple,
    x::NTuple{D,TX},
) where {P,TL,D,TX}
    log_det, quadratic, diag_ok = _device_mvnormaldense_solve(scale_packed, mu, x)
    base = _device_mvstudentt_density(nu, D, log_det, quadratic)
    return ifelse(diag_ok, base, oftype(base, -Inf))
end

# ---- lkj cholesky correlation ---------------------------------------------------

# Numerically stable log(1 - tanh(z)^2), mirroring the CPU `_log1m_tanh_sq`
# exactly: 2 * (log(2) - |z| - log1p(exp(-2|z|))), symmetric so exp never
# overflows. 0.6931471805599453 = log(2).
@inline function _device_log1m_tanh_sq(z::T) where {T}
    magnitude = abs(z)
    return 2 * (T(0.6931471805599453) - magnitude - log1p(exp(-2 * magnitude)))
end

# Stan's cholesky_corr_constrain over compile-time tuples, mirroring the CPU
# `_to_constrained_cholesky_corr!` (row-major below-diagonal consumption of the
# d(d-1)/2 unconstrained entries, column-major packed d(d+1)/2 output), fully
# unrolled from the dimension in the type like `_device_mvnormaldense_solve`.
# Returns (packed_tuple, logabsdet).
#
# The row's leftover variance is tracked MULTIPLICATIVELY as
#   remaining = prod_{k<j} (1 - tanh(z_k)^2)   ( == 1 - sum_{k<j} L[row,k]^2 ),
# rather than by the subtractive `1 - sum_sqs` (issue #104). Both give the same
# value, but the multiplicative form makes the forward-mode DERIVATIVE
# well-conditioned near tanh saturation: `d/dz_j log(remaining) = -2 tanh(z_j)`
# is bounded, and the imprecise `(1 - w^2)` factor cancels exactly between the
# value and derivative channels of `log(remaining)` / `sqrt(remaining)`, so the
# device gradient stays within FP noise of the CPU analytic gradient on
# in-support states instead of drifting as `sum_sqs -> 1`. Saturation is still
# exact: when tanh rounds to +-1, `1 - w^2` is exactly 0, so `remaining` hits 0
# and the ifelse guards select the CONSTANT -Inf / zero branches (the eagerly
# evaluated `sqrt(0)` / `log(0)` duals carry 0/0 derivative NaNs that would
# otherwise poison downstream gradients; ifelse discards the unselected branch).
@generated function _device_cholesky_corr_constrain(z::NTuple{Q,T}, ::Val{D}) where {Q,T,D}
    packed_index(row, col) = (col - 1) * D - ((col - 1) * (col - 2)) ÷ 2 + (row - col + 1)
    entry_symbols = Vector{Symbol}(undef, (D * (D + 1)) ÷ 2)
    entry_symbols[packed_index(1, 1)] = :unit
    body = Expr[]
    push!(body, :(unit = one(z[1])))
    push!(body, :(lad = zero(z[1])))
    z_position = 0
    for row = 2:D
        rem_sym = Symbol(:remaining_, row)
        sum_sym = Symbol(:sum_sqs_, row)
        push!(body, :($rem_sym = one(z[1]))) # leftover variance before column 1 is 1
        push!(body, :($sum_sym = zero(z[1]))) # additive sum tracked ONLY for the saturation trigger
        for col = 1:(row-1)
            z_position += 1
            entry = Symbol(:entry_, row, :_, col)
            entry_symbols[packed_index(row, col)] = entry
            wsym = Symbol(:w_, row, :_, col)
            saturated = Symbol(:saturated_, row, :_, col)
            push!(body, :($wsym = tanh(z[$z_position])))
            # Saturation trigger matches the CPU reference (`sum_sqs >= 1`) so the
            # device support boundary is unchanged; the values/derivatives below
            # come from the well-conditioned multiplicative `remaining` instead.
            push!(body, :($saturated = $sum_sym >= one(z[1])))
            # log(remaining) has the clean, bounded derivative -2*tanh(z_k) per
            # earlier column; guard the eager log/sqrt arguments at zero and let
            # the ifelse select the constant -Inf/zero on the saturated branch.
            push!(
                body,
                :(
                    lad +=
                        _device_log1m_tanh_sq(z[$z_position]) + ifelse(
                            $saturated,
                            oftype($rem_sym, -Inf),
                            log(max($rem_sym, zero($rem_sym))) / 2,
                        )
                ),
            )
            push!(
                body,
                :(
                    $entry =
                        $wsym * ifelse($saturated, zero($rem_sym), sqrt(max($rem_sym, zero($rem_sym))))
                ),
            )
            push!(body, :($sum_sym += $entry * $entry))
            # remaining *= (1 - w^2); exactly 0 once tanh saturates to +-1
            push!(body, :($rem_sym = ifelse($saturated, zero($rem_sym), $rem_sym * (one($wsym) - $wsym * $wsym))))
        end
        diagonal = Symbol(:entry_, row, :_, row)
        entry_symbols[packed_index(row, row)] = diagonal
        row_saturated = Symbol(:saturated_, row, :_, row)
        push!(body, :($row_saturated = $sum_sym >= one(z[1])))
        push!(body, :($diagonal = ifelse($row_saturated, zero($rem_sym), sqrt(max($rem_sym, zero($rem_sym))))))
    end
    return quote
        $(body...)
        (($(entry_symbols...),), lad)
    end
end

# Log of the LKJ normalizing constant, mirroring the CPU
# `_lkj_log_normalizing_constant` with the compile-time dimension unrolling the
# cvine level loop; eta may be a dual (latent concentration), so every constant
# is formed in eta's own type. 0.6931471805599453 = log(2).
@inline function _device_lkj_log_normalizer(eta::T, ::Val{D}) where {T,D}
    log_c = zero(eta)
    for k = 1:(D-1)
        a = eta + (D - 1 - k) * T(0.5)
        log_beta = _device_loggamma(a) + _device_loggamma(a) - _device_loggamma(2 * a)
        log_c += (D - k) * ((2 * eta - 2 + (D - k)) * T(0.6931471805599453) + log_beta)
    end
    return -log_c
end

# LKJ log density over a packed correlation Cholesky factor, mirroring the CPU
# `logpdf(::LKJCholeskyDist, x)`: sum_{row=2..d} (d - row + 2*eta - 2) *
# log(L[row,row]) plus the normalizer, with the same support acceptance
# (positive diagonals, row square sums within sqrt(eps)*d*16 of one -- unit
# rows, issue #78; the tolerance scales with the value precision, matching the
# CPU f32 support discipline from issue #49). log(abs(...)) keeps the
# eagerly-evaluated branch
# GPU-safe; out-of-support values select the -Inf branch instead of throwing.
@generated function _device_lkjcholesky_logpdf(eta::TE, packed::NTuple{P,T}, ::Val{D}) where {TE,T,P,D}
    packed_index(row, col) = (col - 1) * D - ((col - 1) * (col - 2)) ÷ 2 + (row - col + 1)
    body = Expr[]
    push!(body, :(accumulator = _device_lkj_log_normalizer(eta, Val(D)) + zero(first(packed))))
    push!(body, :(tolerance = sqrt(eps(typeof(first(packed)))) * $(D * 16)))
    push!(body, :(in_support = true))
    for row = 1:D
        diagonal = :(packed[$(packed_index(row, row))])
        push!(body, :(in_support &= $diagonal > zero($diagonal)))
        sum_expr = :($diagonal * $diagonal)
        for col = 1:(row-1)
            sum_expr = :($sum_expr + packed[$(packed_index(row, col))] * packed[$(packed_index(row, col))])
        end
        push!(body, :(in_support &= abs(($sum_expr) - one(first(packed))) <= tolerance))
        row >= 2 && push!(body, :(accumulator += ($(D - row) + 2 * eta - 2) * log(abs($diagonal))))
    end
    return quote
        $(body...)
        ifelse(in_support, accumulator, oftype(accumulator, -Inf))
    end
end

# Parameter transform codes shared by lowering, kernel, and staging.
const DEVICE_TRANSFORM_IDENTITY = Int32(0)
const DEVICE_TRANSFORM_LOG = Int32(1)
const DEVICE_TRANSFORM_LOGIT = Int32(2)

# Applies an unconstrained -> constrained transform, returning
# (constrained_value, logabsdet_jacobian). Mirrors src/parameters.jl and the
# batched transform branch in src/batched/workspace.jl for the supported set.
@inline function _device_transform(tcode::Int32, u::T) where {T}
    if tcode == DEVICE_TRANSFORM_LOG
        return (exp(u), u)
    elseif tcode == DEVICE_TRANSFORM_LOGIT
        c = inv(one(T) + exp(-u))
        # Exact log-Jacobian -|u| - 2*log1p(exp(-|u|)) mirroring the CPU
        # `_logit_logabsdetjac`: finite for every u, where log(c) + log1p(-c)
        # collapses to -Inf once c rounds to 1 (u ~ 16.6 in Float32). abs,
        # exp, and log1p all carry the DeviceDual derivative channel, so the
        # gradient (-tanh(u/2)) stays finite as well.
        magnitude = abs(u)
        return (c, -magnitude - 2 * log1p(exp(-magnitude)))
    else # DEVICE_TRANSFORM_IDENTITY
        return (u, zero(T))
    end
end
