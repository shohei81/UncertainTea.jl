# Wide (P-partial) forward-mode dual for the persistent NUTS in-kernel gradient
# (issue #154, increment 4).
#
# WHY THIS EXISTS. The shipped persistent kernel (`persistent_nuts.jl`, increment 2)
# computes the leapfrog gradient by walking the lowered device plan ONCE PER PARAMETER
# with the scalar `DeviceDual{T}` (`dual.jl`): P serial plan walks, each O(N) over the
# observations, all on the one lane that owns the chain. On a heavy-per-gradient model
# (`logistic_large`: P=16 coefficients, N=8000 observations) that is O(P*N) serial work
# per leaf -- the increment-4 bottleneck. Crucially, EACH of the P scalar walks
# recomputes the FULL logjoint value -- including every transcendental in every logpdf
# (the `log1pexp` in a `bernoullilogit` GLM term, etc.) -- just to read off one partial
# derivative. So a heavy GLM evaluates N*(P+1) transcendentals per gradient where only N
# are mathematically distinct.
#
# WHAT THIS IS. `DeviceGradN{N,T}` carries a value AND all `N == P` directional
# derivatives at once (`partials::NTuple{N,T}`), so a SINGLE forward-mode plan walk
# yields the whole gradient. Each transcendental value is then computed ONCE and its
# derivative is fanned out across the N partials by cheap multiply-adds. For the heavy
# GLM this collapses the N*(P+1) transcendentals to N, which is the increment-4 win; the
# non-transcendental partial propagation stays O(N) per op, i.e. the same O(P*N) FLOP
# class as the scalar path, so the wide walk is never slower and is strictly faster
# whenever a logpdf spends real work in a transcendental. It is `isbits`
# (a `T` + an `NTuple{N,T}`), allocation-free and exception-free, so the identical code
# runs inside a KernelAbstractions kernel on CPU() (Float64) and on Metal (Float32),
# exactly like `DeviceDual`.
#
# WHY NOT threadgroup-per-chain (the #154 sketch's headline). See the increment-4
# section of docs/persistent-nuts.md: KA's `@synchronize`/`@localmem` are only usable
# in the `@kernel` body (empirically: they cannot be captured inside a called `@inline`
# device function -- "used outside kernel or not captured"), so a threadgroup variant
# would have to inline the whole ~250-line data-dependent tree with barriers threaded
# through its doubling/leaf loops, an untested Metal deadlock risk with no precedent in
# this package. This wide-dual walk is the #154 issue's sanctioned P-partial fallback:
# it stays LANE-PER-CHAIN (no barriers, no shared memory, the RNG stream is byte-for-
# byte the scalar kernel's) and compiles on Metal unchanged, while still delivering the
# heavy-model speedup. The threadgroup tree remains future work.
#
# The op set mirrors `DeviceDual` exactly (the device logjoint / transform exercise the
# same functions), so any density that differentiates through `DeviceDual` differentiates
# through `DeviceGradN` unchanged; only the derivative channel widens from 1 to N.

struct DeviceGradN{N,T<:Real} <: Number
    value::T
    partials::NTuple{N,T}
end

# A bare real seeds a CONSTANT (all-zero partials).
@inline DeviceGradN{N,T}(value::Real) where {N,T<:Real} =
    DeviceGradN{N,T}(convert(T, value), ntuple(_ -> zero(T), Val(N)))

@inline _device_dual_basetype(::Type{DeviceGradN{N,T}}) where {N,T} = T

# ---- partials helpers (fully unrolled NTuple maps) -----------------------------
@inline _gn_scale(p::NTuple{N,T}, s::T) where {N,T} = ntuple(i -> @inbounds(p[i]) * s, Val(N))
@inline _gn_add(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} = ntuple(i -> @inbounds(a[i] + b[i]), Val(N))
@inline _gn_sub(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} = ntuple(i -> @inbounds(a[i] - b[i]), Val(N))
@inline _gn_neg(a::NTuple{N,T}) where {N,T} = ntuple(i -> @inbounds(-a[i]), Val(N))

# ---- promotion / conversion ----------------------------------------------------
Base.promote_rule(::Type{DeviceGradN{N,T}}, ::Type{S}) where {N,T<:Real,S<:Real} =
    DeviceGradN{N,promote_type(T, S)}
Base.promote_rule(::Type{DeviceGradN{N,T}}, ::Type{DeviceGradN{N,S}}) where {N,T<:Real,S<:Real} =
    DeviceGradN{N,promote_type(T, S)}

Base.convert(::Type{DeviceGradN{N,T}}, x::Real) where {N,T<:Real} = DeviceGradN{N,T}(convert(T, x))
@inline Base.convert(::Type{DeviceGradN{N,T}}, x::DeviceGradN{N,S}) where {N,T<:Real,S} =
    DeviceGradN{N,T}(convert(T, x.value), ntuple(i -> convert(T, @inbounds x.partials[i]), Val(N)))
Base.convert(::Type{DeviceGradN{N,T}}, x::DeviceGradN{N,T}) where {N,T<:Real} = x

Base.eps(::Type{DeviceGradN{N,T}}) where {N,T<:Real} = DeviceGradN{N,T}(eps(T))
Base.zero(::Type{DeviceGradN{N,T}}) where {N,T<:Real} = DeviceGradN{N,T}(zero(T))
Base.one(::Type{DeviceGradN{N,T}}) where {N,T<:Real} = DeviceGradN{N,T}(one(T))
Base.zero(x::DeviceGradN{N,T}) where {N,T<:Real} = zero(DeviceGradN{N,T})
Base.one(x::DeviceGradN{N,T}) where {N,T<:Real} = one(DeviceGradN{N,T})
Base.float(x::DeviceGradN) = x

# ---- arithmetic ----------------------------------------------------------------
@inline Base.:+(a::DeviceGradN{N,T}, b::DeviceGradN{N,T}) where {N,T} =
    DeviceGradN{N,T}(a.value + b.value, _gn_add(a.partials, b.partials))
@inline Base.:-(a::DeviceGradN{N,T}, b::DeviceGradN{N,T}) where {N,T} =
    DeviceGradN{N,T}(a.value - b.value, _gn_sub(a.partials, b.partials))
@inline Base.:-(a::DeviceGradN{N,T}) where {N,T} = DeviceGradN{N,T}(-a.value, _gn_neg(a.partials))
@inline Base.:+(a::DeviceGradN{N,T}) where {N,T} = a
@inline Base.:*(a::DeviceGradN{N,T}, b::DeviceGradN{N,T}) where {N,T} = DeviceGradN{N,T}(
    a.value * b.value,
    ntuple(i -> @inbounds(a.partials[i] * b.value + a.value * b.partials[i]), Val(N)),
)
@inline function Base.:/(a::DeviceGradN{N,T}, b::DeviceGradN{N,T}) where {N,T}
    inv_b = one(T) / b.value
    inv_b2 = inv_b * inv_b
    return DeviceGradN{N,T}(
        a.value * inv_b,
        ntuple(i -> @inbounds((a.partials[i] * b.value - a.value * b.partials[i]) * inv_b2), Val(N)),
    )
end
@inline Base.inv(a::DeviceGradN{N,T}) where {N,T} = one(DeviceGradN{N,T}) / a

# ---- powers --------------------------------------------------------------------
@inline function Base.:^(a::DeviceGradN{N,T}, n::Integer) where {N,T}
    v = a.value^n
    coeff = T(n) * a.value^(n - 1)
    return DeviceGradN{N,T}(v, _gn_scale(a.partials, coeff))
end
@inline function Base.:^(a::DeviceGradN{N,T}, p::Real) where {N,T}
    v = a.value^p
    coeff = T(p) * a.value^(p - one(T))
    return DeviceGradN{N,T}(v, _gn_scale(a.partials, coeff))
end
@inline function Base.:^(a::DeviceGradN{N,T}, b::DeviceGradN{N,T}) where {N,T}
    v = a.value^b.value
    # d(a^b) = a^b (b' log a + b a'/a)
    la = log(a.value)
    inv_a = one(T) / a.value
    return DeviceGradN{N,T}(
        v,
        ntuple(i -> @inbounds(v * (b.partials[i] * la + b.value * a.partials[i] * inv_a)), Val(N)),
    )
end

# ---- elementary functions ------------------------------------------------------
@inline function Base.exp(a::DeviceGradN{N,T}) where {N,T}
    e = exp(a.value)
    return DeviceGradN{N,T}(e, _gn_scale(a.partials, e))
end
@inline Base.log(a::DeviceGradN{N,T}) where {N,T} =
    DeviceGradN{N,T}(log(a.value), _gn_scale(a.partials, one(T) / a.value))
@inline Base.log1p(a::DeviceGradN{N,T}) where {N,T} =
    DeviceGradN{N,T}(log1p(a.value), _gn_scale(a.partials, one(T) / (one(T) + a.value)))
@inline function Base.expm1(a::DeviceGradN{N,T}) where {N,T}
    return DeviceGradN{N,T}(expm1(a.value), _gn_scale(a.partials, exp(a.value)))
end
@inline function Base.sqrt(a::DeviceGradN{N,T}) where {N,T}
    s = sqrt(a.value)
    return DeviceGradN{N,T}(s, _gn_scale(a.partials, one(T) / (T(2) * s)))
end
@inline function Base.abs(a::DeviceGradN{N,T}) where {N,T}
    s = ifelse(a.value > zero(T), one(T), ifelse(a.value < zero(T), -one(T), zero(T)))
    return DeviceGradN{N,T}(abs(a.value), _gn_scale(a.partials, s))
end
# round is locally constant -> zero derivative channel (matches DeviceDual).
@inline Base.round(a::DeviceGradN{N,T}) where {N,T} =
    DeviceGradN{N,T}(round(a.value), ntuple(_ -> zero(T), Val(N)))
@inline Base.sin(a::DeviceGradN{N,T}) where {N,T} =
    DeviceGradN{N,T}(sin(a.value), _gn_scale(a.partials, cos(a.value)))
@inline Base.cos(a::DeviceGradN{N,T}) where {N,T} =
    DeviceGradN{N,T}(cos(a.value), _gn_scale(a.partials, -sin(a.value)))
@inline function Base.tanh(a::DeviceGradN{N,T}) where {N,T}
    t = tanh(a.value)
    return DeviceGradN{N,T}(t, _gn_scale(a.partials, one(T) - t * t))
end

# ---- min / max / clamp (value-selected branches, whole dual chosen) -------------
@inline Base.min(a::DeviceGradN{N,T}, b::DeviceGradN{N,T}) where {N,T} = ifelse(a.value <= b.value, a, b)
@inline Base.max(a::DeviceGradN{N,T}, b::DeviceGradN{N,T}) where {N,T} = ifelse(a.value >= b.value, a, b)
@inline function Base.clamp(x::DeviceGradN{N,T}, lo::DeviceGradN{N,T}, hi::DeviceGradN{N,T}) where {N,T}
    return ifelse(x.value < lo.value, lo, ifelse(x.value > hi.value, hi, x))
end

# ---- comparisons (on the value channel) ----------------------------------------
@inline Base.:(==)(a::DeviceGradN, b::DeviceGradN) = a.value == b.value
@inline Base.:<(a::DeviceGradN, b::DeviceGradN) = a.value < b.value
@inline Base.:<=(a::DeviceGradN, b::DeviceGradN) = a.value <= b.value
@inline Base.isless(a::DeviceGradN, b::DeviceGradN) = isless(a.value, b.value)
@inline Base.isnan(a::DeviceGradN) = isnan(a.value)
@inline Base.isfinite(a::DeviceGradN) = isfinite(a.value)
@inline Base.isinf(a::DeviceGradN) = isinf(a.value)

# Value/partials accessors (also accept a bare real, as `DeviceDual`'s do).
@inline _device_dual_value(x::DeviceGradN) = x.value
@inline _device_gn_partials(x::DeviceGradN) = x.partials

# ---- seed primitives shared by BOTH dual widths --------------------------------
# The device gradient step family (`gradient_kernel.jl`) builds its differentiation
# duals through these two primitives ONLY, so the SAME step code serves the scalar
# `DeviceDual` walk (one partial: derivative 1 iff the parameter row is the walk's
# target `pidx`) and the wide `DeviceGradN` walk (an N-partial one-hot at the parameter
# row, independent of `pidx`). `_seed_latent` seeds a latent parameter value at absolute
# unconstrained row `row`; `_seed_obs` seeds an observed / constant value (no derivative).
# Scalar seeding is byte-for-byte the pre-refactor expression, so the masked / grid
# gradient paths are unchanged.
@inline _seed_latent(::Type{DeviceDual{T}}, raw::Real, row::Int32, pidx) where {T} =
    DeviceDual{T}(convert(T, raw), ifelse(row == Int32(pidx), one(T), zero(T)))
@inline _seed_obs(::Type{DeviceDual{T}}, value::Real) where {T} = DeviceDual{T}(convert(T, value), zero(T))

@inline function _seed_latent(::Type{DeviceGradN{N,T}}, raw::Real, row::Int32, pidx) where {N,T}
    partials = ntuple(i -> ifelse(Int32(i) == row, one(T), zero(T)), Val(N))
    return DeviceGradN{N,T}(convert(T, raw), partials)
end
@inline _seed_obs(::Type{DeviceGradN{N,T}}, value::Real) where {N,T} = DeviceGradN{N,T}(convert(T, value))
