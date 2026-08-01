# Minimal scalar forward-mode dual number for the device gradient kernel.
#
# `DeviceDual{T}(value, deriv)` carries a value and a single directional derivative.
# It is `isbits` (two `T` fields), allocation-free, and exception-free, so it runs
# inside a KernelAbstractions kernel on the CPU reference backend and on a GPU
# (e.g. Metal at Float32) unchanged. We do NOT use ForwardDiff on device: ForwardDiff
# `Dual`s carry a partials `Tuple` / `Partials` wrapper whose layout and tag plumbing
# is heavier than the one-partial scalar case we need here.
#
# Only the operations exercised by the device logjoint (see `math.jl` and
# `_device_transform`) are implemented: +, -, *, /, ^ (integer & real powers),
# exp, log, log1p, sqrt, abs, min, max, clamp, sin, cos, tanh, round, and value-based comparisons.
# `DeviceDual` is a `Number`, so mixed arithmetic with a plain real scalar is resolved
# by Julia's promotion fallback (`+(::Number, ::Number) = +(promote(...)...)`) once we
# register `promote_rule` / `convert` below.

struct DeviceDual{T<:Real} <: Number
    value::T
    deriv::T
end

function DeviceDual(value::Real, deriv::Real)
    v, d = promote(value, deriv)
    return DeviceDual{typeof(v)}(v, d)
end

# A bare real seeds a constant dual (zero derivative).
DeviceDual{T}(value::Real) where {T<:Real} = DeviceDual{T}(convert(T, value), zero(T))

@inline _device_dual_basetype(::Type{DeviceDual{T}}) where {T} = T

# ---- promotion / conversion ----------------------------------------------------

Base.promote_rule(::Type{DeviceDual{T}}, ::Type{S}) where {T<:Real,S<:Real} =
    DeviceDual{promote_type(T, S)}
Base.promote_rule(::Type{DeviceDual{T}}, ::Type{DeviceDual{S}}) where {T<:Real,S<:Real} =
    DeviceDual{promote_type(T, S)}

Base.convert(::Type{DeviceDual{T}}, x::Real) where {T<:Real} = DeviceDual{T}(convert(T, x), zero(T))
Base.convert(::Type{DeviceDual{T}}, x::DeviceDual) where {T<:Real} =
    DeviceDual{T}(convert(T, x.value), convert(T, x.deriv))
Base.convert(::Type{DeviceDual{T}}, x::DeviceDual{T}) where {T<:Real} = x

Base.eps(::Type{DeviceDual{T}}) where {T<:Real} = DeviceDual{T}(eps(T), zero(T))
Base.zero(::Type{DeviceDual{T}}) where {T<:Real} = DeviceDual{T}(zero(T), zero(T))
Base.one(::Type{DeviceDual{T}}) where {T<:Real} = DeviceDual{T}(one(T), zero(T))
Base.zero(x::DeviceDual{T}) where {T<:Real} = zero(DeviceDual{T})
Base.one(x::DeviceDual{T}) where {T<:Real} = one(DeviceDual{T})
Base.float(x::DeviceDual) = x

# ---- arithmetic ----------------------------------------------------------------

@inline Base.:+(a::DeviceDual{T}, b::DeviceDual{T}) where {T} =
    DeviceDual{T}(a.value + b.value, a.deriv + b.deriv)
@inline Base.:-(a::DeviceDual{T}, b::DeviceDual{T}) where {T} =
    DeviceDual{T}(a.value - b.value, a.deriv - b.deriv)
@inline Base.:-(a::DeviceDual{T}) where {T} = DeviceDual{T}(-a.value, -a.deriv)
@inline Base.:+(a::DeviceDual{T}) where {T} = a
@inline Base.:*(a::DeviceDual{T}, b::DeviceDual{T}) where {T} =
    DeviceDual{T}(a.value * b.value, a.deriv * b.value + a.value * b.deriv)
@inline function Base.:/(a::DeviceDual{T}, b::DeviceDual{T}) where {T}
    inv_b = one(T) / b.value
    DeviceDual{T}(a.value * inv_b, (a.deriv * b.value - a.value * b.deriv) * inv_b * inv_b)
end
@inline Base.inv(a::DeviceDual{T}) where {T} = one(DeviceDual{T}) / a

# ---- powers --------------------------------------------------------------------

@inline function Base.:^(a::DeviceDual{T}, n::Integer) where {T}
    v = a.value^n
    DeviceDual{T}(v, T(n) * a.value^(n - 1) * a.deriv)
end
@inline function Base.:^(a::DeviceDual{T}, p::Real) where {T}
    v = a.value^p
    DeviceDual{T}(v, T(p) * a.value^(p - one(T)) * a.deriv)
end
@inline function Base.:^(a::DeviceDual{T}, b::DeviceDual{T}) where {T}
    v = a.value^b.value
    # d(a^b) = a^b (b' log a + b a'/a)
    DeviceDual{T}(v, v * (b.deriv * log(a.value) + b.value * a.deriv / a.value))
end

# ---- elementary functions ------------------------------------------------------

@inline function Base.exp(a::DeviceDual{T}) where {T}
    e = exp(a.value)
    DeviceDual{T}(e, e * a.deriv)
end
@inline Base.log(a::DeviceDual{T}) where {T} = DeviceDual{T}(log(a.value), a.deriv / a.value)
@inline Base.log1p(a::DeviceDual{T}) where {T} =
    DeviceDual{T}(log1p(a.value), a.deriv / (one(T) + a.value))
@inline function Base.expm1(a::DeviceDual{T}) where {T}
    DeviceDual{T}(expm1(a.value), exp(a.value) * a.deriv)
end
@inline function Base.sqrt(a::DeviceDual{T}) where {T}
    s = sqrt(a.value)
    DeviceDual{T}(s, a.deriv / (T(2) * s))
end
@inline function Base.abs(a::DeviceDual{T}) where {T}
    s = ifelse(a.value > zero(T), one(T), ifelse(a.value < zero(T), -one(T), zero(T)))
    DeviceDual{T}(abs(a.value), s * a.deriv)
end
# round is used only by the discrete-support checks; a rounded value is locally
# constant, so its derivative channel is zero.
@inline Base.round(a::DeviceDual{T}) where {T} = DeviceDual{T}(round(a.value), zero(T))
@inline function Base.sin(a::DeviceDual{T}) where {T}
    DeviceDual{T}(sin(a.value), cos(a.value) * a.deriv)
end
@inline function Base.cos(a::DeviceDual{T}) where {T}
    DeviceDual{T}(cos(a.value), -sin(a.value) * a.deriv)
end
@inline function Base.tanh(a::DeviceDual{T}) where {T}
    t = tanh(a.value)
    DeviceDual{T}(t, (one(T) - t * t) * a.deriv)
end
@inline function Base.tan(a::DeviceDual{T}) where {T}
    t = tan(a.value)
    DeviceDual{T}(t, (one(T) + t * t) * a.deriv)
end
@inline function Base.sinh(a::DeviceDual{T}) where {T}
    DeviceDual{T}(sinh(a.value), cosh(a.value) * a.deriv)
end
@inline function Base.cosh(a::DeviceDual{T}) where {T}
    DeviceDual{T}(cosh(a.value), sinh(a.value) * a.deriv)
end
@inline function Base.atan(a::DeviceDual{T}) where {T}
    DeviceDual{T}(atan(a.value), a.deriv / (one(T) + a.value * a.value))
end

# ---- min / max / clamp (value-selected branches) --------------------------------

@inline Base.min(a::DeviceDual{T}, b::DeviceDual{T}) where {T} = ifelse(a.value <= b.value, a, b)
@inline Base.max(a::DeviceDual{T}, b::DeviceDual{T}) where {T} = ifelse(a.value >= b.value, a, b)
@inline function Base.clamp(x::DeviceDual{T}, lo::DeviceDual{T}, hi::DeviceDual{T}) where {T}
    return ifelse(x.value < lo.value, lo, ifelse(x.value > hi.value, hi, x))
end

# ---- comparisons (on the value channel) ----------------------------------------

@inline Base.:(==)(a::DeviceDual, b::DeviceDual) = a.value == b.value
@inline Base.:<(a::DeviceDual, b::DeviceDual) = a.value < b.value
@inline Base.:<=(a::DeviceDual, b::DeviceDual) = a.value <= b.value
@inline Base.isless(a::DeviceDual, b::DeviceDual) = isless(a.value, b.value)
@inline Base.isnan(a::DeviceDual) = isnan(a.value)
@inline Base.isfinite(a::DeviceDual) = isfinite(a.value)
@inline Base.isinf(a::DeviceDual) = isinf(a.value)

# Value/derivative accessors that also accept a bare real (a step whose contribution
# never touched the seeded parameter stays a plain `T`).
@inline _device_dual_value(x::DeviceDual) = x.value
@inline _device_dual_value(x::Real) = x
@inline _device_dual_deriv(x::DeviceDual) = x.deriv
@inline _device_dual_deriv(x::Real) = zero(x)
