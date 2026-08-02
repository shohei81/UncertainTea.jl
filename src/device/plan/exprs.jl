# ---- device expressions --------------------------------------------------------

abstract type AbstractDeviceExpr end

struct DeviceLiteralExpr{T} <: AbstractDeviceExpr
    value::T
end

struct DeviceSlotExpr <: AbstractDeviceExpr
    slot::Int32
end

struct DevicePrimitiveExpr{Op,A<:Tuple} <: AbstractDeviceExpr
    args::A
end

DevicePrimitiveExpr(op::Symbol, args::Tuple) = DevicePrimitiveExpr{op,typeof(args)}(args)

# Per-element covariate leaf inside a broadcast observation expression (issue #134).
# A broadcast-normal mu/sigma expression references a per-observation COVARIATE
# vector (a model argument like `xs` in `{:y} ~ normal.(a .+ b .* xs, s)`). Such a
# vector argument is NOT a scalar slot the kernel can read from the `slots` buffer:
# it varies per observed element, so -- exactly like the GLM covariate column
# (issue #150) -- staging rides its per-element values on the observation buffer,
# addresses erased. Each distinct covariate vector referenced by mu/sigma gets a
# fixed `offset` inside the per-element observation block; the broadcast handler
# reads it at `elem_base + offset`. Constants (zero derivative in the gradient
# kernel): they never seed a dual.
struct DeviceObservedColumnExpr <: AbstractDeviceExpr
    offset::Int32
end
