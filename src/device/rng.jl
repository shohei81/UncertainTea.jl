# On-device counter-based RNG (Philox4x32-10) -- issue #154, increment 1.
#
# WHY this exists at all. The persistent per-chain NUTS tree kernel (#154's
# endgame: one kernel launch per iteration, each threadgroup building its own
# device-resident tree) cannot use the host-pre-drawn randomness the current
# masked device NUTS relies on. The masked design can pre-draw all of a doubling
# ROUND's uniforms on the host and upload them (`leaf_uniform` per round) ONLY
# because the round structure is fixed and identical across chains. A persistent
# tree is DATA-DEPENDENT: each chain U-turns or diverges at a different depth, so
# the host has no way to know, ahead of the launch, which per-chain randomness
# will actually be consumed. The kernel must therefore draw its own randomness
# *inside* the kernel. Until this file there was NO on-device RNG anywhere in the
# package -- momenta and round uniforms were all host-drawn (`Random` on the
# host) and uploaded. This is that missing prerequisite; nothing calls it yet
# (increment 2 wires it into the tree kernel).
#
# WHY counter-based (Philox) rather than a stateful stream (e.g. a per-chain
# xoshiro carried in threadgroup memory). A counter-based generator is a PURE
# FUNCTION from a coordinate to a random block: `philox(key, counter) -> block`.
# That buys three things the persistent kernel needs and a stateful PRNG cannot
# give cheaply:
#   * No mutable RNG state to thread through the (register/threadgroup) tree
#     walk, and no serial dependence between draws -- any leaf can draw its own
#     randomness from its (chain, iteration, purpose, index) coordinate directly.
#   * Reproducibility WITHOUT host coordination: the same coordinate yields the
#     same draw on every backend and every launch, so a chain's trajectory is
#     replayable for debugging and the #121 statistical-equivalence gate.
#   * Deterministic and BIT-IDENTICAL across the CPU() reference backend and
#     Metal at the same precision, because the algorithm is pure integer
#     arithmetic (32-bit multiply/xor/add) with no floating-point in the mixing.
#
# WHY Philox4x32 specifically (vs Philox2x32 / Threefry). The coordinate a
# persistent kernel indexes by is genuinely 4-dimensional -- (chain_id,
# iteration, stream_id, draw_index) -- and Philox4x32 has a 2-word (64-bit) key
# and a 4-word (128-bit) counter, so every coordinate maps to its own key/counter
# word with ZERO folding or collision risk: key = (chain_id, iteration),
# counter = (stream_id, draw_index, 0, 0). It emits four 32-bit words per call
# (two Float64 or four Float32 uniforms, or a Box-Muller normal pair) -- efficient
# for drawing a whole momentum vector. Philox is also the generator #154's design
# sketch names, and its 10-round variant is the Random123 / Salmon et al. (2011)
# default with published known-answer test vectors we validate against.
#
# HARD CONSTRAINTS honored here (so this compiles and runs inside an `@kernel`):
# every function is `@inline`, pure, allocation-free (only isbits `NTuple`s and
# scalars), branch-free in the hot path, and contains no `Random`, no `throw`,
# and no mutable global state. The 10 rounds are fully unrolled so there is no
# dynamic loop for a GPU compiler to trip over. Precision is generic: the derived
# `device_rand_*` helpers take the working type `T` (Float32 for Metal, Float64
# for CPU()) and build a uniform/normal from the integer block for that T.

# --- Philox4x32-10 constants (Salmon et al. 2011 / Random123) -----------------
# Multipliers for the two 32-bit lanes and the two Weyl key-bump increments
# (golden ratio and sqrt(3)-1 fractional parts). Stored as UInt32.
const _PHILOX_M4x32_0 = 0xD2511F53 % UInt32
const _PHILOX_M4x32_1 = 0xCD9E8D57 % UInt32
const _PHILOX_W32_0 = 0x9E3779B9 % UInt32
const _PHILOX_W32_1 = 0xBB67AE85 % UInt32

# 32x32 -> 64-bit widening multiply, split into (hi, lo) 32-bit halves. Widening
# through UInt64 is device-safe (both CPU() and Metal do 64-bit integer multiply)
# and, crucially, keeps the mixing in EXACT integer arithmetic so the result is
# bit-identical across backends.
@inline function _philox_mulhilo(a::UInt32, b::UInt32)
    prod = UInt64(a) * UInt64(b)
    return (UInt32(prod >> 32), UInt32(prod & 0x00000000ffffffff))
end

# One Philox4x32 round: two independent multiply-xor lanes that cross-feed the
# counter words (out[0] mixes lane-1's high half with ctr[1] and key[0], etc.).
# Matches the Random123 `_philox4x32round` word ordering exactly.
@inline function _philox4x32_round(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    hi0, lo0 = _philox_mulhilo(_PHILOX_M4x32_0, ctr[1])
    hi1, lo1 = _philox_mulhilo(_PHILOX_M4x32_1, ctr[3])
    return (hi1 ⊻ ctr[2] ⊻ key[1], lo1, hi0 ⊻ ctr[4] ⊻ key[2], lo0)
end

# Weyl key schedule: bump each key word by its odd increment between rounds.
@inline _philox4x32_bumpkey(key::NTuple{2,UInt32}) =
    (key[1] + _PHILOX_W32_0, key[2] + _PHILOX_W32_1)

"""
    philox4x32(key::NTuple{2,UInt32}, counter::NTuple{4,UInt32}) -> NTuple{4,UInt32}

Pure, stateless Philox4x32-10 bijection: maps a `(key, counter)` coordinate to a
128-bit random block (four 32-bit words). Device-safe (no allocation, no
branches, fully unrolled) and bit-identical across the CPU() KA backend and
Metal at any given input. Validated against the Random123 / Salmon et al. (2011)
known-answer test vectors (see `test/uncertaintea/core/device_rng.jl`).

The `key` is the "which independent stream" coordinate and the `counter` is the
"position within the stream"; for the persistent NUTS kernel these are formed
from `(chain_id, iteration)` and `(stream_id, draw_index, 0, 0)` respectively via
the `device_rand_*` helpers.
"""
@inline function philox4x32(key::NTuple{2,UInt32}, counter::NTuple{4,UInt32})
    # 10 rounds, unrolled. Round 1 uses the raw key; each subsequent round uses
    # the Weyl-bumped key (bump BEFORE the round, matching Random123's
    # `if(R>1){ key = bumpkey(key); ctr = round(ctr,key); }`).
    ctr = _philox4x32_round(counter, key)          # round 1
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 2
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 3
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 4
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 5
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 6
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 7
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 8
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 9
    key = _philox4x32_bumpkey(key)
    ctr = _philox4x32_round(ctr, key)              # round 10
    return ctr
end

# --- integer block -> floating-point draws ------------------------------------
# WHY these particular conversions. We want a uniform in the half-open [0, 1)
# using the full mantissa the type can carry, with NO floating-point rounding to
# exactly 1.0 (which would blow up `log` in the normal transform). We take the
# TOP bits of the integer word(s) (they are the best-mixed) and scale by an exact
# negative power of two, so the map is a clean fixed-point-to-float divide that is
# bit-identical across backends.

# Float32: 24-bit mantissa. Take the top 24 bits of one 32-bit word and multiply
# by 2^-24. Range [0, 1 - 2^-24].
@inline _philox_u01(::Type{Float32}, w::UInt32) = Float32(w >> 8) * Float32(0x1p-24)

# Float64: 53-bit mantissa. Assemble a 64-bit value from two words, take the top
# 53 bits and multiply by 2^-53. Range [0, 1 - 2^-53].
@inline function _philox_u01(::Type{Float64}, w0::UInt32, w1::UInt32)
    bits = (UInt64(w0) << 32) | UInt64(w1)
    return Float64(bits >> 11) * 0x1p-53
end

# Precision-generic uniform from a full Philox block: Float32 uses the first
# output word; Float64 uses the first two. (Higher words are left for callers
# that want a second independent draw from the same block, e.g. Box-Muller.)
@inline _philox_uniform(::Type{Float32}, block::NTuple{4,UInt32}) =
    _philox_u01(Float32, block[1])
@inline _philox_uniform(::Type{Float64}, block::NTuple{4,UInt32}) =
    _philox_u01(Float64, block[1], block[2])

# --- coordinate-indexed public helpers ----------------------------------------
# WHY the (chain_id, iteration, stream_id, draw_index) coordinate. This is the
# interface the persistent kernel actually wants: a leaf deep in chain `c`'s tree
# at iteration `it` that needs its `d`-th draw for purpose `s` (e.g. s=0 momentum,
# s=1 slice/accept uniforms) calls one of these with (c, it, s, d) and gets a
# reproducible, independent draw with no host coordination and no shared state.
# `stream_id` separates purposes so momentum draws never correlate with accept
# draws; `draw_index` walks position within a purpose.

"""
    device_rand_uniform(::Type{T}, chain_id, iteration, stream_id, draw_index) -> T

A uniform in `[0, 1)` of working precision `T`, drawn deterministically from the
coordinate. Pure and device-safe. `key = (chain_id, iteration)`,
`counter = (stream_id, draw_index, 0, 0)`; distinct coordinates give independent
draws (Philox is a bijection, so no two coordinates collide).
"""
@inline function device_rand_uniform(
    ::Type{T},
    chain_id::UInt32,
    iteration::UInt32,
    stream_id::UInt32,
    draw_index::UInt32,
) where {T}
    block = philox4x32((chain_id, iteration), (stream_id, draw_index, 0x00000000, 0x00000000))
    return _philox_uniform(T, block)
end

"""
    device_rand_normal(::Type{T}, chain_id, iteration, stream_id, draw_index) -> T

A standard-normal `N(0, 1)` draw of working precision `T` via the Box-Muller
transform of two uniforms taken from a SINGLE Philox block (Float32: words 1 and
2; Float64: word-pairs (1,2) and (3,4)). Pure and device-safe -- this is the
momentum-draw primitive the persistent kernel needs. `1 - u` maps the radius
uniform into `(0, 1]` so `log` never sees an exact zero.

CROSS-BACKEND CAVEAT (issue #154 risk (a), measured on Apple M4): unlike
`philox4x32` and `device_rand_uniform` -- which are BIT-IDENTICAL across the
CPU() backend and Metal because they are pure integer mixing plus an exact
integer->float divide -- this Gaussian is only STATISTICALLY equivalent across
backends. Box-Muller evaluates `log` and `cos`, whose platform libm differs from
the host's in the last bit(s) (~5e-7 Float32 drift observed), so a chain's
momentum draws are NOT reproducible bitwise between CPU and Metal. The persistent
NUTS kernel must therefore be validated against the CPU reference by the #121
statistical-equivalence gate, not a bitwise comparison. Within a single backend
at a single precision the draw is still fully deterministic from its coordinate.
"""
@inline function device_rand_normal(
    ::Type{T},
    chain_id::UInt32,
    iteration::UInt32,
    stream_id::UInt32,
    draw_index::UInt32,
) where {T}
    block = philox4x32((chain_id, iteration), (stream_id, draw_index, 0x00000000, 0x00000000))
    return _box_muller(T, block)
end

# Box-Muller: z = sqrt(-2 ln u1) * cos(2*pi*u2). One block yields both uniforms.
@inline function _box_muller(::Type{Float32}, block::NTuple{4,UInt32})
    u1 = _philox_u01(Float32, block[1])
    u2 = _philox_u01(Float32, block[2])
    radius = sqrt(-2.0f0 * log(1.0f0 - u1))   # 1 - u1 in (0, 1], safe for log
    return radius * cos(6.2831855f0 * u2)     # 2*pi in Float32
end
@inline function _box_muller(::Type{Float64}, block::NTuple{4,UInt32})
    u1 = _philox_u01(Float64, block[1], block[2])
    u2 = _philox_u01(Float64, block[3], block[4])
    radius = sqrt(-2.0 * log(1.0 - u1))       # 1 - u1 in (0, 1], safe for log
    return radius * cos(6.283185307179586 * u2)  # 2*pi in Float64
end
