# Persistent per-chain NUTS tree kernel -- true GPU-native NUTS (issue #154, increment 2).
#
# WHAT THIS IS. `batched_nuts(...; tree_strategy=:persistent, backend=...)` builds the
# ENTIRE No-U-Turn tree for one NUTS iteration inside a SINGLE kernel launch. Each
# lane of the grid owns one chain (`ndrange = num_chains`) and runs the whole
# iterative-doubling recursion device-resident: it draws its own momentum and its own
# slice/direction/merge randomness from the on-device Philox RNG (increment 1,
# `src/device/rng.jl`), leapfrogs, computes the leapfrog gradient in-kernel by walking
# the lowered device plan in forward-mode duals (the SAME `_device_grad_score_steps`
# the grid gradient kernel uses, `src/device/gradient_kernel.jl`), keeps its dyadic
# U-turn checkpoint stack in a per-chain slice of a device buffer, and biased-
# progressively samples a proposal -- all with NO host round-trip mid-iteration.
#
# WHY ONE-LAUNCH-PER-ITERATION (vs the masked path's one-sync-per-round). The masked
# device NUTS (`nuts_kernels.jl`) is grid-parallel over (parameter, chain) and the
# HOST orchestrates the doubling, pre-drawing each round's randomness (only possible
# because every chain runs the SAME fixed round schedule in lockstep) and paying a
# host sync per doubling round. A persistent tree is DATA-DEPENDENT -- each chain
# U-turns/diverges at a different depth and consumes a different amount of randomness
# in a different order -- so the schedule cannot be pre-drawn. The kernel must draw
# its own randomness, which is exactly what the increment-1 counter-based RNG enables:
# a chain's draw is a pure function of the coordinate (chain_id, iteration, stream_id,
# draw_index), so no host coordination and no serial RNG state is needed. The per-
# iteration cost collapses to 1 launch + 1 sync + O(C) download.
#
# DESIGN SHIPPED: ONE THREAD PER CHAIN (not one threadgroup per chain). Each chain is
# one grid lane; its per-chain vector state (position / momentum / gradient of length
# P, and the P x (max_depth+1) checkpoint stack) lives in a per-chain COLUMN of a
# device buffer indexed by the chain id `b`, exactly like the gradient kernel's
# `slots[:, pidx, b]`. This deliberately AVOIDS threadgroup-shared memory and
# `@synchronize` (and the intra-group divergence hazards KA's group indexing invites),
# so the kernel compiles unchanged on CPU() and Metal. It is the right shape for the
# epic's headline regime -- thousands of chains (64-16384) -- where one lane per chain
# already saturates the GPU; the obs-tiled / P-parallel THREADGROUP-per-chain variant
# the #154 sketch also mentions helps the few-chains/large-N regime and is left as
# increment-3 follow-up (documented in docs/persistent-nuts.md).
#
# METAL ARGUMENT-BUFFER PACKING. Metal caps a kernel at 31 indirect argument-buffer
# resources. The per-chain tree needs many P x C / C working buffers, so related
# buffers are PACKED into multi-plane arrays to stay under the cap: the two frontiers
# each pack (position, momentum, gradient) into one `P x 3 x C` array, the subtree
# proposal packs (position, gradient) into `P x 2 x C`, the checkpoint stack packs
# (position, momentum) into `P x (D+1) x 2 x C`, the working sub-tree logjoints pack
# into `3 x C`, and the scalar diagnostics pack into `3 x C` (real) + `2 x C` (int).
# This keeps the launch at ~22 buffers, comfortably under the cap.
#
# STATISTICAL (NOT BITWISE) EQUIVALENCE. Per #154 risk (a), the on-device draws are
# not the host `Random` draws, so this path is validated by the #121 statistical gate
# (R-hat, posterior mean/quantile within MCSE, ~0 divergence), NOT by a bitwise oracle
# -- same contract as the masked async path. The algorithm ported here is the CANONICAL
# multinomial NUTS reproduced verbatim from the CPU reference:
#   * `_build_nuts_subtree`          (inference/nuts/tree_dynamics.jl) -- the per-leaf
#                                     leapfrog + even-leaf checkpoint + odd-leaf dyadic
#                                     U-turn + unbiased within-subtree multinomial pick.
#   * `_continue_nuts_proposal!`     (tree_dynamics.jl) -- the outer doubling loop,
#                                     invalid-doubling discard, biased progressive merge
#                                     (`_merge_subtree_stats`), whole-tree U-turn.
#   * `_initialize_nuts_first_step!` (nuts/state_init.jl) -- the first single-leaf
#                                     trajectory + unbiased initial multinomial.
#   * `leapfrog_step!`               (integrator.jl) -- half-kick / drift / half-kick.
#   * `_is_turning` / `_dyadic_turning` / `_logaddexp` / `_advance_tree_leaf` /
#     `_sample_momentum` / `_hamiltonian` -- the shared tree math.
#
# PRECISION-GENERIC: Float64 on CPU(), Float32 on Metal. Diagonal mass only (the device
# leapfrog consumes a single shared inverse-mass P-vector, as the masked path does).

# ---- RNG stream ids ------------------------------------------------------------
# Each purpose gets its own Philox `stream_id` so its draws never correlate with
# another purpose's (see rng.jl). `draw_index` walks position within a stream.
const _PERSIST_STREAM_MOMENTUM = 0x00000000  # momentum: draw_index = parameter index
const _PERSIST_STREAM_DIRECTION = 0x00000001 # doubling direction bit: draw_index = round
const _PERSIST_STREAM_LEAF = 0x00000002      # within-subtree multinomial: draw_index = global leaf
const _PERSIST_STREAM_MERGE = 0x00000003     # biased progressive merge: draw_index = round

# Frontier/checkpoint plane indices (see the packing note above).
const _PERSIST_POS = 1
const _PERSIST_MOM = 2
const _PERSIST_GRAD = 3

# ---- in-kernel device helpers --------------------------------------------------

# Device `_logaddexp` (transcription of nuts/state_init.jl `_logaddexp`), on plain T.
@inline function _persist_logaddexp(x::T, y::T) where {T}
    ninf = T(-Inf)
    if x == ninf
        return y
    elseif y == ninf
        return x
    end
    hi = max(x, y)
    return hi + log1p(exp(min(x, y) - hi))
end

# Kinetic energy 0.5 * sum(p^2 * inverse_mass) for chain b (diagonal metric), matching
# `_kinetic_energy` / `_hamiltonian` (hmc_support.jl).
@inline function _persist_kinetic(pmom, inverse_mass, b, num_params::Int, ::Type{T}) where {T}
    acc = zero(T)
    @inbounds for pidx = 1:num_params
        m = pmom[pidx, b]
        acc += m * m * inverse_mass[pidx]
    end
    return acc / 2
end

# Metric-aware whole-trajectory U-turn between the two packed frontiers `L` and `R`
# (planes POS/MOM), transcribing `_is_turning(left, right, left_mom, right_mom,
# inverse_mass)` (state_init.jl): the trajectory velocity is M^{-1} p, so the dots
# project momenta through inverse_mass.
@inline function _persist_is_turning_lr(L, R, inverse_mass, b, num_params::Int, ::Type{T}) where {T}
    left_dot = zero(T)
    right_dot = zero(T)
    @inbounds for pidx = 1:num_params
        delta = R[pidx, _PERSIST_POS, b] - L[pidx, _PERSIST_POS, b]
        im = inverse_mass[pidx]
        left_dot += delta * im * L[pidx, _PERSIST_MOM, b]
        right_dot += delta * im * R[pidx, _PERSIST_MOM, b]
    end
    return left_dot <= zero(T) || right_dot <= zero(T)
end

# Odd-leaf dyadic U-turn fold (transcription of `_dyadic_turning`, tree_expand.jl):
# for each dyadic block ending at `leaf_index` compare the block-start checkpoint
# (packed `ckpt[:, slot, POS/MOM, b]`) against the current endpoint (params/pmom).
# `forward` orients (checkpoint -> current); otherwise the pair swaps. Returns true as
# soon as any block turns.
@inline function _persist_dyadic_turning(
    ckpt, params, pmom, leaf_index::Int, forward::Bool, inverse_mass, b, num_params::Int, ::Type{T},
) where {T}
    @inbounds for k = 1:trailing_ones(leaf_index)
        block_start = leaf_index - (1 << k) + 1
        slot = count_ones(block_start) + 1
        left_dot = zero(T)
        right_dot = zero(T)
        for pidx = 1:num_params
            cp = ckpt[pidx, slot, 1, b]
            cm = ckpt[pidx, slot, 2, b]
            qp = params[pidx, b]
            qm = pmom[pidx, b]
            im = inverse_mass[pidx]
            if forward
                delta = qp - cp
                left_dot += delta * im * cm
                right_dot += delta * im * qm
            else
                delta = cp - qp
                left_dot += delta * im * qm
                right_dot += delta * im * cm
            end
        end
        if left_dot <= zero(T) || right_dot <= zero(T)
            return true
        end
    end
    return false
end

# In-kernel logjoint + full gradient at the position currently in `params[:, b]`.
# Two dispatch-selected variants (issue #154 increment 4), both walking the SAME lowered
# device plan the grid `_device_gradient_kernel!` uses (gradient_kernel.jl) so the whole
# leapfrog trajectory stays on-device. Both write `gradients[:, b]` and return
# `(logjoint, all_finite)`; `all_finite` folds the leapfrog validity check
# (`isfinite(value) && all(isfinite, grad)`, integrator.jl `leapfrog_step!`).
#
# SCALAR (`DeviceDual` slots): the shipped increment-2 path. Loops the differentiation
# target `pidx` over all P parameters and does a full O(N) scalar-dual plan walk per
# parameter -- O(P*N) serial, recomputing the full logjoint value P times. Kept as the
# default for small P (few-parameter models where the wide dual's N-wide arithmetic is
# not worth it) and unchanged from increment 2.
@inline function _persist_eval_grad!(
    gradients, plan, params, observed, observed_int, slots::AbstractArray{<:DeviceDual},
    trip_counts, loop_starts, b, num_params::Int, ::Type{T},
) where {T}
    logjoint = zero(T)
    ok = true
    @inbounds for pidx = 1:num_params
        total, _ = _device_grad_score_steps(
            plan.steps, slots, params, observed, observed_int, trip_counts, loop_starts, pidx, b, Int32(1),
        )
        g = _device_dual_deriv(total)
        gradients[pidx, b] = g
        ok = ok & isfinite(g)
        if pidx == 1
            logjoint = _device_dual_value(total)
        end
    end
    ok = ok & isfinite(logjoint)
    return (logjoint, ok)
end

# WIDE (`DeviceGradN{P}` slots): the increment-4 path. ONE forward-mode plan walk with a
# P-partial dual (`grad_wide.jl`) produces the whole gradient vector at once -- the
# `.partials` ARE `gradients[:, b]` and `.value` is the logjoint -- so each transcendental
# in each logpdf is evaluated ONCE instead of once per parameter. This is the heavy-per-
# gradient win (`logistic_large`: P=16, N=8000): the N*(P+1) transcendentals of the scalar
# path collapse to N. The single differentiation target is fixed at `1` (the wide seed is
# a one-hot at the parameter's own row, independent of the walk target -- see
# `_seed_latent`), and the slots buffer has a single `pidx` column. Auto-selected for
# large P; see the workspace constructor.
@inline function _persist_eval_grad!(
    gradients, plan, params, observed, observed_int, slots::AbstractArray{<:DeviceGradN},
    trip_counts, loop_starts, b, num_params::Int, ::Type{T},
) where {T}
    total, _ = _device_grad_score_steps(
        plan.steps, slots, params, observed, observed_int, trip_counts, loop_starts, 1, b, Int32(1),
    )
    partials = _device_gn_partials(total)
    ok = true
    @inbounds for pidx = 1:num_params
        g = partials[pidx]
        gradients[pidx, b] = g
        ok = ok & isfinite(g)
    end
    logjoint = _device_dual_value(total)
    ok = ok & isfinite(logjoint)
    return (logjoint, ok)
end

# ---- the per-chain tree driver -------------------------------------------------
#
# A PLAIN device function (not an `@kernel`), so it may use `break`/`return` freely --
# the `@kernel` entry point below just resolves `b` and calls this. All per-chain
# vector state lives in column `b` of the passed device buffers; all scalar tree state
# lives in registers. This mirrors, step for step, the CPU `_nuts_proposal` ->
# `_initialize_nuts_first_step!` -> `_continue_nuts_proposal!` -> `_build_nuts_subtree`
# call chain, with host `rand()` replaced by coordinate-indexed Philox draws.
@inline function _persist_nuts_chain!(
    b,
    params, gradients, pmom, q0, cur_logjoint,
    L, R, sub, lj3, prop_pos, prop_grad, prop_lj,
    ckpt,
    plan, observed, observed_int, slots, trip_counts, loop_starts,
    inverse_mass, step,
    out_real, out_int, out_div, out_moved,
    iteration::UInt32, max_tree_depth::Int, max_delta_energy::T, num_params::Int, ::Type{T},
) where {T}
    chain = UInt32(b)
    h = @inbounds step[b]
    lj0 = @inbounds cur_logjoint[b]

    # -- draw momentum (stream MOMENTUM, draw_index = pidx) and the initial energy --
    # `p = z / sqrt(inverse_mass)` reproduces `_sample_momentum` (hmc_support.jl):
    # p ~ N(0, M) for the diagonal metric M = diag(1 / inverse_mass).
    kinetic0 = zero(T)
    @inbounds for pidx = 1:num_params
        z = device_rand_normal(T, chain, iteration, _PERSIST_STREAM_MOMENTUM, UInt32(pidx - 1))
        p = z / sqrt(inverse_mass[pidx])
        pmom[pidx, b] = p
        kinetic0 += p * p * inverse_mass[pidx]
    end
    initial_hamiltonian = kinetic0 / 2 - lj0

    # -- seed the continuation frontier + proposal with the current state --
    # left = right = proposal = (q0, momentum, g0, lj0), matching the
    # `_copyto_nuts_state!` seeds in `_initialize_nuts_first_step!`.
    @inbounds for pidx = 1:num_params
        q = params[pidx, b]
        p = pmom[pidx, b]
        g = gradients[pidx, b]
        q0[pidx, b] = q
        L[pidx, _PERSIST_POS, b] = q
        L[pidx, _PERSIST_MOM, b] = p
        L[pidx, _PERSIST_GRAD, b] = g
        R[pidx, _PERSIST_POS, b] = q
        R[pidx, _PERSIST_MOM, b] = p
        R[pidx, _PERSIST_GRAD, b] = g
        prop_pos[pidx, b] = q
        prop_grad[pidx, b] = g
    end
    @inbounds lj3[_PERSIST_POS, b] = lj0   # plane 1 = left logjoint
    @inbounds lj3[_PERSIST_MOM, b] = lj0   # plane 2 = right logjoint
    @inbounds prop_lj[b] = lj0

    cont_log_weight = -initial_hamiltonian
    accept_sum = zero(T)
    accept_count = 0
    tree_depth = 1
    integration_steps = 0
    divergent = false
    turning = false
    proposal_energy = initial_hamiltonian
    proposal_error = zero(T)
    leaf_counter = 0

    # -- first single-leaf trajectory (`_initialize_nuts_first_step!`) --
    u = device_rand_uniform(T, chain, iteration, _PERSIST_STREAM_DIRECTION, UInt32(0))
    dir = u < T(0.5) ? -one(T) : one(T)
    forward = dir > zero(T)
    step_signed = dir * h
    half = step_signed / 2
    @inbounds for pidx = 1:num_params
        pmom[pidx, b] += half * gradients[pidx, b]
        params[pidx, b] += step_signed * (inverse_mass[pidx] * pmom[pidx, b])
    end
    lj1, ok1 = _persist_eval_grad!(
        gradients, plan, params, observed, observed_int, slots, trip_counts, loop_starts, b, num_params, T,
    )
    if !ok1
        # First step invalid: divergent, no move (integration_steps stays 0).
        divergent = true
    else
        @inbounds for pidx = 1:num_params
            pmom[pidx, b] += half * gradients[pidx, b]
        end
        # Record the far frontier in the integrated direction.
        F = forward ? R : L
        @inbounds for pidx = 1:num_params
            F[pidx, _PERSIST_POS, b] = params[pidx, b]
            F[pidx, _PERSIST_MOM, b] = pmom[pidx, b]
            F[pidx, _PERSIST_GRAD, b] = gradients[pidx, b]
        end
        @inbounds lj3[forward ? _PERSIST_MOM : _PERSIST_POS, b] = lj1
        proposed_energy = _persist_kinetic(pmom, inverse_mass, b, num_params, T) - lj1
        delta = proposed_energy - initial_hamiltonian
        integration_steps = 1
        if !isfinite(delta) || delta > max_delta_energy
            divergent = true
        else
            accept_prob = min(one(T), exp(min(zero(T), -delta)))
            accept_sum += accept_prob
            accept_count += 1
            candidate = -proposed_energy
            combined = _persist_logaddexp(cont_log_weight, candidate)
            uleaf = device_rand_uniform(T, chain, iteration, _PERSIST_STREAM_LEAF, UInt32(leaf_counter))
            # Unbiased multinomial between the initial point and the first leaf.
            if !isfinite(cont_log_weight) || log(uleaf) < candidate - combined
                @inbounds for pidx = 1:num_params
                    prop_pos[pidx, b] = params[pidx, b]
                    prop_grad[pidx, b] = gradients[pidx, b]
                end
                @inbounds prop_lj[b] = lj1
                proposal_energy = proposed_energy
                proposal_error = delta
            end
            cont_log_weight = combined
            turning = _persist_is_turning_lr(L, R, inverse_mass, b, num_params, T)
        end
        leaf_counter += 1
    end

    # -- outer doubling loop (`_continue_nuts_proposal!`) --
    while tree_depth < max_tree_depth && !divergent && !turning
        round = tree_depth  # distinct direction/merge draw index per doubling
        ud = device_rand_uniform(T, chain, iteration, _PERSIST_STREAM_DIRECTION, UInt32(round))
        dir = ud < T(0.5) ? -one(T) : one(T)
        forward = dir > zero(T)
        step_signed = dir * h
        half = step_signed / 2

        # Load the subtree start = continuation frontier in `dir` into the working
        # (params/pmom/gradients) current state.
        F = forward ? R : L
        @inbounds for pidx = 1:num_params
            params[pidx, b] = F[pidx, _PERSIST_POS, b]
            pmom[pidx, b] = F[pidx, _PERSIST_MOM, b]
            gradients[pidx, b] = F[pidx, _PERSIST_GRAD, b]
        end
        cur_lj_local = @inbounds lj3[forward ? _PERSIST_MOM : _PERSIST_POS, b]

        # Subtree scalar accumulators + subtree proposal seeded at the start state.
        sub_log_weight = T(-Inf)
        sub_integration = 0
        sub_accept_sum = zero(T)
        sub_accept_count = 0
        sub_divergent = false
        sub_turning = false
        sub_energy = zero(T)
        sub_error = zero(T)

        nleaves = 1 << tree_depth
        for _leaf = 1:nleaves
            @inbounds for pidx = 1:num_params
                pmom[pidx, b] += half * gradients[pidx, b]
                params[pidx, b] += step_signed * (inverse_mass[pidx] * pmom[pidx, b])
            end
            ljc, okc = _persist_eval_grad!(
                gradients, plan, params, observed, observed_int, slots, trip_counts, loop_starts, b, num_params, T,
            )
            if !okc
                sub_divergent = true
                break
            end
            @inbounds for pidx = 1:num_params
                pmom[pidx, b] += half * gradients[pidx, b]
            end
            cur_lj_local = ljc
            sub_integration += 1
            proposed_energy = _persist_kinetic(pmom, inverse_mass, b, num_params, T) - ljc
            delta = proposed_energy - initial_hamiltonian
            if !isfinite(delta) || delta > max_delta_energy
                sub_divergent = true
                break
            end
            accept_prob = min(one(T), exp(min(zero(T), -delta)))
            sub_accept_sum += accept_prob
            sub_accept_count += 1
            candidate = -proposed_energy
            combined = _persist_logaddexp(sub_log_weight, candidate)
            uleaf = device_rand_uniform(T, chain, iteration, _PERSIST_STREAM_LEAF, UInt32(leaf_counter))
            if !isfinite(sub_log_weight) || log(uleaf) < candidate - combined
                @inbounds for pidx = 1:num_params
                    sub[pidx, 1, b] = params[pidx, b]
                    sub[pidx, 2, b] = gradients[pidx, b]
                end
                @inbounds lj3[_PERSIST_GRAD, b] = ljc  # plane 3 = subtree proposal logjoint
                sub_energy = proposed_energy
                sub_error = delta
            end
            sub_log_weight = combined
            leaf_counter += 1

            leaf_index = sub_integration - 1
            if iseven(leaf_index)
                slot = count_ones(leaf_index) + 1
                @inbounds for pidx = 1:num_params
                    ckpt[pidx, slot, 1, b] = params[pidx, b]
                    ckpt[pidx, slot, 2, b] = pmom[pidx, b]
                end
            elseif _persist_dyadic_turning(
                ckpt, params, pmom, leaf_index, forward, inverse_mass, b, num_params, T,
            )
                sub_turning = true
                break
            end
        end

        tree_depth += 1
        if sub_integration == 0
            divergent = sub_divergent
            break
        end
        if sub_turning || sub_divergent
            # Canonical NUTS discards the whole invalid doubling: keep only the
            # accounting + flags (matches `_continue_nuts_proposal!`).
            integration_steps += sub_integration
            accept_sum += sub_accept_sum
            accept_count += sub_accept_count
            turning = sub_turning
            divergent = sub_divergent
            break
        end

        # Valid subtree: extend the continuation frontier in `dir` to the far endpoint.
        @inbounds for pidx = 1:num_params
            F[pidx, _PERSIST_POS, b] = params[pidx, b]
            F[pidx, _PERSIST_MOM, b] = pmom[pidx, b]
            F[pidx, _PERSIST_GRAD, b] = gradients[pidx, b]
        end
        @inbounds lj3[forward ? _PERSIST_MOM : _PERSIST_POS, b] = cur_lj_local

        # Biased progressive proposal swap (`_merge_subtree_stats`): P(swap) =
        # min(1, w_subtree / w_continuation), evaluated only for a finite subtree
        # weight, against the OLD continuation weight.
        combined_cw = cont_log_weight
        if isfinite(sub_log_weight)
            combined_cw = _persist_logaddexp(cont_log_weight, sub_log_weight)
            umerge = device_rand_uniform(T, chain, iteration, _PERSIST_STREAM_MERGE, UInt32(round))
            if log(umerge) < sub_log_weight - cont_log_weight
                @inbounds for pidx = 1:num_params
                    prop_pos[pidx, b] = sub[pidx, 1, b]
                    prop_grad[pidx, b] = sub[pidx, 2, b]
                end
                @inbounds prop_lj[b] = lj3[_PERSIST_GRAD, b]
                proposal_energy = sub_energy
                proposal_error = sub_error
            end
        end
        integration_steps += sub_integration
        accept_sum += sub_accept_sum
        accept_count += sub_accept_count
        if isfinite(sub_log_weight)
            cont_log_weight = combined_cw
        end
        # Whole-trajectory U-turn after the merge (`_merge_nuts_continuation_turning!`).
        turning = sub_turning || _persist_is_turning_lr(L, R, inverse_mass, b, num_params, T)
    end

    # Movement (accepted_step) in the backend's own precision: proposal != original.
    moved = 0x00
    @inbounds for pidx = 1:num_params
        if prop_pos[pidx, b] != q0[pidx, b]
            moved = 0x01
        end
    end
    @inbounds out_moved[b] = moved
    @inbounds out_real[1, b] = accept_count == 0 ? zero(T) : accept_sum / T(accept_count)
    @inbounds out_real[2, b] = proposal_energy
    @inbounds out_real[3, b] = proposal_error
    @inbounds out_div[b] = divergent ? 0x01 : 0x00
    @inbounds out_int[1, b] = Int32(tree_depth)
    @inbounds out_int[2, b] = Int32(integration_steps)
    return nothing
end

# ---- the kernel entry point ----------------------------------------------------
# One lane per chain; the whole per-chain tree runs in `_persist_nuts_chain!`.
@kernel function _device_persistent_nuts_kernel!(
    params, gradients, pmom, q0, @Const(cur_logjoint),
    L, R, sub, lj3, prop_pos, prop_grad, prop_lj,
    ckpt,
    plan, @Const(observed), @Const(observed_int), slots, @Const(trip_counts), @Const(loop_starts),
    @Const(inverse_mass), @Const(step),
    out_real, out_int, out_div, out_moved,
    iteration::UInt32, max_tree_depth::Int, max_delta_energy, num_params::Int,
)
    b = @index(Global)
    _persist_nuts_chain!(
        b,
        params, gradients, pmom, q0, cur_logjoint,
        L, R, sub, lj3, prop_pos, prop_grad, prop_lj,
        ckpt,
        plan, observed, observed_int, slots, trip_counts, loop_starts,
        inverse_mass, step,
        out_real, out_int, out_div, out_moved,
        iteration, max_tree_depth, convert(eltype(params), max_delta_energy), num_params, eltype(params),
    )
end

# ---- host-side workspace -------------------------------------------------------
#
# Self-contained (does NOT touch the masked `DeviceNUTSWorkspace`): wraps a
# `DeviceBatchedWorkspace` for the lowered plan + dual gradient scratch, and allocates
# the per-chain tree buffers (packed per the Metal note above). Diagonal mass only.
mutable struct DevicePersistentNUTSWorkspace{T,B<:KernelAbstractions.Backend}
    inner::DeviceBatchedWorkspace{T}
    backend::B
    num_params::Int
    num_chains::Int
    max_tree_depth::Int
    gradient_mode::Symbol    # :scalar (per-parameter DeviceDual walk) or :wide (single
    # DeviceGradN{P} walk); resolved from the constructor's `gradient_mode`/`P`.
    grad_slots::Any          # the dual-slot scratch handed to the kernel: the inner
    # `DeviceDual{T}` buffer for :scalar, or a staged `DeviceGradN{P,T}` buffer for :wide.
    inverse_mass::Any        # P
    step::Any                # C   per-chain step size
    cur_logjoint::Any        # C   current-position logjoint (input)
    pmom::Any                # P x C  working momentum
    q0::Any                  # P x C  device copy of the current position (moved test)
    left::Any                # P x 3 x C  (pos, mom, grad)
    right::Any               # P x 3 x C
    sub::Any                 # P x 2 x C  subtree proposal (pos, grad)
    lj3::Any                 # 3 x C   working logjoints (left, right, subtree-proposal)
    prop_pos::Any            # P x C   continuation proposal position (downloaded)
    prop_grad::Any           # P x C   continuation proposal gradient (downloaded)
    prop_lj::Any             # C       continuation proposal logjoint (downloaded)
    ckpt::Any                # P x (max_depth+1) x 2 x C  (pos, mom)
    out_real::Any            # 3 x C  (accept_prob, proposed_energy, energy_error)
    out_int::Any             # 2 x C  (tree_depth, integration_steps) Int32
    out_div::Any             # C  UInt8 divergent
    out_moved::Any           # C  UInt8 accepted/moved
    # host staging (no per-iteration allocation)
    host_energy::Vector{T}
    host_u8::Vector{UInt8}
    host_i32::Vector{Int32}
    host_mat::Matrix{T}
    host_real::Matrix{T}     # 3 x C
    host_int::Matrix{Int32}  # 2 x C
    step_host::Vector{T}
    inverse_mass_host::Vector{T}
    # Per-RUN seed folded into the Philox iteration coordinate so different host `rng`
    # seeds give different (but reproducible) device draws. Drawn lazily from the
    # sampler's `rng` on the first iteration (`seeded` gates it); `run_seed + iteration`
    # is distinct across iterations, and the Philox key `(chain_id, that)` is distinct
    # across chains, iterations, AND runs -- no cross-run collisions.
    run_seed::UInt32
    seeded::Bool
end

# Auto-selection threshold (issue #154 increment 4). Below this many unconstrained
# parameters the scalar per-parameter walk wins (its P plan walks are cheap and the wide
# dual's P-wide NTuple arithmetic adds fixed overhead per op); at or above it the wide
# single-walk's collapse of the redundant per-parameter logjoint/transcendental
# recomputation pays off. 8 keeps every small model (gauss P<=2, the two-parameter model
# P=2, eight-schools) on the proven scalar path while routing the heavy GLMs
# (`logistic` P=9, `logistic_large` P=17) to the wide path. Tunable per call via
# `gradient_mode=:scalar|:wide`.
const _PERSIST_WIDE_MIN_PARAMS = 8

# Stage the model's Real scalar arguments into the wide (`DeviceGradN{P,T}`) slot buffer
# as zero-partial constants, mirroring `_device_stage_gradient_arguments!` for the scalar
# buffer but with a single `pidx` column (the wide walk uses one column). Non-Real args
# (e.g. the covariate matrix) are supplied per-observation, not through a slot.
function _device_stage_gradient_arguments_wide!(
    wide_slots, model::TeaModel, args, batch_size::Int, ::Type{T}, ::Val{P},
) where {T,P}
    argument_slots = executionplan(model).environment_layout.argument_slots
    isempty(argument_slots) && return nothing
    TD = DeviceGradN{P,T}
    staged = Array{TD}(undef, 1, 1, batch_size)
    for (argument_index, slot) in enumerate(argument_slots)
        if args isa Tuple
            value = args[argument_index]
            value isa Real || continue
            fill!(staged, _seed_obs(TD, value))
        else
            all(args[batch_index][argument_index] isa Real for batch_index = 1:batch_size) || continue
            for batch_index = 1:batch_size
                staged[1, 1, batch_index] = _seed_obs(TD, args[batch_index][argument_index])
            end
        end
        copyto!(view(wide_slots, slot:slot, :, :), staged)
    end
    return nothing
end

function DevicePersistentNUTSWorkspace(
    model::TeaModel,
    num_chains::Integer,
    max_tree_depth::Integer;
    backend::KernelAbstractions.Backend=KernelAbstractions.CPU(),
    precision::Type=Float64,
    args=(),
    constraints=choicemap(),
    gradient_mode::Symbol=:auto,
)
    gradient_mode in (:auto, :scalar, :wide) || throw(
        ArgumentError(
            "DevicePersistentNUTSWorkspace gradient_mode must be :auto, :scalar, or :wide, got $(repr(gradient_mode))",
        ),
    )
    inner = DeviceBatchedWorkspace(
        model, num_chains; backend=backend, precision=precision, args=args, constraints=constraints,
    )
    _device_ensure_gradient_buffers!(inner)
    T = precision
    P = inner.parameter_count
    C = inner.batch_size
    D = Int(max_tree_depth)
    # Resolve :auto -> :scalar / :wide from the parameter count (see the threshold note).
    resolved_mode = gradient_mode === :auto ? (P >= _PERSIST_WIDE_MIN_PARAMS ? :wide : :scalar) : gradient_mode
    # The kernel is generic over the slot buffer's dual type: the scalar buffer is the
    # inner `DeviceDual{T}` scratch; the wide buffer is a `DeviceGradN{P,T}` scratch with
    # a single `pidx` column, staged with the model's constant arguments.
    grad_slots = if resolved_mode === :wide
        wide = KernelAbstractions.allocate(backend, DeviceGradN{P,T}, inner.slot_count, 1, C)
        copyto!(wide, fill(zero(DeviceGradN{P,T}), inner.slot_count, 1, C))
        _device_stage_gradient_arguments_wide!(wide, model, args, C, T, Val(P))
        wide
    else
        inner.grad_slots_device
    end
    mat() = fill!(KernelAbstractions.allocate(backend, T, P, C), zero(T))
    plane(k) = fill!(KernelAbstractions.allocate(backend, T, P, k, C), zero(T))
    vecT() = fill!(KernelAbstractions.allocate(backend, T, C), zero(T))
    vecU8() = fill!(KernelAbstractions.allocate(backend, UInt8, C), 0x00)
    ckptbuf = fill!(KernelAbstractions.allocate(backend, T, P, max(D + 1, 1), 2, C), zero(T))
    return DevicePersistentNUTSWorkspace{T,typeof(backend)}(
        inner, backend, P, C, D,
        resolved_mode, grad_slots,
        fill!(KernelAbstractions.allocate(backend, T, P), zero(T)),
        vecT(), vecT(),
        mat(), mat(),
        plane(3), plane(3), plane(2),
        fill!(KernelAbstractions.allocate(backend, T, 3, C), zero(T)),
        mat(), mat(), vecT(),
        ckptbuf,
        fill!(KernelAbstractions.allocate(backend, T, 3, C), zero(T)),
        fill!(KernelAbstractions.allocate(backend, Int32, 2, C), Int32(0)),
        vecU8(), vecU8(),
        Vector{T}(undef, C),
        Vector{UInt8}(undef, C),
        Vector{Int32}(undef, C),
        Matrix{T}(undef, P, C),
        Matrix{T}(undef, 3, C),
        Matrix{Int32}(undef, 2, C),
        Vector{T}(undef, C),
        Vector{T}(undef, P),
        0x00000000,
        false,
    )
end

# ---- host driver (one launch per iteration) ------------------------------------
#
# Fills the host `BatchedNUTSWorkspace` (ws) proposal outputs for one iteration from a
# SINGLE persistent-kernel launch. Host work: upload current (position, gradient,
# logjoint) + mass/step, launch, one sync, download O(C) scalars + the P x C proposal
# position/gradient. No per-round host RNG or round-trip.
function _device_persistent_nuts_proposals!(
    dws::DevicePersistentNUTSWorkspace{T},
    ws::BatchedNUTSWorkspace,
    model::TeaModel,
    position::AbstractMatrix{Float64},
    current_logjoint::AbstractVector{Float64},
    current_gradient::AbstractMatrix{Float64},
    inverse_mass_matrix,
    args,
    constraints,
    step_size,
    max_tree_depth::Int,
    max_delta_energy::Float64,
    iteration::Integer,
    rng::AbstractRNG,
) where {T}
    be = dws.backend
    inner = dws.inner
    P = dws.num_params
    C = dws.num_chains

    # Draw the per-run seed once (from the sampler rng) so seeds vary results.
    if !dws.seeded
        dws.run_seed = rand(rng, UInt32)
        dws.seeded = true
    end
    iteration_coord = dws.run_seed + UInt32(iteration)

    # Upload current state. `params_device` is BOTH the leapfrog start position (read
    # by the in-kernel gradient) and the working position; the kernel snapshots it into
    # `q0` before integrating for the post-hoc movement test.
    _upload_matrix!(inner.params_device, position, dws.host_mat)
    _upload_matrix!(inner.gradients_device, current_gradient, dws.host_mat)
    @inbounds for c = 1:C
        dws.host_energy[c] = current_logjoint[c]
    end
    copyto!(dws.cur_logjoint, dws.host_energy)

    # Shared diagonal mass (P-vector) + per-chain step (C-vector). A scalar step
    # broadcasts into the C-vector so the kernel always indexes `step[b]`.
    dws.inverse_mass_host .= convert.(T, inverse_mass_matrix)
    copyto!(dws.inverse_mass, dws.inverse_mass_host)
    if step_size isa AbstractVector
        @inbounds for c = 1:C
            dws.step_host[c] = convert(T, step_size[c])
        end
    else
        s = convert(T, step_size)
        @inbounds for c = 1:C
            dws.step_host[c] = s
        end
    end
    copyto!(dws.step, dws.step_host)

    _device_persistent_nuts_kernel!(be)(
        inner.params_device, inner.gradients_device, dws.pmom, dws.q0, dws.cur_logjoint,
        dws.left, dws.right, dws.sub, dws.lj3, dws.prop_pos, dws.prop_grad, dws.prop_lj,
        dws.ckpt,
        inner.plan, inner.observed_device, inner.observed_int_device, dws.grad_slots,
        inner.trip_counts_device, inner.loop_starts_device,
        dws.inverse_mass, dws.step,
        dws.out_real, dws.out_int, dws.out_div, dws.out_moved,
        iteration_coord, Int(max_tree_depth), convert(T, max_delta_energy), P;
        ndrange=C,
    )
    KernelAbstractions.synchronize(be)

    # Download the accepted proposal (P x C) + the O(C) diagnostics into the host ws.
    _download_matrix!(ws.proposal_position, dws.prop_pos, dws.host_mat)
    _download_matrix!(ws.proposal_gradient, dws.prop_grad, dws.host_mat)
    _download_reals!(ws.proposed_logjoint, dws.prop_lj, dws.host_energy)
    _download_bits!(ws.control.divergent_step, dws.out_div, dws.host_u8)
    _download_bits!(ws.control.accepted_step, dws.out_moved, dws.host_u8)
    copyto!(dws.host_real, dws.out_real)
    copyto!(dws.host_int, dws.out_int)
    @inbounds for c = 1:C
        ws.accept_prob[c] = Float64(dws.host_real[1, c])
        ws.proposed_energy[c] = Float64(dws.host_real[2, c])
        ws.energy_error[c] = Float64(dws.host_real[3, c])
        ws.control.tree_depths[c] = Int(dws.host_int[1, c])
        ws.control.integration_steps[c] = Int(dws.host_int[2, c])
    end
    return ws
end

# ---- device-strategy dispatch --------------------------------------------------
# `batched_nuts`'s device call sites hand a device workspace + the iteration index to
# ONE entry point; the workspace TYPE selects the strategy. The masked workspace host-
# pre-draws its RNG and so ignores `iteration`; the persistent workspace uses it as the
# Philox iteration coordinate and ignores `rng` (the kernel draws its own randomness).
function _device_nuts_proposals_dispatch!(
    dws::DeviceNUTSWorkspace, ws, model, position, current_logjoint, current_gradient,
    inverse_mass_matrix, args, constraints, step_size, max_tree_depth, max_delta_energy, iteration, rng,
)
    return _device_batched_nuts_proposals_masked!(
        dws, ws, model, position, current_logjoint, current_gradient, inverse_mass_matrix,
        args, constraints, step_size, max_tree_depth, max_delta_energy, rng,
    )
end

function _device_nuts_proposals_dispatch!(
    dws::DevicePersistentNUTSWorkspace, ws, model, position, current_logjoint, current_gradient,
    inverse_mass_matrix, args, constraints, step_size, max_tree_depth, max_delta_energy, iteration, rng,
)
    return _device_persistent_nuts_proposals!(
        dws, ws, model, position, current_logjoint, current_gradient, inverse_mass_matrix,
        args, constraints, step_size, max_tree_depth, max_delta_energy, iteration, rng,
    )
end
