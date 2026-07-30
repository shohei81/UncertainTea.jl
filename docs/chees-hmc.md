# ChEES-HMC (issue #161) — design note

ChEES-HMC (Hoffman, Radul, Sountsov, *An Adaptive-MCMC Scheme for Setting Trajectory Lengths*, AISTATS 2021) is a many-parallel-chain sampler that replaces NUTS's per-chain recursive tree with **fixed-length jittered HMC** whose trajectory length is adapted from **cross-chain** statistics. Every lane does identical work every step (no control-flow divergence, no per-leaf host sync), so it maps to a handful of kernels with one sync per iteration — the GPU-native route the #121 audit wants. We add it as a NEW sampler `batched_chees`; NUTS stays the reference/default.

Primary sources (verify the exact gradient against these in increment 2):
- Paper: https://proceedings.mlr.press/v130/hoffman21a/hoffman21a.pdf
- BlackJAX reference implementation: `blackjax.adaptation.chees_adaptation` (https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/chees_adaptation/index.html)

## Algorithm

Per iteration, all `C` chains share one step size `ε`, one diagonal mass, and one trajectory length `T`:

1. Draw momentum per chain (`p ~ N(0, M)`).
2. Jittered step count: `L = floor(T / ε * (1 + jitter_amount * h))`, clamped to `>= 1`, where `h ∈ [0,1)` is the current Halton-sequence value (one draw per iteration, shared across chains). `jitter_amount` defaults to 1.0.
3. Integrate every chain `L` leapfrog steps (reuse `_batched_leapfrog!`).
4. Standard Metropolis accept per chain.
5. Warmup adaptation:
   - **Step size** `ε`: dual averaging to a target accept (default `OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651`), driven by the **harmonic mean** of the chains' accept probabilities.
   - **Diagonal mass**: Welford over a window (reuse existing pooled-mass machinery).
   - **Trajectory length** `T`: gradient *ascent* on the ChEES criterion via Adam on `log T` (see below).

### ChEES criterion and trajectory-length gradient

Criterion (θ̄ = cross-chain mean of positions):

    ChEES = (1/4) · E[ ( ‖θ' − θ̄‖² − ‖θ − θ̄‖² )² ]

Halfway identity: extending the trajectory time by `dT` moves the endpoint θ' by `v' · dT`, where `v' = M⁻¹ p'` is the endpoint velocity. Hence `∂/∂T ‖θ'−θ̄‖² = 2 (θ'−θ̄) · v'`, so the per-chain stochastic gradient is

    g_i ∝ ( ‖θ'_i − θ̄‖² − ‖θ_i − θ̄‖² ) · ( (θ'_i − θ̄) · v'_i )

weighted by the chain's acceptance probability `p_accept_i` and averaged across chains. BlackJAX whitens the distances by the inverse mass and uses the *proposal* (not the accepted state) with an acceptance weight; `decay_rate` (default 0.5) sets the Adam moving-average weighting and `LOG_UPDATE_CLIP = 0.35` clips the per-iteration `log T` update. **Increment 2 must reproduce BlackJAX's exact estimator (whitening, weighting, half-step velocity, Adam constants) — cross-check against the source, do not approximate from this note.**

## Reuse map (existing infrastructure)

- `_batched_leapfrog!` (`src/inference/integrator.jl`) — batched `L`-step integrator.
- `_batched_hamiltonian!` / `_hamiltonian` — energies + MH ratio.
- `_sample_batched_momentum!`, `_batched_acceptance_probability!`.
- Shared dual-averaging + pooled diagonal mass (the `per_chain_adaptation=false` path in `_batched_hmc_*`); ChEES is inherently a shared-ensemble adaptation.
- `_initial_batched_hmc_positions` (+ #162 retry / `init=:uniform`).
- Device: `_device_nuts_leaf_*` kick/drift/Hamiltonian kernels already exist; the device ChEES loop needs no tree code (one sync/iteration + an O(C) host or device reduction for θ̄ and the ChEES gradient).

## Incremental plan (each an independent PR)

1. **Scaffold (CPU, this note's increment 1):** `batched_chees` = Halton-jittered fixed-length HMC with shared dual-averaging step size + pooled diagonal mass and a FIXED user trajectory length (`num_leapfrog_steps` / `trajectory_length`), no ChEES adaptation yet. Returns the same `HMCChains` result type. Validate: samples the conjugate gauss posterior (mean/std within tolerance), determinism under a seed, argument validation. Keeps NUTS untouched.
2. **ChEES adaptation (CPU):** add the cross-chain criterion + Adam-on-`log T` during warmup (BlackJAX-exact). Validate on gauss / eight-schools / an ill-conditioned target; check ESS/s and that warmup converges `T`.
3. **Correctness gates + SBC:** add a `sampler=chees` variant to `bench/crossppl`; run the #121 gates and `sbc(...)` (`bench/sbc_validation.jl`). Report as its own bench row, not replacing NUTS.
4. **Device (follow-up):** mirror onto the device kernels (one sync/iteration, host/device O(C) reduction). Delegate after the CPU sampler is validated.

## Non-goals / invariants

- NUTS remains the CPU/reference default; ChEES is additive.
- No change to existing `batched_hmc` / `batched_nuts` behavior or RNG order.
