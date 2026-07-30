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

## Increment 2 implementation notes (CPU, `src/inference/api_batched_chees.jl`)

The estimator reproduces `blackjax.adaptation.chees_adaptation` (`base.compute_parameters`, chees_adaptation.py). Confirmed against the source line-for-line:

- **Jitter / step count.** `jitter_val = halton(i)*jitter_amount + (1 - jitter_amount) ∈ [1-jitter_amount, 1)` (source `jitter_gn`, lines 763-765); `L_iter = max(1, ceil(jitter_val * T / ε))` (source `integration_steps_fn` lines 767-771, with `num_leapfrog_steps = T/ε` line 862). The *same* jitter draw feeds both the step count and the gradient (source lines 461 & 769). This REPLACES the increment-1 scaffold formula `floor(num_leapfrog_steps·(1+jitter_amount·h))`.
- **Whitening** (Σ ≡ diagonal inverse-mass, source lines 450-458): `Δx'_w = (proposal - proposals_mean)/√Σ`, `Δx_w = (initial - initials_mean)/√Σ`, `v'_w = p'·Σ/√Σ = p'·√Σ`. `proposals_mean` is the acceptance-weighted, divergence-masked mean; `initials_mean` is the plain cross-chain mean (source lines 376-386).
- **Per-chain gradient** `g_c = jitter_val·T·(‖Δx'_w‖² − ‖Δx_w‖²)·⟨Δx'_w, v'_w⟩` (source lines 460-466); **reduction** `trajectory_gradient = Σ_{~div} accept_c·g_c / Σ_{~div} (accept_c + EPS)` with `EPS = 1e-20` (source lines 468-471).
- **Adam on `log T`**, then clip to `±LOG_UPDATE_CLIP = 0.35`, then moving average `log_T_ma = (1-w)·log_T_ma + w·log_T_new`, `w = step^(-decay_rate)`, `decay_rate = 0.5`, `T = clamp(exp(log_T_ma), ε, max_leapfrog_steps·ε)` (source lines 371-501). Non-finite updates revert both `log T` and the Adam moments (source lines 482-489).
- **Adam hyperparameters.** BlackJAX leaves `optim` to the caller (line 741); its own ChEES tests/examples uniformly pass `optax.adam(learning_rate=0.5, b1=0, b2=0.95)` (eps default 1e-8). Those are our defaults (`trajectory_learning_rate=0.5`, `trajectory_adam_beta1=0.0`, `trajectory_adam_beta2=0.95`, `trajectory_adam_epsilon=1e-8`).
- **Step size** dual-averages to the **harmonic mean** of the chains' accept probs over non-divergent chains (source lines 358-363); `target_accept = 0.651`.

### Deviations from BlackJAX (with justification)

1. **Ascent sign.** We MAXIMIZE ChEES by ascending: `log T += clip(+lr·mhat/(√vhat+eps), ±0.35)`. BlackJAX passes the *raw* `trajectory_gradient` to `optim.update` and applies it with `optax.apply_updates` (lines 474-481); composed with the `optax.adam` it uses (a minimizer, whose updates are `−lr·…`), that path descends the gradient. Since the estimator equals `+d(ChEES)/d(log T)` and the documented intent is to "maximize the ChEES criterion" (docstring line 259), we ascend directly. Empirically confirmed correct: `T` converges to a finite sensible value and posteriors are recovered (a descent sign drives `T` to the `ε` floor).
2. **Warm-started `T`.** We initialize `T = num_leapfrog_steps · ε` (using the step size entering the loop) so the first iteration matches the scaffold's fixed step count and `adapt_trajectory_length=false` preserves fixed-length sampling. BlackJAX starts `T = ε` (one step) and adapts up. The moving average uses `w = 1` on the first step (`step` starts at 1), so the warm start only affects behavior before the first update / when adaptation is off.
3. **Float-robust `ceil`.** `ceil(jitter_val·T/ε)` snaps a value within a `1e-9` relative tolerance of an integer to it before rounding up, so `T = L·ε` gives `L` exactly (a bare `ceil` on `1.2/0.1 = 12.0000000002` would return 13). Behavior-invisible except at that round-off boundary.
4. **Halton index base.** Our Halton is 1-based (`_halton_base2(iteration)`, reused from the merged scaffold and its test) versus BlackJAX's 0-based `random_generator_arg`; this only shifts the deterministic jitter schedule by one index and does not affect the algorithm.
5. **Diagnostics seam.** The adapted `T` trace is exposed through a private `_trajectory_trace` keyword (fills a caller-supplied vector) rather than a new result field, so `HMCChains`/`HMCChain` are unchanged.

Not ported (BlackJAX extras outside increment-2 scope): the opt-in `mass_matrix_estimation="diagonal"` Welford metric, the slow-direction length floor (`CHEES_LENGTH_FLOOR_FACTOR`), and the `_whiten_criterion` ablation seam. UncertainTea supplies the shared diagonal mass via the existing pooled `WarmupDriver`.

## Reuse map (existing infrastructure)

- `_batched_leapfrog!` (`src/inference/integrator.jl`) — batched `L`-step integrator.
- `_batched_hamiltonian!` / `_hamiltonian` — energies + MH ratio.
- `_sample_batched_momentum!`, `_batched_acceptance_probability!`.
- Shared dual-averaging + pooled diagonal mass (the `per_chain_adaptation=false` path in `_batched_hmc_*`); ChEES is inherently a shared-ensemble adaptation.
- `_initial_batched_hmc_positions` (+ #162 retry / `init=:uniform`).
- Device: `_device_nuts_leaf_*` kick/drift/Hamiltonian kernels already exist; the device ChEES loop needs no tree code (one sync/iteration + an O(C) host or device reduction for θ̄ and the ChEES gradient).

## Incremental plan (each an independent PR)

1. **Scaffold (CPU, this note's increment 1):** `batched_chees` = Halton-jittered fixed-length HMC with shared dual-averaging step size + pooled diagonal mass and a FIXED user trajectory length (`num_leapfrog_steps` / `trajectory_length`), no ChEES adaptation yet. Returns the same `HMCChains` result type. Validate: samples the conjugate gauss posterior (mean/std within tolerance), determinism under a seed, argument validation. Keeps NUTS untouched.
2. **ChEES adaptation (CPU) — DONE (this increment):** cross-chain whitened ChEES criterion + scalar Adam gradient ascent on `log T` during warmup, step size dual-averaged to the harmonic mean of accept probs. Matches `blackjax.adaptation.chees_adaptation` (see "Increment 2 implementation notes" below). Validated: warmup converges `T` on the conjugate gauss (`T ≈ 15`) and an ill-conditioned diagonal Gaussian with 100× scale span (`T ≈ 1.1`, ≈ the quarter-turn `π/2` ChEES converges to under a whitening mass); posterior mean/marginal-sd recovered within tolerance with `rhat < 1.05`; ESS-per-gradient is stable across a 8× sweep of the initial `num_leapfrog_steps` (fixed-L swings ~3×), 2.4× better than fixed-L at an over-long initial length; SBC (`sampler=:chees`) rank-uniformity p ≈ 0.89 (no warnings).
3. **Correctness gates + SBC — DONE (this increment):** added a `chees` variant to the `bench/crossppl` runner (`run.jl`) and a `chees` leg to `run_all.sh`, reported under the `uncertaintea-chees` label. `sbc(sampler=:chees)` already landed in increment 2. Measured (gauss, target-accept 0.651, the many-chain GPU-story model): ChEES **PASSES** the #121 correctness gate (rank-normalized split R-hat < 1.01, mean/quantile within combined MCSEs) at 4 and 64 chains with 0 divergences, at min-bulk ESS/s ≈ **3.4e5** — vs NUTS legs on gauss (uncertaintea-batched-cpu ≈ 1.5e3, numpyro-parallel ≈ 6.8e3, stan ≈ 5.5e3) — because a well-tuned fixed short trajectory on this easy target beats tree overhead. Caveat: gauss is a trivial target, so this ESS/s is a best case, not representative of hard posteriors. The eight-schools funnel is **excluded** from the ChEES leg: `batched_hmc`/`batched_chees` currently throw on `eight_schools_noncentered` (a #157-class gap — the noncentered-reparam finite-check is not caught by the reject path inside the batched HMC leapfrog gradient; `batched_nuts` is unaffected). Tracked as a follow-up; add the funnel models to the ChEES leg once fixed.
4. **Device — DONE (this increment):** `batched_chees(...; backend=<KA backend>, precision=...)` now routes to `_run_device_batched_chees` (`src/device/chees_kernels.jl`), the device analogue of the shared-mode `_run_device_batched_hmc`. It reuses the residency-looped `_device_leapfrog_integrate!` (one `KernelAbstractions.synchronize` per iteration, regardless of the jittered `L_iter`) and the device Hamiltonian/accept-column kernels, and it REUSES the CPU ChEES math verbatim — `_chees_jitter_value` / `_chees_leapfrog_steps_from_jitter` for the per-iteration step count and `_chees_trajectory_update!` / `_ChEESTrajectoryState` / `_chees_harmonic_mean_accept` for the adaptation (no fork, no CPU behavior change; the increment-1 `backend`-guard throw is removed and `backend`/`precision` thread through). Validated on the conjugate gauss (device CPU() backend, Float64) as statistically equivalent to the host `batched_chees` — posterior mean/sd recovered, adapted `T` converges to `T ≈ 15` (same neighborhood as the host), and on an ill-conditioned diagonal Gaussian; the Metal Float32 leg (`test/gpu/runtests.jl`) recovers the gauss posterior (mean ≈ 0.147, sd ≈ 0.706, R-hat ≈ 1.0, `T ≈ 16`).

   **Device-specific deviations (with justification):**
   - **Shared mass only.** The device leapfrog kernels consume a single SHARED diagonal inverse-mass vector (same constraint as device NUTS/HMC), so the device path uses a shared step + shared (pooled) diagonal mass. ChEES is inherently an ensemble adaptation, so this matches the CPU ChEES adaptation mode — nothing is lost relative to the host `batched_chees`.
   - **Adaptation download during warmup only.** The trajectory TIME `T` is a host scalar; each WARMUP iteration downloads the PROPOSAL position (`ws.inner.params_device`) and the ENDPOINT momentum (`ws.working_momentum`), both P×C, so the host can run the shared `_chees_trajectory_update!`. The `initials` are the pre-transition positions the host `position` mirror still holds at the top of the iteration (snapshotted before the end-of-iteration accepted-position download). This ChEES download is the ONLY per-iteration transfer beyond device HMC's, and it happens ONLY during warmup — **sampling iterations download NOTHING for adaptation**, staying a pure device loop with one sync per iteration exactly like device HMC. With `adapt_trajectory_length=false` the ChEES download is elided entirely (and `T` is frozen).
   - **Endpoint-momentum sign.** The device final half-kick (`_device_hmc_final_halfkick!`) negates the endpoint momentum, matching the CPU `_batched_leapfrog!` `_negate_column!`, so the downloaded `ws.working_momentum` feeds the ChEES gradient's whitened-velocity term with the same sign as the host path.

## Non-goals / invariants

- NUTS remains the CPU/reference default; ChEES is additive.
- No change to existing `batched_hmc` / `batched_nuts` behavior or RNG order.
