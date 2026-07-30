# Design Notes

UncertainTea keeps a set of design and reference documents alongside the code.
They are accurate to the direction of the project, though several currently lag
the implementation (tracked as the doc-currency issues #213–#217) and are being
migrated into this site over time.

For this first cut of the documentation site, the most user-facing material has
been adapted into [Getting Started](getting-started.md) and the
[Inference Overview](inference.md); the design documents below remain the
authoritative deep-dives. They live under `docs/` in the repository:

## Architecture and DSL

- [architecture.md](https://github.com/shohei81/uncertaintea/blob/main/docs/architecture.md)
  — the recommended architecture and layering strategy.
- [dsl.md](https://github.com/shohei81/uncertaintea/blob/main/docs/dsl.md)
  — the minimal DSL proposal, close to official Gen syntax.
- [constraint-driven-conditioning.md](https://github.com/shohei81/uncertaintea/blob/main/docs/constraint-driven-conditioning.md)
  — how observations vs. latents are classified from the conditioning signature.

## Batched and device execution

- [batched-inference.md](https://github.com/shohei81/uncertaintea/blob/main/docs/batched-inference.md)
  — the phased design for batched evaluators and GPU-oriented execution.
- [device-backend.md](https://github.com/shohei81/uncertaintea/blob/main/docs/device-backend.md)
  — device-resident batched logjoint via KernelAbstractions.
- [vector-backend-lowering.md](https://github.com/shohei81/uncertaintea/blob/main/docs/vector-backend-lowering.md)
  — bringing vector-valued latent families into the device subset.
- [device-vector-latents.md](https://github.com/shohei81/uncertaintea/blob/main/docs/device-vector-latents.md)
  — device-resident vector latents.

## Samplers

- [gpu-native-nuts.md](https://github.com/shohei81/uncertaintea/blob/main/docs/gpu-native-nuts.md)
  — design constraints for iterative, GPU-oriented NUTS.
- [persistent-nuts.md](https://github.com/shohei81/uncertaintea/blob/main/docs/persistent-nuts.md)
  — the persistent per-chain tree kernel (true one-launch GPU-native NUTS).
- [chees-hmc.md](https://github.com/shohei81/uncertaintea/blob/main/docs/chees-hmc.md)
  — ChEES-HMC, the cross-chain trajectory-length adaptive sampler.
- [mh-within-gibbs.md](https://github.com/shohei81/uncertaintea/blob/main/docs/mh-within-gibbs.md)
  — Metropolis-within-Gibbs for discrete structure.
- [discrete-enumeration.md](https://github.com/shohei81/uncertaintea/blob/main/docs/discrete-enumeration.md)
  — discrete enumeration / marginalization.
- [noncentered-reparam.md](https://github.com/shohei81/uncertaintea/blob/main/docs/noncentered-reparam.md)
  — automatic non-centered reparameterization with address preservation.

## Benchmarks and research

- [benchmarks.md](https://github.com/shohei81/uncertaintea/blob/main/docs/benchmarks.md)
  — cross-PPL correctness + performance results (NumPyro/Stan) and methodology;
  harness in `bench/crossppl/`.
- [research.md](https://github.com/shohei81/uncertaintea/blob/main/docs/research.md)
  — ecosystem research behind a GPU-native Julia PPL.

!!! info "Follow-up"
    Migrating these design docs into first-class pages (with their currency gaps
    corrected per #213–#217) and adding more Literate demos — a mixture model
    with `marginalize=:enumerate`, a logistic GLM on the device path, and a
    ChEES-vs-NUTS ESS comparison — are tracked as follow-up work.
