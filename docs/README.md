# Documentation

This folder tracks the current design direction of UncertainTea.

## Contents

- [research.md](research.md): ecosystem research and the implications for a GPU-native Julia PPL
- [architecture.md](architecture.md): the current recommended architecture and layering strategy
- [dsl.md](dsl.md): a minimal DSL proposal that is intentionally closer to official Gen syntax
- [batched-inference.md](batched-inference.md): the phased design for batched evaluators and GPU-oriented execution
- [gpu-native-nuts.md](gpu-native-nuts.md): the design constraints and staged plan for iterative, GPU-oriented NUTS
- [persistent-nuts.md](persistent-nuts.md): the one-launch-per-iteration persistent per-chain device NUTS tree (`tree_strategy=:persistent`, issue #154)
- [chees-hmc.md](chees-hmc.md): ChEES-HMC — cross-chain trajectory-length adaptation for the batched/device samplers (issue #161)
- [device-backend.md](device-backend.md): the KernelAbstractions device execution backend and its residency model
- [device-vector-latents.md](device-vector-latents.md): device lowering of vector-valued latent families and the unroll/dimension caps
- [vector-backend-lowering.md](vector-backend-lowering.md): staged design notes for bringing vector-valued latent families into the backend-native GPU subset
- [discrete-enumeration.md](discrete-enumeration.md): `marginalize=:enumerate` — summing discrete latents out of the log-joint
- [mh-within-gibbs.md](mh-within-gibbs.md): Metropolis-within-Gibbs for discrete latents alongside HMC/NUTS continuous updates
- [constraint-driven-conditioning.md](constraint-driven-conditioning.md): how observation constraints drive the signature-aware conditioned layout
- [noncentered-reparam.md](noncentered-reparam.md): staged design for automatic non-centered reparameterization with address preservation
- [benchmarks.md](benchmarks.md): cross-PPL correctness + performance results (NumPyro/Stan) with the methodology behind them; harness in `bench/crossppl/`

## Documentation Rules

- Update the relevant docs when a design decision changes
- Keep research notes dated and backed by primary-source links
- Update `dsl.md` and `architecture.md` together when the surface syntax changes
- Keep the boundary between the GPU static subset and CPU fallback explicit
