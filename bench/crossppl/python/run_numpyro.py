"""NumPyro runner for the cross-PPL benchmark (issue #121).

Usage (from bench/crossppl):
    uv run --project python python/run_numpyro.py --model eight_schools_noncentered \
        --chains 4 --samples 1000 --warmup 1000 --seed 1 --reps 3 \
        --chain-method parallel

Timing protocol: `MCMC.warmup()` and `MCMC.run()` are timed separately, so
warmup and sampling wall-clock are measured directly.  The first
warmup+sampling pair in the process includes JIT compilation and is reported
as ttfx_s; the timed repetitions run on the already-compiled kernels.
"""

import argparse
import time

import numpy as np

import numpyro


def build_model(model_name: str, data: dict):
    import jax.numpy as jnp
    import numpyro.distributions as dist

    if model_name.startswith("eight_schools"):
        y = jnp.asarray(data["y"])
        sigma = jnp.asarray(data["sigma"])
        J = data["J"]
        centered = model_name == "eight_schools_centered"

        def model():
            mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
            tau = numpyro.sample("tau", dist.HalfCauchy(5.0))
            if centered:
                with numpyro.plate("J", J):
                    theta = numpyro.sample("theta", dist.Normal(mu, tau))
            else:
                with numpyro.plate("J", J):
                    z = numpyro.sample("theta_z", dist.Normal(0.0, 1.0))
                theta = numpyro.deterministic("theta", mu + tau * z)
            with numpyro.plate("obs", J):
                numpyro.sample("y", dist.Normal(theta, sigma), obs=y)

        return model, ["mu", "tau", "theta"]

    if model_name in ("logistic", "logistic_large"):
        X = jnp.asarray(np.array(data["X"]))  # (n, d) rows
        y = jnp.asarray(np.array(data["y"], dtype=np.float32))
        d = data["d"]

        def model():
            alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.5))
            with numpyro.plate("D", d):
                beta = numpyro.sample("beta", dist.Normal(0.0, 2.5))
            logits = alpha + X @ beta
            with numpyro.plate("obs", X.shape[0]):
                numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

        return model, ["alpha", "beta"]

    if model_name == "poisson_glm":
        # issue #317: the vectorized Poisson GLM — NumPyro's natural formulation,
        # matching UncertainTea's broadcast `{:y} ~ poisson.(exp.(a .+ b .* x))`
        # and Stan's `y ~ poisson_log(a + b * x)`.
        x = jnp.asarray(data["x"])
        y = jnp.asarray(np.array(data["y"], dtype=np.int32))

        def model():
            a = numpyro.sample("a", dist.Normal(0.0, 1.0))
            b = numpyro.sample("b", dist.Normal(0.0, 1.0))
            with numpyro.plate("obs", x.shape[0]):
                numpyro.sample("y", dist.Poisson(jnp.exp(a + b * x)), obs=y)

        return model, ["a", "b"]

    if model_name == "schools_large":
        # issue #317: J=32 hierarchy with log-normal tau (the P >= 24
        # reverse-mode leg on the UncertainTea side), NONCENTERED since issue
        # #365 (post-#326). log_tau is the sampled parameter in every framework
        # and theta is exported as a deterministic so the columns align by name.
        y = jnp.asarray(data["y"])
        sigma = jnp.asarray(data["sigma"])
        J = data["J"]

        def model():
            mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
            log_tau = numpyro.sample("log_tau", dist.Normal(0.0, 1.0))
            with numpyro.plate("J", J):
                z = numpyro.sample("theta_z", dist.Normal(0.0, 1.0))
            theta = numpyro.deterministic("theta", mu + jnp.exp(log_tau) * z)
            with numpyro.plate("obs", J):
                numpyro.sample("y", dist.Normal(theta, sigma), obs=y)

        return model, ["mu", "log_tau", "theta"]

    if model_name == "gauss":
        y = jnp.asarray(data["y"])

        def model():
            mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
            s = numpyro.sample("s", dist.Gamma(2.0, 1.0))
            with numpyro.plate("obs", y.shape[0]):
                numpyro.sample("y", dist.Normal(mu, s), obs=y)

        return model, ["mu", "s"]

    if model_name == "mixture":
        # issue #224: 2-component Gaussian mixture, ordered means, shared scale,
        # fixed 0.4/0.6 weights, marginalized via MixtureSameFamily -- the same
        # marginal density Stan hand-writes with log_mix and the DSL's `mixture`
        # machinery computes. Same parameterization as the Julia/Stan twins.
        y = jnp.asarray(data["y"])
        weights = jnp.array([0.4, 0.6])

        def model():
            mu1 = numpyro.sample("mu1", dist.Normal(0.0, 3.0))
            log_gap = numpyro.sample("log_gap", dist.Normal(0.0, 1.0))
            s = numpyro.sample("s", dist.Gamma(2.0, 1.0))
            mu2 = mu1 + jnp.exp(log_gap)
            mixing = dist.Categorical(probs=weights)
            components = dist.Normal(jnp.stack([mu1, mu2]), s)
            with numpyro.plate("obs", y.shape[0]):
                numpyro.sample("y", dist.MixtureSameFamily(mixing, components), obs=y)

        return model, ["mu1", "log_gap", "s"]

    if model_name == "lkj":
        # issue #224: d=2 LKJ-Cholesky correlation prior + log-normal scales. The
        # deterministic `Omega` reproduces UncertainTea's packed lower-triangular
        # correlation factor ([L11, L21, L22]) so the gate aligns by name.
        y = jnp.asarray(np.array(data["y"]))  # (n, 2)

        def model():
            L = numpyro.sample("L", dist.LKJCholesky(2, concentration=2.0))
            with numpyro.plate("d", 2):
                tau = numpyro.sample("tau", dist.LogNormal(0.0, 0.5))
            numpyro.deterministic("Omega", jnp.array([L[0, 0], L[1, 0], L[1, 1]]))
            scale_tril = tau[:, None] * L
            with numpyro.plate("obs", y.shape[0]):
                numpyro.sample("y", dist.MultivariateNormal(jnp.zeros(2), scale_tril=scale_tril), obs=y)

        return model, ["Omega", "tau"]

    raise ValueError(f"unknown model {model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chain-method", default="parallel",
                    choices=["parallel", "vectorized", "sequential"])
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--max-tree-depth", type=int, default=10)
    ap.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True,
                    help="float64 like Stan/UncertainTea-CPU (default); "
                         "--no-x64 matches the float32 device sweep")
    args = ap.parse_args()

    if args.chain_method == "parallel":
        numpyro.set_host_device_count(args.chains)
    if args.x64:
        numpyro.enable_x64()

    import jax
    from numpyro.infer import MCMC, NUTS

    from common import flatten_params, git_commit, load_data, write_result

    data_name = "eight_schools" if args.model.startswith("eight_schools") else args.model
    data = load_data(data_name)
    model, order = build_model(args.model, data)

    kernel = NUTS(model, target_accept_prob=args.target_accept,
                  max_tree_depth=args.max_tree_depth, dense_mass=False)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, chain_method=args.chain_method,
                progress_bar=False)

    def one_pass(seed: int):
        key_warmup, key_sample = jax.random.split(jax.random.PRNGKey(seed))
        t0 = time.perf_counter()
        mcmc.warmup(key_warmup, collect_warmup=False)
        jax.block_until_ready(mcmc.post_warmup_state)
        t1 = time.perf_counter()
        # After warmup(), run() continues from the post-warmup state.
        mcmc.run(key_sample, extra_fields=("diverging",))
        samples = mcmc.get_samples(group_by_chain=True)
        jax.block_until_ready(samples)
        t2 = time.perf_counter()
        return samples, t1 - t0, t2 - t1

    # Cold pass: includes JIT compilation of warmup and sampling kernels.
    _, cold_warm, cold_samp = one_pass(args.seed)
    ttfx_s = cold_warm + cold_samp
    print(f"cold pass (ttfx): {ttfx_s:.2f} s")

    for rep in range(1, args.reps + 1):
        seed = args.seed + rep
        samples, warmup_s, sampling_s = one_pass(seed)
        # sequential chain_method returns already-grouped arrays either way
        names, draws = flatten_params(samples, order)
        extra = mcmc.get_extra_fields(group_by_chain=True)
        div_rate = float(np.mean(np.asarray(extra["diverging"]))) if "diverging" in extra else None
        stem = write_result(
            model=args.model, framework="numpyro", variant=args.chain_method,
            params=names, draws=np.asarray(draws),
            num_chains=args.chains, num_samples=args.samples, num_warmup=args.warmup,
            seed=seed, rep=rep,
            timings={"ttfx_s": ttfx_s, "warmup_s": warmup_s,
                     "sampling_s": sampling_s, "total_s": warmup_s + sampling_s},
            sampler={"kind": "nuts", "target_accept": args.target_accept,
                     "max_tree_depth": args.max_tree_depth, "metric": "diag",
                     "chain_parallelism": args.chain_method,
                     "precision": "float64" if args.x64 else "float32"},
            env={"numpyro": numpyro.__version__, "jax": jax.__version__,
                 "jax_platform": jax.default_backend(), "bench_commit": git_commit()},
            diagnostics={"divergence_rate": div_rate},
        )
        print(f"{stem}  warmup={warmup_s:.3f}s  sampling={sampling_s:.3f}s  div={div_rate}")


if __name__ == "__main__":
    main()
