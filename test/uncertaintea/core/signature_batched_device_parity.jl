# issue #95, PR-4: the batched and device layout/staging derive the
# observed-vs-latent split from the CONDITIONING SIGNATURE (the set of
# constrained addresses), the same source of truth the CPU `logjoint` path uses
# (PR-2/PR-3). The core deliverable is that CPU, the CPU backend (batched), and
# the device path all agree on a model where the SAME bound address is observed
# in one run (constrained) and latent in another (unconstrained) -- two
# conditioning signatures, two cached layouts -- and that the batched
# parameter-vector length follows the signature.
#
# All device calls run on KernelAbstractions.CPU() (no GPU in CI).

using KernelAbstractions: CPU as SigParityCPU

# A bound choice `y` that is used downstream. `:y` observed -> observation (no
# slot); `:y` unconstrained -> latent (slot). `:z` is always observed here.
@tea static function sig_parity_model()
    mu ~ normal(0.0, 1.0)
    y = ({:y} ~ normal(mu, 1.0))
    {:z} ~ normal(y, 1.0)
end

# manual reference log-density in constrained space, for a conjugate crosscheck.
function _sig_parity_logdensity(mu, y, z)
    n(x, m, s) = -0.5 * log(2pi) - log(s) - 0.5 * ((x - m) / s)^2
    return n(mu, 0.0, 1.0) + n(y, mu, 1.0) + n(z, y, 1.0)
end

@testset "signature_batched_device_parity" begin
    cpu_backend = SigParityCPU()
    zval = 0.5

    # --- signature A: :y observed (constrained) -> only mu is latent ---------
    cons_a = choicemap(:y => 2.5, :z => zval)
    layout_a =
        UncertainTea._resolve_signature_plan(sig_parity_model, cons_a, ()).plan.parameter_layout
    @test parametercount(layout_a) == 1        # mu only
    @test parametervaluecount(layout_a) == 1

    params_a = reshape([0.3], 1, 1)
    cpu_a = logjoint_unconstrained(sig_parity_model, [0.3], (), cons_a)
    bat_a = batched_logjoint_unconstrained(sig_parity_model, params_a, (), cons_a)
    dev_a = device_batched_logjoint(sig_parity_model, params_a, (), cons_a; backend=cpu_backend)
    @test bat_a[1] ≈ cpu_a rtol = 1e-12
    @test dev_a[1] ≈ cpu_a rtol = 1e-12
    # closed form: mu identity transform, so unconstrained == constrained.
    @test cpu_a ≈ _sig_parity_logdensity(0.3, 2.5, zval) rtol = 1e-12

    # gradient parity (CPU vs batched vs device)
    g_cpu_a = logjoint_gradient_unconstrained(sig_parity_model, [0.3], (), cons_a)
    g_bat_a = batched_logjoint_gradient_unconstrained(sig_parity_model, params_a, (), cons_a)
    _, g_dev_a = device_batched_logjoint_gradient(sig_parity_model, params_a, (), cons_a; backend=cpu_backend)
    @test vec(g_bat_a) ≈ g_cpu_a rtol = 1e-10
    @test vec(g_dev_a) ≈ g_cpu_a rtol = 1e-10

    # constrained-space entry point (batched_logjoint) also follows the signature.
    cons_a_score = batched_logjoint(sig_parity_model, params_a, (), cons_a)
    @test cons_a_score[1] ≈ logjoint(sig_parity_model, [0.3], (), cons_a) rtol = 1e-12

    # --- signature B: :y latent (unconstrained) -> mu AND y are latent -------
    cons_b = choicemap(:z => zval)
    layout_b =
        UncertainTea._resolve_signature_plan(sig_parity_model, cons_b, ()).plan.parameter_layout
    @test parametercount(layout_b) == 2        # mu and y
    @test parametervaluecount(layout_b) == 2

    params_b = reshape([0.3, 1.7], 2, 1)
    cpu_b = logjoint_unconstrained(sig_parity_model, [0.3, 1.7], (), cons_b)
    bat_b = batched_logjoint_unconstrained(sig_parity_model, params_b, (), cons_b)
    dev_b = device_batched_logjoint(sig_parity_model, params_b, (), cons_b; backend=cpu_backend)
    @test bat_b[1] ≈ cpu_b rtol = 1e-12
    @test dev_b[1] ≈ cpu_b rtol = 1e-12
    @test cpu_b ≈ _sig_parity_logdensity(0.3, 1.7, zval) rtol = 1e-12

    g_cpu_b = logjoint_gradient_unconstrained(sig_parity_model, [0.3, 1.7], (), cons_b)
    g_bat_b = batched_logjoint_gradient_unconstrained(sig_parity_model, params_b, (), cons_b)
    _, g_dev_b = device_batched_logjoint_gradient(sig_parity_model, params_b, (), cons_b; backend=cpu_backend)
    @test vec(g_bat_b) ≈ g_cpu_b rtol = 1e-10
    @test vec(g_dev_b) ≈ g_cpu_b rtol = 1e-10

    # --- the batched parameter-vector length follows the signature -----------
    # signature A expects 1 latent, so a 2-row matrix (sized for B) is rejected.
    @test_throws DimensionMismatch batched_logjoint_unconstrained(sig_parity_model, params_b, (), cons_a)
    # signature B expects 2 latents, so a 1-row matrix (sized for A) is rejected.
    @test_throws DimensionMismatch batched_logjoint_unconstrained(sig_parity_model, params_a, (), cons_b)

    # --- device lowering report is signature-aware and supported both ways ---
    supported_a, issues_a = device_lowering_report(sig_parity_model; constraints=cons_a)
    supported_b, issues_b = device_lowering_report(sig_parity_model; constraints=cons_b)
    @test supported_a
    @test isempty(issues_a)
    @test supported_b
    @test isempty(issues_b)

    # --- memoization: repeated runs at the same conditioning reuse the cache --
    cache_before = length(sig_parity_model.signature_cache[])
    for _ = 1:3
        batched_logjoint_unconstrained(sig_parity_model, params_a, (), cons_a)
        batched_logjoint_unconstrained(sig_parity_model, params_b, (), cons_b)
        device_batched_logjoint(sig_parity_model, params_a, (), cons_a; backend=cpu_backend)
    end
    # still exactly the two signatures resolved above (A and B); no recompute.
    @test length(sig_parity_model.signature_cache[]) == cache_before
end
