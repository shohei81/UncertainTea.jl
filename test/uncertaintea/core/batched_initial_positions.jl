# Batched initial-position draw (issue #156): `_initial_batched_hmc_positions`
# draws each chain's prior initial position through the latent-only compiled
# plan walk (`_sample_prior_steps!`) instead of a per-chain traced `generate`
# over every observation address. The walk must consume the chain RNG at
# exactly the sites the traced path did (constrained sites read their value
# without touching the RNG), so with the unchanged per-chain seed derivation
# (seeds = rand(rng, UInt, num_chains); chain_rng = MersenneTwister(seed))
# every chain's position is BITWISE identical to the pre-#156 implementation.
# These tests pin that identity by keeping the old code path here verbatim.

@tea static function bip_gauss_model(n)
    mu ~ normal(0.0, 1.0)
    sigma ~ lognormal(0.0, 1.0)
    for i = 1:n
        {:y => i} ~ normal(mu, sigma)
    end
    return mu
end

@tea static function bip_schools_model(J, sigmas)
    mu ~ normal(0.0, 5.0)
    tau ~ lognormal(0.0, 1.0)
    theta ~ iid(normal(mu, tau), 8; reparam=:noncentered)
    for j = 1:J
        {:y => j} ~ normal(theta[j], sigmas[j])
    end
    return theta
end

@tea static function bip_broadcast_model(xs)
    slope ~ normal(0.0, 10.0)
    sigma ~ lognormal(0.0, 1.0)
    {:y} ~ normal.(slope .* xs, sigma)
end

# The pre-#156 per-chain draw, kept verbatim as the reference: a full traced
# `generate` (walking every observation address), slot readback through the
# signature layout, and the signature-aware transform, with the signature plan
# re-resolved per chain.
function bip_reference_signature_initial_parameters(model, args, resolved, constraints, rng)
    trace, _ = generate(model, args, constraints; rng=rng)
    layout = resolved.plan.parameter_layout
    params = Vector{Float64}(undef, UncertainTea.parametervaluecount(layout))
    for slot in layout.slots
        UncertainTea._write_slot_value!(params, slot, trace[UncertainTea._static_address(slot.address)])
    end
    return params
end

function bip_reference_batched_positions(model, args, constraints, rng, num_params, num_chains)
    positions = Matrix{Float64}(undef, num_params, num_chains)
    seeds = rand(rng, UInt, num_chains)
    for chain_index = 1:num_chains
        chain_rng = MersenneTwister(seeds[chain_index])
        chain_constraints = constraints isa AbstractVector ? constraints[chain_index] : constraints
        resolved = UncertainTea._resolve_signature_plan(model, chain_constraints, ())
        constrained =
            bip_reference_signature_initial_parameters(model, args, resolved, chain_constraints, chain_rng)
        positions[:, chain_index] = transform_to_unconstrained(model, constrained, args, chain_constraints)
    end
    return positions
end

@testset "batched initial positions skip observation tracing (issue #156)" begin
    bip_data_rng = MersenneTwister(99)
    bip_n = 12
    bip_gauss_constraints = choicemap([(:y => i) => randn(bip_data_rng) for i = 1:bip_n]...)
    bip_sigmas = fill(1.5, 8)
    bip_schools_constraints = choicemap([(:y => j) => randn(bip_data_rng) for j = 1:8]...)
    bip_xs = collect(0.0:0.25:2.0)
    bip_broadcast_constraints = choicemap(:y => randn(bip_data_rng, length(bip_xs)))

    bip_cases = [
        ("gauss loop obs", bip_gauss_model, (bip_n,), bip_gauss_constraints),
        ("eight-schools iid noncentered", bip_schools_model, (8, bip_sigmas), bip_schools_constraints),
        ("broadcast normal", bip_broadcast_model, (bip_xs,), bip_broadcast_constraints),
    ]

    for (bip_label, bip_model, bip_args, bip_constraints) in bip_cases
        @testset "$bip_label" begin
            bip_resolved = UncertainTea._resolve_signature_plan(bip_model, bip_constraints, ())
            bip_num_params = UncertainTea.parametercount(bip_resolved.plan.parameter_layout)
            bip_constrained_num_params = UncertainTea.parametervaluecount(bip_resolved.plan.parameter_layout)
            for bip_num_chains in (4, 64)
                bip_expected = bip_reference_batched_positions(
                    bip_model,
                    bip_args,
                    bip_constraints,
                    MersenneTwister(1234),
                    bip_num_params,
                    bip_num_chains,
                )
                bip_actual = UncertainTea._initial_batched_hmc_positions(
                    bip_model,
                    bip_args,
                    bip_constraints,
                    nothing,
                    MersenneTwister(1234),
                    bip_num_params,
                    bip_constrained_num_params,
                    bip_num_chains,
                )
                # bitwise identity, not ≈: the draw values themselves must not change
                @test bip_actual == bip_expected
            end
        end
    end

    @testset "per-chain constraint vectors share one resolution" begin
        bip_num_chains = 4
        bip_chain_constraints = [
            choicemap([(:y => i) => randn(MersenneTwister(100 + c)) for i = 1:bip_n]...) for c = 1:bip_num_chains
        ]
        bip_resolved = UncertainTea._resolve_signature_plan(bip_gauss_model, bip_chain_constraints[1], ())
        bip_num_params = UncertainTea.parametercount(bip_resolved.plan.parameter_layout)
        bip_expected = bip_reference_batched_positions(
            bip_gauss_model,
            (bip_n,),
            bip_chain_constraints,
            MersenneTwister(77),
            bip_num_params,
            bip_num_chains,
        )
        bip_actual = UncertainTea._initial_batched_hmc_positions(
            bip_gauss_model,
            (bip_n,),
            bip_chain_constraints,
            nothing,
            MersenneTwister(77),
            bip_num_params,
            UncertainTea.parametervaluecount(bip_resolved.plan.parameter_layout),
            bip_num_chains,
        )
        @test bip_actual == bip_expected
    end

    @testset "single-chain prior draw matches the traced path" begin
        bip_resolved = UncertainTea._resolve_signature_plan(bip_gauss_model, bip_gauss_constraints, ())
        bip_expected = bip_reference_signature_initial_parameters(
            bip_gauss_model,
            (bip_n,),
            bip_resolved,
            bip_gauss_constraints,
            MersenneTwister(4321),
        )
        bip_actual = UncertainTea._signature_initial_parameters(
            bip_gauss_model,
            (bip_n,),
            bip_resolved,
            bip_gauss_constraints;
            rng=MersenneTwister(4321),
        )
        @test bip_actual == bip_expected
    end
end
