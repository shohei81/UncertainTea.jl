# Every file below is wrapped in its own named @testset and depends only on
# ../fixtures.jl, so any file can be run standalone (include fixtures.jl first)
# and the include order is free.
#
# UNCERTAINTEA_TEST_GROUP selects a CI shard (.github/workflows/ci.yml); the
# default "all" runs everything, so a plain local `Pkg.test()` is unchanged.
# Groups are balanced by measured wall-clock time; put a new file in the
# topically matching group and rebalance only if one group grows far beyond
# the others. sampling.jl belongs to the "sampling" group (see runtests.jl).
include("fixtures.jl")

core_test_files = [
    # (file, group)
    ("dsl_static_model_semantics.jl", "dsl"),
    ("constraint_driven_conditioning.jl", "dsl"),
    ("dsl_contract_guards.jl", "dsl"),
    ("batched_logjoint_and_gradient.jl", "dsl"),
    ("interpreter_obs_staging.jl", "dsl"),
    ("static_vector_obs_staging.jl", "dsl"),
    ("hmc_and_nuts_workspace.jl", "dsl"),
    ("nuts_scheduler_and_backend.jl", "dsl"),
    ("tuple_and_loop_addresses.jl", "dsl"),
    ("custom_distribution_registration.jl", "dsl"),
    ("reparam_scaffolding.jl", "dsl"),
    ("discrete_enum_scaffolding.jl", "dsl"),
    ("dist_exponential_poisson.jl", "dist"),
    ("dist_gamma_studentt.jl", "dist"),
    ("dist_beta_categorical.jl", "dist"),
    ("dist_inversegamma_weibull_binomial.jl", "dist"),
    ("dist_laplace_geometric_negbinom.jl", "dist"),
    ("dist_scalar_priors.jl", "dist"),
    ("scalar_kernel_partials.jl", "dist"),
    ("dist_positive_heavytail.jl", "dist"),
    ("dist_dirichlet.jl", "dist"),
    ("dist_mvnormal_diag.jl", "dist"),
    ("dist_truncated.jl", "dist"),
    ("dist_mixture.jl", "dist"),
    ("dist_mvnormal_dense.jl", "dist"),
    ("dist_multivariate.jl", "dist"),
    ("dist_lkj_cholesky.jl", "dist"),
    ("dist_gaussian_process.jl", "dist"),
    ("dist_sparse_gaussian_process.jl", "dist"),
    ("dist_hidden_markov.jl", "dist"),
    ("dist_gp_latent.jl", "dist"),
    ("transform_logit_saturation.jl", "dist"),
    ("dist_integer_params.jl", "dist"),
    ("dist_count_logfactorial.jl", "dist"),
    ("dist_bernoulli.jl", "dist"),
    ("dist_bernoulli_logit.jl", "dist"),
    ("dist_discrete_gaps.jl", "dist"),
    # "backend": CPU/backend batched scoring and vector-latent paths.
    ("vector_backend_sampler.jl", "backend"),
    ("batched_scoring_eltype_f32.jl", "backend"),
    ("vectorized_obs_iid_latents.jl", "backend"),
    ("backend_native_families.jl", "backend"),
    ("backend_glm_logistic.jl", "backend"),
    ("batched_observed_loop_gradient.jl", "backend"),
    ("batched_adtype.jl", "backend"),
    ("backend_trig_primitives.jl", "backend"),
    ("broadcast_scalar_families.jl", "backend"),
    ("batched_observed_loop_suffstats.jl", "backend"),
    ("threaded_batched_gradient.jl", "backend"),
    # "device": the KernelAbstractions device kernels (heavier compile).
    ("device_rng.jl", "device"),
    ("device_lowering_parity.jl", "device"),
    ("device_kernel_host_parity.jl", "device"),
    ("device_marginalize_lowering.jl", "device"),
    ("signature_batched_device_parity.jl", "device"),
    ("device_gradient_dual.jl", "device"),
    ("device_tiled_post_loop_obs.jl", "device"),
    ("device_hmc_advi.jl", "device"),
    ("device_masked_nuts.jl", "device"),
    ("device_persistent_nuts.jl", "device"),
    ("device_persistent_nuts_tiled.jl", "device"),
    ("device_masked_nuts_compaction.jl", "device"),
    ("device_glm_lowering.jl", "device"),
    ("device_broadcast_normal_lowering.jl", "device"),
    ("device_perchain_stranding.jl", "device"),
    ("host_device_pooled_agreement.jl", "device"),
    ("device_chees.jl", "device"),
    ("plan_build_nospecialize.jl", "backend"),
    ("gradient_crosscheck.jl", "crosscheck"),
    ("generated_scorer_identity.jl", "crosscheck"),
    ("generated_scorer_suffstats.jl", "crosscheck"),
    ("sampler_value_path_obs_cache.jl", "crosscheck"),
    # "inference": diagnostics, VI, predictive, and lighter-weight sampler checks.
    ("batched_advi_particle.jl", "inference"),
    ("batched_svgd.jl", "inference"),
    ("proposal_diagnostics_overflow.jl", "inference"),
    ("integrator_nuts_proposal.jl", "inference"),
    ("nuts_biased_merge_workspace_reuse.jl", "inference"),
    ("nuts_uturn_turning.jl", "inference"),
    ("mcmc_diagnostics_ess_mcse.jl", "inference"),
    ("predictive_sampling_smc_resampling.jl", "inference"),
    ("nested_sampling.jl", "inference"),
    ("waic_psis_loo.jl", "inference"),
    ("pointwise_marginal_enum.jl", "inference"),
    ("map_laplace_approximation.jl", "inference"),
    ("advi_structured_guides.jl", "inference"),
    ("advi_flow_iwae.jl", "inference"),
    ("pathfinder_init.jl", "inference"),
    ("pathfinder_mass_seed.jl", "inference"),
    ("batched_initial_positions.jl", "inference"),
    # "sampling": the wall-clock-heavy MCMC draws (many samples/chains). Split out
    # of "inference" so the two shards run in balanced parallel; sampling.jl
    # (see runtests.jl) also belongs here.
    ("tempered_batched_smc.jl", "sampling"),
    ("nuts_fixed_step_moments.jl", "sampling"),
    ("warmup_driver_regression.jl", "sampling"),
    ("per_chain_warmup_batched.jl", "sampling"),
    ("reject_invalid_parameters.jl", "sampling"),
    ("dense_mass_matrix_single_chain.jl", "sampling"),
    ("masked_batched_nuts.jl", "sampling"),
    ("masked_lane_compaction.jl", "sampling"),
    ("init_robustness_batched.jl", "sampling"),
    ("allocfree_leapfrog_bitwise.jl", "sampling"),
    ("sbc_calibration.jl", "sampling"),
    ("reparam_noncentered_cpu.jl", "sampling"),
    ("noncentered_reject_hmc.jl", "sampling"),
    ("discrete_enum_cpu.jl", "sampling"),
    ("mh_within_gibbs.jl", "sampling"),
    ("multichain_threaded_reproducibility.jl", "sampling"),
    ("batched_chees_scaffold.jl", "sampling"),
    ("batched_chees_adaptation.jl", "sampling"),
    ("batched_meads.jl", "sampling"),
]

let registered = Set(first.(core_test_files)), on_disk = Set(f for f in readdir(joinpath(@__DIR__, "core")) if endswith(f, ".jl"))

    unregistered = sort!(collect(setdiff(on_disk, registered)))
    isempty(unregistered) ||
        error("Test files not registered in core_test_files (so they would never run): $unregistered")
    missing_files = sort!(collect(setdiff(registered, on_disk)))
    isempty(missing_files) ||
        error("core_test_files entries with no file on disk: $missing_files")
end

test_group = get(ENV, "UNCERTAINTEA_TEST_GROUP", "all")
known_test_groups = ("all", "dsl", "dist", "backend", "device", "inference", "sampling", "crosscheck")
test_group in known_test_groups ||
    error("Unknown UNCERTAINTEA_TEST_GROUP=\"$test_group\"; expected one of $known_test_groups")

for (file, group) in core_test_files
    if test_group == "all" || test_group == group
        include("core/$file")
    end
end
