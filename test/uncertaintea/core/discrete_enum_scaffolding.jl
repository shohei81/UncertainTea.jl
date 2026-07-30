# PR-2 scaffolding for marginalize=:enumerate (docs/discrete-enumeration.md,
# issue #13): the frontend parses and validates the flag, the IR carries it,
# generate keeps forward-sampling the latent, and every execution path that
# cannot marginalize yet rejects it honestly. The logsumexp semantics land in
# PR-3 (compiled CPU) and PR-4/5 (backend).

@tea static function denum_bernoulli_model()
    p ~ beta(2.0f0, 2.0f0)
    z ~ bernoulli(p; marginalize=:enumerate)
    {:y} ~ normal(z * 2.0f0, 1.0f0)
    return z
end

@tea static function denum_categorical_model()
    z ~ categorical([0.2f0, 0.3f0, 0.5f0]; marginalize=:enumerate)
    {:y} ~ normal(1.0f0 * z, 1.0f0)
    return z
end

# an explicit :none flag must behave exactly like no flag
@tea static function denum_none_model()
    z ~ bernoulli(0.5f0; marginalize=:none)
    {:y} ~ normal(z * 1.0f0, 1.0f0)
    return z
end

@testset "discrete_enum_scaffolding" begin
    @testset "denum_spec_and_layout" begin
        spec = modelspec(denum_bernoulli_model)
        z_choice = spec.choices[findfirst(c -> c.binding === :z, spec.choices)]
        @test z_choice.rhs isa DistributionSpec
        @test z_choice.rhs.marginalize === :enumerate
        @test z_choice.rhs.arguments == Any[:p]

        cat_spec = modelspec(denum_categorical_model)
        cat_choice = cat_spec.choices[findfirst(c -> c.binding === :z, cat_spec.choices)]
        @test cat_choice.rhs.marginalize === :enumerate

        none_spec = modelspec(denum_none_model)
        none_choice = none_spec.choices[findfirst(c -> c.binding === :z, none_spec.choices)]
        @test none_choice.rhs.marginalize === :none

        # a marginalized latent has no parameter slot (it is summed out, not
        # sampled): only the continuous beta latent counts
        @test parametercount(parameterlayout(denum_bernoulli_model)) == 1
        @test parametercount(parameterlayout(denum_categorical_model)) == 0
    end

    @testset "denum_runtime_forward_sampling" begin
        # the keyword is stripped from the runtime body, so generate still
        # forward-samples the latent into the trace
        trace, logw = generate(denum_bernoulli_model, (), choicemap((:y, 0.4f0)); rng=MersenneTwister(31))
        @test haskey(trace.choices, :z)
        @test trace[:z] isa Bool
        @test isfinite(logw)
        cat_trace, _ = generate(denum_categorical_model, (), choicemap((:y, 1.4f0)); rng=MersenneTwister(32))
        @test cat_trace[:z] in 1:3

        # conditioning on the discrete latent scores the plain joint, exactly
        # like an unflagged slotless choice
        z_constraints = choicemap((:z, true), (:y, 0.4f0))
        params = parameter_vector(trace)
        @test logjoint(denum_bernoulli_model, params, (), z_constraints) ≈
              assess(
            denum_bernoulli_model,
            (),
            choicemap((:p, trace[:p]), (:z, true), (:y, 0.4f0)),
        ) atol = 1e-6
    end

    @testset "denum_honest_rejections" begin
        # PR-3 landed the suffix logsumexp: an unconstrained flagged latent
        # scores the finite marginal (semantics tested in discrete_enum_cpu.jl)
        trace, _ = generate(denum_bernoulli_model, (), choicemap((:y, 0.4f0)); rng=MersenneTwister(33))
        @test isfinite(
            logjoint(denum_bernoulli_model, parameter_vector(trace), (), choicemap((:y, 0.4f0))),
        )

        # PR-4 landed the suffix-owning backend step and PR-5 its analytic
        # gradient; issue #67 landed the device mirror, so the step now lowers to
        # the device plan (parity in device_marginalize_lowering.jl)
        denum_report = backend_report(denum_bernoulli_model)
        @test denum_report.supported == true
        denum_plan = backend_execution_plan(denum_bernoulli_model)
        @test denum_plan.steps[end] isa UncertainTea.BackendMarginalizeChoicePlanStep
        @test UncertainTea._backend_gradient_supported(denum_plan)
        denum_device_supported, denum_device_issues = device_lowering_report(denum_bernoulli_model)
        @test denum_device_supported
        @test isempty(denum_device_issues)

        # the noncentered dependency audit must keep walking the suffix the
        # marginalize step owns: a noncentered location depending on the
        # enumerated binding is rejected by the CPU transform, so the backend
        # must reject it too
        @eval @tea static function denum_enum_noncentered_model()
            tau ~ lognormal(0.0, 0.5)
            z ~ bernoulli(0.5; marginalize=:enumerate)
            x ~ normal(z * 1.0, tau; reparam=:noncentered)
            {:y} ~ normal(x, 0.5)
            return x
        end
        denum_nc_report = backend_report(denum_enum_noncentered_model)
        @test denum_nc_report.supported == false
        @test any(issue -> occursin("noncentered", issue), denum_nc_report.issues)
    end

    @testset "denum_macro_time_errors" begin
        # unknown marginalize value
        @test_throws LoadError @eval @tea static function denum_bad_value_model()
            z ~ bernoulli(0.5f0; marginalize=:exhaustive)
            return z
        end
        # non-literal value
        @test_throws LoadError @eval @tea static function denum_dynamic_value_model(flag)
            z ~ bernoulli(0.5f0; marginalize=flag)
            return z
        end
        # ineligible family
        @test_throws LoadError @eval @tea static function denum_bad_family_model()
            z ~ normal(0.0f0, 1.0f0; marginalize=:enumerate)
            return z
        end
        # categorical needs a macro-time literal probability container
        @test_throws LoadError @eval @tea static function denum_dynamic_probs_model(ps)
            z ~ categorical(ps; marginalize=:enumerate)
            return z
        end
        # loop-scoped marginalized choices are rejected (suffix semantics do
        # not extend into loop bodies in v1)
        @test_throws LoadError @eval @tea static function denum_loop_model(n)
            for i = 1:n
                {:z => i} ~ bernoulli(0.5f0; marginalize=:enumerate)
            end
            return n
        end
        # nested occurrences diverge runtime body from spec, like reparam=
        @test_throws LoadError @eval @tea static function denum_nested_model()
            x ~ mixture((0.5f0, 0.5f0), normal(0.0f0, 1.0f0; marginalize=:enumerate), normal(2.0f0, 1.0f0))
            return x
        end
        # iid is not an eligible callee
        @test_throws LoadError @eval @tea static function denum_iid_model()
            z ~ iid(bernoulli(0.5f0), 3; marginalize=:enumerate)
            return z
        end
        # a tuple probability container would strip to categorical((...)),
        # which the runtime constructor rejects — vector literals only
        @test_throws LoadError @eval @tea static function denum_tuple_probs_model()
            z ~ categorical((0.2f0, 0.8f0); marginalize=:enumerate)
            return z
        end
        # generative (submodel) calls never supported keywords: the spec would
        # drop them while the runtime body still passes them to the callee, so
        # even the no-op values are macro-time errors
        @test_throws LoadError @eval @tea static function denum_submodel_kw_model()
            z = ({:sub} ~ denum_bernoulli_model(; marginalize=:none))
            return z
        end
        @test_throws LoadError @eval @tea static function denum_submodel_reparam_model()
            z = ({:sub} ~ denum_bernoulli_model(; reparam=:centered))
            return z
        end
    end

    @testset "denum_inlined_loop_rejection" begin
        # a child model with an enumerated latent called from a parent loop
        # would bypass the macro-time loop-scope check via inlining; the plan
        # build re-validates after inlining (definition-time ArgumentError)
        @eval @tea static function denum_enum_child()
            z ~ bernoulli(0.5f0; marginalize=:enumerate)
            return z
        end
        @test_throws ArgumentError @eval @tea static function denum_loop_parent(n)
            for i = 1:n
                z = ({:child => i} ~ denum_enum_child())
            end
            return n
        end
    end
end
