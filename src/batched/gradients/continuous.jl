# Hand-derived analytic batched logjoint gradients: continuous scalar families.
#
# The per-family `_accumulate_<family>_gradient!` loops and the per-family
# `_score_backend_step_and_gradient!` wrappers are @eval-generated from
# DISTRIBUTION_FAMILY_TABLE (issue #331 stage 2) on top of the issue-#285
# single-source kernels `_backend_<family>_logpdf(params..., value)` /
# `_<family>_logpdf_partials(params..., value)` with standardized
# `(dvalue, dparams...)` channel order. Families whose bodies deviate from the
# common template stay hand-written below with a comment saying why (weibull's
# issue-#86 boundary scale channel, uniform's value-channel-free partials, the
# noncentered-normal z-space step).

# Off-support value with a live derivative seed (issue #343): a latent flowing
# through a saturating transform can land EXACTLY on the support boundary
# (sigmoid(theta) rounds to 1.0 for theta >~ 36.74, exp(theta) underflows to
# 0.0 below ~-745). The logpdf contribution is then -Inf, but the accumulate
# loops used to skip the partials entirely, leaving only the finite transform
# Jacobian term in the unconstrained gradient -- a silently wrong FINITE
# gradient that leapfrog gradient guards never reject. Whenever the partials
# are skipped for a value carrying derivative information, poison the affected
# gradient rows with NaN so the -Inf logjoint and the gradient reject
# together. Observed off-support values have an all-zero value-gradient seed
# and keep the previous skip semantics (finite parameter partials dropped,
# whole evaluation already scored -Inf).
function _poison_offsupport_value_gradient!(
    gradients::AbstractMatrix{T},
    value_gradients::AbstractMatrix{T},
    batch_index::Integer,
) where {T<:AbstractFloat}
    for parameter_index in axes(gradients, 1)
        iszero(value_gradients[parameter_index, batch_index]) ||
            (gradients[parameter_index, batch_index] = T(NaN))
    end
    return nothing
end

# ---- table-generated accumulate loops (issue #331 stage 2) --------------------
#
# One loop body per family: score the kernel into the totals, guard the support
# (poisoning per issue #343 when the table carries an `offsupport` predicate),
# then chain the analytic partials through the per-channel gradient planes.
# Weibull and uniform deviate and stay hand-written below.
for family in (
    :normal,
    :lognormal,
    :exponential,
    :gamma,
    :inversegamma,
    :beta,
    :studentt,
    :laplace,
    :cauchy,
    :halfnormal,
    :halfcauchy,
    :logistic,
    :gumbel,
    :pareto,
    :frechet,
    :rayleigh,
    :inversegaussian,
)
    spec = _distribution_family_spec(family)
    parameters = spec.params
    accumulate_name = Symbol("_accumulate_", family, "_gradient!")
    kernel_name = Symbol("_backend_", family, "_logpdf")
    partials_name = Symbol("_", family, "_logpdf_partials")
    values_names = [Symbol(parameter, "_values") for parameter in parameters]
    gradients_names = [Symbol(parameter, "_gradients") for parameter in parameters]
    partial_names = [Symbol("d", parameter) for parameter in parameters]
    signature = Any[]
    for (values_name, gradients_name) in zip(values_names, gradients_names)
        push!(signature, :($values_name::AbstractVector{T}))
        push!(signature, :($gradients_name::AbstractMatrix{T}))
    end
    parameter_loads =
        [:($parameter = $values_name[batch_index]) for (parameter, values_name) in zip(parameters, values_names)]
    guard = if isnothing(spec.offsupport)
        nothing
    else
        quote
            if $(spec.offsupport)
                _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
                continue
            end
        end
    end
    accumulation = Expr(
        :call,
        :+,
        :(dvalue * value_gradients[parameter_index, batch_index]),
        (
            :($partial_name * $gradients_name[parameter_index, batch_index]) for
            (partial_name, gradients_name) in zip(partial_names, gradients_names)
        )...,
    )
    @eval function $accumulate_name(
        totals::AbstractVector{T},
        gradients::AbstractMatrix{T},
        value_values::AbstractVector{T},
        value_gradients::AbstractMatrix{T},
        $(signature...),
    ) where {T<:AbstractFloat}
        for batch_index in eachindex(totals)
            value = value_values[batch_index]
            $(parameter_loads...)
            totals[batch_index] += $kernel_name($(parameters...), value)
            $guard
            (dvalue, $(partial_names...)) = $partials_name($(parameters...), value)
            for parameter_index in axes(gradients, 1)
                gradients[parameter_index, batch_index] += $accumulation
            end
        end
        return totals, gradients
    end
end

# ---- hand-written accumulate deviations ---------------------------------------

# Weibull stays hand-written: at the x == 0, shape == 1 boundary the density is
# finite (-log(scale)) with a live scale channel (issue #86), which the common
# poison-and-continue guard cannot express.
function _accumulate_weibull_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    shape_values::AbstractVector{T},
    shape_gradients::AbstractMatrix{T},
    scale_values::AbstractVector{T},
    scale_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        shape = shape_values[batch_index]
        scale = scale_values[batch_index]
        totals[batch_index] += _backend_weibull_logpdf(shape, scale, value)
        if !(value > 0)
            # At the x == 0, shape == 1 boundary the logpdf is -log(scale)
            # (see logpdf(::WeibullDist, x)), so the scale partial is finite
            # while the value/shape channels stay zero as in the CPU
            # ForwardDiff path (issue #86).
            if value == 0 && shape == 1
                dscale_boundary = -1 / scale
                for parameter_index in axes(gradients, 1)
                    gradients[parameter_index, batch_index] +=
                        dscale_boundary * scale_gradients[parameter_index, batch_index]
                end
            else
                # non-finite boundary/off-support score: reject via the
                # gradient too (issue #343)
                _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            end
            continue
        end
        dvalue, dshape, dscale = _weibull_logpdf_partials(shape, scale, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dvalue * value_gradients[parameter_index, batch_index] +
                dshape * shape_gradients[parameter_index, batch_index] +
                dscale * scale_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

# Uniform stays hand-written: `_uniform_logpdf_partials` returns only the bound
# channels (d/dvalue is 0 on the open interval), deviating from the standard
# `(dvalue, dparams...)` order the generated loops assume.
function _accumulate_uniform_gradient!(
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    value_values::AbstractVector{T},
    value_gradients::AbstractMatrix{T},
    lower_values::AbstractVector{T},
    lower_gradients::AbstractMatrix{T},
    upper_values::AbstractVector{T},
    upper_gradients::AbstractMatrix{T},
) where {T<:AbstractFloat}
    for batch_index in eachindex(totals)
        value = value_values[batch_index]
        lower = lower_values[batch_index]
        upper = upper_values[batch_index]
        totals[batch_index] += _backend_uniform_logpdf(lower, upper, value)
        if !(lower <= value <= upper)
            _poison_offsupport_value_gradient!(gradients, value_gradients, batch_index)
            continue
        end
        # d/dvalue is 0 on the open interval; the bound partials are the only
        # nonzero channels (relevant only for dynamic-bound observations).
        dlower, dupper = _uniform_logpdf_partials(lower, upper, value)
        for parameter_index in axes(gradients, 1)
            gradients[parameter_index, batch_index] +=
                dlower * lower_gradients[parameter_index, batch_index] +
                dupper * upper_gradients[parameter_index, batch_index]
        end
    end
    return totals, gradients
end

# ---- table-generated per-family gradient step wrappers (issue #331 stage 2) ----
#
# The wrapper template is uniform even for the families whose ACCUMULATE loop
# deviates (weibull, uniform): stage the choice values/gradient seed, evaluate
# each parameter expression into a scratch values/gradients pair, delegate to
# the family's accumulate loop, then assign the binding slot. Scratch layout
# matches the historical hand-written bodies exactly: parameter k uses numeric
# scratch k and gradient scratch k+1 (the choice gradient holds scratch 1), and
# the expression evaluations start at base index nparams + 2.
for family in (
    :normal,
    :lognormal,
    :exponential,
    :gamma,
    :inversegamma,
    :weibull,
    :beta,
    :studentt,
    :laplace,
    :cauchy,
    :halfnormal,
    :halfcauchy,
    :uniform,
    :logistic,
    :gumbel,
    :pareto,
    :frechet,
    :rayleigh,
    :inversegaussian,
)
    spec = _distribution_family_spec(family)
    parameters = spec.params
    parameter_count = length(parameters)
    step_type = Symbol("Backend", spec.step, "ChoicePlanStep")
    accumulate_name = Symbol("_accumulate_", family, "_gradient!")
    scratch = Any[]
    evaluations = Any[]
    accumulate_arguments = Any[:totals, :gradients, :value_values, :value_gradients]
    for (position, parameter) in enumerate(parameters)
        values_name = Symbol(parameter, "_values")
        gradients_name = Symbol(parameter, "_gradients")
        push!(scratch, :($values_name = _batched_numeric_scratch!(env, $position)))
        push!(scratch, :($gradients_name = _batched_backend_gradient_scratch!(cache, $(position + 1))))
        push!(
            evaluations,
            :(_eval_backend_numeric_expr_and_gradient!(
                $values_name,
                $gradients_name,
                cache,
                env,
                step.$parameter,
                $(parameter_count + 1 + position),
            )),
        )
        push!(accumulate_arguments, values_name, gradients_name)
    end
    @eval function _score_backend_step_and_gradient!(
        step::$step_type,
        totals::AbstractVector{T},
        gradients::AbstractMatrix{T},
        cache::BatchedBackendGradientCache,
        env::BatchedPlanEnvironment{T},
        params::AbstractMatrix{T},
        constraints,
    ) where {T<:AbstractFloat}
        value_values = env.observed_values
        value_gradients = _batched_backend_gradient_scratch!(cache, 1)
        $(scratch...)
        address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

        _batched_choice_numeric_values!(value_values, step.parameter_slot, params, constraints, address_parts)
        _fill_choice_gradient!(value_gradients, step.parameter_slot, cache.seed_rows)
        $(evaluations...)
        $accumulate_name($(accumulate_arguments...))
        isnothing(step.binding_slot) ||
            _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, value_values, value_gradients)
        return totals, gradients
    end
end

# Noncentered normal stays hand-written: it scores the STANDARD normal on the
# z-space choice and materializes theta = mu + sigma * z into the binding slot,
# which the score-then-accumulate template cannot express.
function _score_backend_step_and_gradient!(
    step::BackendNoncenteredNormalChoicePlanStep,
    totals::AbstractVector{T},
    gradients::AbstractMatrix{T},
    cache::BatchedBackendGradientCache,
    env::BatchedPlanEnvironment{T},
    params::AbstractMatrix{T},
    constraints,
) where {T<:AbstractFloat}
    z_values = env.observed_values
    z_gradients = _batched_backend_gradient_scratch!(cache, 1)
    mu_values = _batched_numeric_scratch!(env, 1)
    mu_gradients = _batched_backend_gradient_scratch!(cache, 2)
    sigma_values = _batched_numeric_scratch!(env, 2)
    sigma_gradients = _batched_backend_gradient_scratch!(cache, 3)
    theta_values = _batched_numeric_scratch!(env, 3)
    theta_gradients = _batched_backend_gradient_scratch!(cache, 4)
    address_parts = _batched_backend_address_parts(env, step.address.parts, 1)

    _batched_choice_numeric_values!(z_values, step.parameter_slot, params, constraints, address_parts)
    _fill_choice_gradient!(z_gradients, step.parameter_slot, cache.seed_rows)
    _eval_backend_numeric_expr_and_gradient!(mu_values, mu_gradients, cache, env, step.mu, 5)
    _eval_backend_numeric_expr_and_gradient!(sigma_values, sigma_gradients, cache, env, step.sigma, 6)
    for batch_index in eachindex(totals)
        z = z_values[batch_index]
        sigma = sigma_values[batch_index]
        (isfinite(sigma) && sigma > 0) || throw(
            BatchedBackendFallback("noncentered normal requires a finite positive scale, got $sigma"),
        )
        totals[batch_index] += _backend_normal_logpdf(zero(z), one(z), z)
        theta_values[batch_index] = mu_values[batch_index] + sigma * z
    end
    # d logpdf(N(0,1), z)/dz = -z through the slot seed of z
    for batch_index in eachindex(totals), parameter_index in axes(gradients, 1)
        z_grad = z_gradients[parameter_index, batch_index]
        iszero(z_grad) || (gradients[parameter_index, batch_index] -= z_values[batch_index] * z_grad)
        theta_gradients[parameter_index, batch_index] =
            mu_gradients[parameter_index, batch_index] +
            z_values[batch_index] * sigma_gradients[parameter_index, batch_index] +
            sigma_values[batch_index] * z_grad
    end
    isnothing(step.binding_slot) ||
        _assign_backend_choice_value!(env, cache.slot_gradients, step.binding_slot, theta_values, theta_gradients)
    return totals, gradients
end
