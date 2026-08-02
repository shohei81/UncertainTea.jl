function _static_address(address::AddressSpec)
    parts = Any[]
    for part in address.parts
        part isa AddressLiteralPart || throw(ArgumentError("parameter vectors require static addresses, got $address"))
        push!(parts, part.value)
    end
    return Tuple(parts)
end

to_constrained(::IdentityTransform, value) = value
to_unconstrained(::IdentityTransform, value) = value
to_constrained(::VectorIdentityTransform, value::AbstractVector) = value
to_unconstrained(::VectorIdentityTransform, value::AbstractVector) = value
to_constrained(::LogTransform, value) = exp(value)
to_unconstrained(::LogTransform, value) = log(value)
to_constrained(::LogitTransform, value) = inv(one(value) + exp(-value))
to_unconstrained(::LogitTransform, value) = log(value) - log1p(-value)
logabsdetjac(::IdentityTransform, value) = zero(value)
logabsdetjac(::VectorIdentityTransform, value::AbstractVector) = isempty(value) ? 0.0 : zero(value[firstindex(value)])
logabsdetjac(::LogTransform, value) = value

# Exact log|d sigmoid(u)/du| = log(sigmoid(u)) + log(sigmoid(-u)) in the
# saturation-free form -|u| - 2*log1p(exp(-|u|)). Symmetric in u and finite for
# every u, where the naive log(c) + log1p(-c) collapses to -Inf once
# c = sigmoid(u) rounds to exactly 1 (u ~ 36.7 in Float64, ~16.6 in Float32).
function _logit_logabsdetjac(value)
    magnitude = abs(float(value))
    return -magnitude - 2 * log1p(exp(-magnitude))
end

logabsdetjac(::LogitTransform, value) = _logit_logabsdetjac(value)

to_constrained(::VectorLogTransform, value::AbstractVector) = map(exp, value)
to_unconstrained(::VectorLogTransform, value::AbstractVector) = map(log, value)
function logabsdetjac(::VectorLogTransform, value::AbstractVector)
    isempty(value) && return 0.0
    total = zero(value[firstindex(value)])
    for element in value
        total += element
    end
    return total
end

to_constrained(::VectorLogitTransform, value::AbstractVector) = map(v -> inv(one(v) + exp(-v)), value)
to_unconstrained(::VectorLogitTransform, value::AbstractVector) = map(v -> log(v) - log1p(-v), value)
function logabsdetjac(::VectorLogitTransform, value::AbstractVector)
    isempty(value) && return 0.0
    total = zero(float(value[firstindex(value)]))
    for element in value
        total += _logit_logabsdetjac(element)
    end
    return total
end

_bounded_sigmoid(value) = inv(one(value) + exp(-value))
to_constrained(transform::BoundedTransform, value) =
    transform.lower + (transform.upper - transform.lower) * _bounded_sigmoid(value)
function to_unconstrained(transform::BoundedTransform, value)
    scaled = (value - transform.lower) / (transform.upper - transform.lower)
    return log(scaled) - log1p(-scaled)
end
function logabsdetjac(transform::BoundedTransform, value)
    return log(transform.upper - transform.lower) + _logit_logabsdetjac(value)
end

to_constrained(transform::LowerBoundedTransform, value) = transform.lower + exp(value)
to_unconstrained(transform::LowerBoundedTransform, value) = log(value - transform.lower)
logabsdetjac(::LowerBoundedTransform, value) = value

to_constrained(transform::UpperBoundedTransform, value) = transform.upper - exp(value)
to_unconstrained(transform::UpperBoundedTransform, value) = log(transform.upper - value)
logabsdetjac(::UpperBoundedTransform, value) = value

function _simplex_logabsdet(constrained::AbstractVector)
    total = zero(eltype(constrained))
    for value in constrained
        total += log(value)
    end
    return total
end

function _to_constrained_simplex!(
    destination::AbstractVector,
    transform::SimplexTransform,
    values::AbstractVector,
)
    length(values) == transform.size - 1 ||
        throw(DimensionMismatch("expected $(transform.size - 1) unconstrained simplex values, got $(length(values))"))
    length(destination) == transform.size ||
        throw(DimensionMismatch("expected simplex destination of length $(transform.size), got $(length(destination))"))

    max_value = zero(promote_type(eltype(destination), eltype(values)))
    for value in values
        max_value = max(max_value, value)
    end

    total = exp(-max_value)
    for index in eachindex(values)
        shifted = exp(values[index] - max_value)
        destination[index] = shifted
        total += shifted
    end
    destination[end] = exp(-max_value)
    for index in eachindex(destination)
        destination[index] /= total
    end
    return destination
end

function _to_unconstrained_simplex!(
    destination::AbstractVector,
    transform::SimplexTransform,
    values::AbstractVector,
)
    length(values) == transform.size ||
        throw(DimensionMismatch("expected simplex values of length $(transform.size), got $(length(values))"))
    length(destination) == transform.size - 1 ||
        throw(
            DimensionMismatch(
                "expected simplex unconstrained destination of length $(transform.size - 1), got $(length(destination))",
            ),
        )

    last_value = values[end]
    total = zero(last_value)
    for index in eachindex(values)
        value = values[index]
        value > zero(value) || throw(
            ArgumentError(
                "simplex values must be strictly positive; component $(index - firstindex(values) + 1) " *
                "of $(transform.size) is $value (small-concentration Dirichlet draws can underflow to zero)",
            ),
        )
        total += value
    end
    abs(total - one(total)) <= sqrt(eps(float(total))) * transform.size * 16 ||
        throw(ArgumentError("simplex values must sum to 1"))

    log_last = log(last_value)
    for index in eachindex(destination)
        destination[index] = log(values[index]) - log_last
    end
    return destination
end

function to_constrained(transform::SimplexTransform, value::AbstractVector)
    destination = similar(collect(value), transform.size)
    return _to_constrained_simplex!(destination, transform, value)
end

function to_unconstrained(transform::SimplexTransform, value::AbstractVector)
    destination = similar(collect(value), transform.size - 1)
    return _to_unconstrained_simplex!(destination, transform, value)
end

function logabsdetjac(transform::SimplexTransform, value::AbstractVector)
    constrained = to_constrained(transform, value)
    return _simplex_logabsdet(constrained)
end

# Numerically stable log(1 - tanh(z)^2) = log(sech(z)^2); symmetric in z, so
# the |z| form never overflows exp for large magnitudes.
function _log1m_tanh_sq(z)
    zf = float(z)
    magnitude = abs(zf)
    return 2 * (log(oftype(magnitude, 2)) - magnitude - log1p(exp(-2 * magnitude)))
end

# Stan's `cholesky_corr_constrain`: map a d*(d-1)/2 unconstrained vector to the
# column-major PACKED lower-triangular Cholesky factor of a correlation matrix
# (length d*(d+1)/2, diagonal included). Unconstrained entries are consumed in
# row-major below-diagonal order ((2,1), (3,1), (3,2), ...): row i uses canonical
# partial correlations w_j = tanh(z_j), sets L[i,j] = w_j * sqrt(1 - sum_{k<j}
# L[i,k]^2), and closes the row with L[i,i] = sqrt(1 - sum_{k<i} L[i,k]^2) so
# every row is a unit vector.
#
# Returns the log-abs-det of the Jacobian of the *below-diagonal* map
# z -> (L[i,j])_{i>j}. That Jacobian is triangular in the consumption order, so
# its log-determinant is the sum over below-diagonal entries of
#   log(1 - tanh(z)^2) + 0.5 * log(1 - sum_{k<j} L[i,k]^2),
# the tanh derivative times the sqrt scaling factor. The diagonal entries are
# determined by the rows' unit norms and are not free coordinates.
function _to_constrained_cholesky_corr!(
    destination::AbstractVector,
    transform::CholeskyCorrTransform,
    values::AbstractVector,
)
    d = transform.size
    length(values) == (d * (d - 1)) ÷ 2 ||
        throw(DimensionMismatch("expected $((d * (d - 1)) ÷ 2) unconstrained cholesky correlation values, got $(length(values))"))
    length(destination) == (d * (d + 1)) ÷ 2 ||
        throw(
            DimensionMismatch(
                "expected packed cholesky correlation destination of length $((d * (d + 1)) ÷ 2), got $(length(destination))",
            ),
        )

    first_value = float(values[firstindex(values)])
    logabsdet = zero(first_value)
    destination[_packed_lower_index(d, 1, 1)] = one(first_value)
    z_position = firstindex(values)
    for row = 2:d
        sum_sqs = zero(first_value)
        for col = 1:(row-1)
            z = values[z_position]
            z_position += 1
            w = tanh(z)
            # tanh saturation at |z| >~ 19 rounds w to exactly +-1, which can
            # push sum_sqs to 1 (or one ulp above); log1p(-sum_sqs) and
            # sqrt(1 - sum_sqs) would then throw a DomainError. Unconstrained
            # space is all of R^n, so match the exception-free device contract
            # (_device_cholesky_corr_constrain): degrade the log-abs-det to
            # -Inf (rejection) and clamp the entry to zero (issue #99). The
            # healthy path is bit-for-bit unchanged.
            saturated = sum_sqs >= one(sum_sqs)
            logabsdet +=
                _log1m_tanh_sq(z) +
                (saturated ? oftype(sum_sqs, -Inf) : log1p(-sum_sqs) / 2)
            entry = saturated ? zero(sum_sqs) : w * sqrt(1 - sum_sqs)
            destination[_packed_lower_index(d, row, col)] = entry
            sum_sqs += entry * entry
        end
        destination[_packed_lower_index(d, row, row)] =
            sum_sqs >= one(sum_sqs) ? zero(sum_sqs) : sqrt(1 - sum_sqs)
    end
    return logabsdet
end

# Inverse of `_to_constrained_cholesky_corr!`: recover w_j = L[i,j] /
# sqrt(1 - sum_{k<j} L[i,k]^2) and z_j = atanh(w_j) per below-diagonal entry.
function _to_unconstrained_cholesky_corr!(
    destination::AbstractVector,
    transform::CholeskyCorrTransform,
    values::AbstractVector,
)
    d = transform.size
    length(values) == (d * (d + 1)) ÷ 2 ||
        throw(
            DimensionMismatch("expected packed cholesky correlation values of length $((d * (d + 1)) ÷ 2), got $(length(values))"),
        )
    length(destination) == (d * (d - 1)) ÷ 2 ||
        throw(
            DimensionMismatch(
                "expected cholesky correlation unconstrained destination of length $((d * (d - 1)) ÷ 2), got $(length(destination))",
            ),
        )

    for row = 1:d
        diagonal = values[_packed_lower_index(d, row, row)]
        diagonal > zero(diagonal) ||
            throw(ArgumentError("cholesky correlation values require strictly positive diagonal entries"))
    end

    z_position = firstindex(destination)
    for row = 2:d
        sum_sqs = zero(float(values[firstindex(values)]))
        for col = 1:(row-1)
            entry = values[_packed_lower_index(d, row, col)]
            remaining = 1 - sum_sqs
            remaining > zero(remaining) ||
                throw(ArgumentError("cholesky correlation rows must have norm at most 1"))
            w = entry / sqrt(remaining)
            -1 < w < 1 ||
                throw(ArgumentError("cholesky correlation rows must have norm at most 1"))
            destination[z_position] = atanh(w)
            z_position += 1
            sum_sqs += entry * entry
        end
    end
    return destination
end

function to_constrained(transform::CholeskyCorrTransform, value::AbstractVector)
    destination = similar(collect(value), (transform.size * (transform.size + 1)) ÷ 2)
    _to_constrained_cholesky_corr!(destination, transform, value)
    return destination
end

function to_unconstrained(transform::CholeskyCorrTransform, value::AbstractVector)
    destination = similar(collect(value), (transform.size * (transform.size - 1)) ÷ 2)
    return _to_unconstrained_cholesky_corr!(destination, transform, value)
end

function logabsdetjac(transform::CholeskyCorrTransform, value::AbstractVector)
    destination = similar(collect(value), (transform.size * (transform.size + 1)) ÷ 2)
    return _to_constrained_cholesky_corr!(destination, transform, value)
end

# log-Cholesky covariance transform. The packed layout is column-major with the
# diagonal included; the diagonal packed coordinates carry log(L[i,i]) in the
# unconstrained space (exp on constrain) while the below-diagonal coordinates are
# passed through unchanged. The log-abs-det of unconstrained -> L is the sum of
# the diagonal unconstrained coordinates (= sum_i log(L[i,i])).
function _to_constrained_cholesky_cov!(
    destination::AbstractVector,
    transform::CholeskyCovTransform,
    values::AbstractVector,
)
    d = transform.size
    expected = (d * (d + 1)) ÷ 2
    length(values) == expected ||
        throw(DimensionMismatch("expected $expected unconstrained cholesky covariance values, got $(length(values))"))
    length(destination) == expected ||
        throw(DimensionMismatch("expected packed cholesky covariance destination of length $expected, got $(length(destination))"))
    logabsdet = zero(float(values[firstindex(values)]))
    for col = 1:d, row = col:d
        position = _packed_lower_index(d, row, col)
        unconstrained = values[position]
        if row == col
            destination[position] = exp(unconstrained)
            logabsdet += unconstrained
        else
            destination[position] = unconstrained
        end
    end
    return logabsdet
end

function _to_unconstrained_cholesky_cov!(
    destination::AbstractVector,
    transform::CholeskyCovTransform,
    values::AbstractVector,
)
    d = transform.size
    expected = (d * (d + 1)) ÷ 2
    length(values) == expected ||
        throw(DimensionMismatch("expected packed cholesky covariance values of length $expected, got $(length(values))"))
    length(destination) == expected ||
        throw(
            DimensionMismatch(
                "expected cholesky covariance unconstrained destination of length $expected, got $(length(destination))",
            ),
        )
    for col = 1:d, row = col:d
        position = _packed_lower_index(d, row, col)
        if row == col
            diagonal = values[position]
            diagonal > zero(diagonal) ||
                throw(ArgumentError("cholesky covariance values require strictly positive diagonal entries"))
            destination[position] = log(diagonal)
        else
            destination[position] = values[position]
        end
    end
    return destination
end

function to_constrained(transform::CholeskyCovTransform, value::AbstractVector)
    destination = similar(collect(value), (transform.size * (transform.size + 1)) ÷ 2)
    _to_constrained_cholesky_cov!(destination, transform, value)
    return destination
end

function to_unconstrained(transform::CholeskyCovTransform, value::AbstractVector)
    destination = similar(collect(value), (transform.size * (transform.size + 1)) ÷ 2)
    return _to_unconstrained_cholesky_cov!(destination, transform, value)
end

function logabsdetjac(transform::CholeskyCovTransform, value::AbstractVector)
    destination = similar(collect(value), (transform.size * (transform.size + 1)) ÷ 2)
    return _to_constrained_cholesky_cov!(destination, transform, value)
end

function _slot_parameter_values(params::AbstractVector, slot::ParameterSlotSpec)
    indices = parametervalueindices(slot)
    if slot.value_length == 1
        return params[first(indices)]
    end
    return collect(view(params, indices))
end

function _write_slot_value!(destination::AbstractVector, slot::ParameterSlotSpec, value)
    indices = parametervalueindices(slot)
    if slot.value_length == 1
        destination[first(indices)] = value
        return destination
    end

    if value isa AbstractVector || value isa Tuple
        length(value) == slot.value_length ||
            throw(DimensionMismatch("expected $(slot.value_length) values for choice $(slot.binding), got $(length(value))"))
        for (index, item) in zip(indices, value)
            destination[index] = Float64(item)
        end
        return destination
    end

    throw(ArgumentError("parameter slot $(slot.binding) expected a vector-like value, got $(typeof(value))"))
end

function _transform_slot_to_constrained!(
    destination::AbstractVector,
    slot::ParameterSlotSpec,
    params::AbstractVector,
)
    (slot.transform isa NoncenteredTransform || slot.transform isa VectorNoncenteredTransform) && throw(
        ErrorException(
            "reparam=:noncentered is accepted by the frontend but its dependent transforms " *
            "are not implemented yet (docs/noncentered-reparam.md, PR-3)",
        ),
    )
    if slot.transform isa IdentityTransform
        destination[slot.value_index] = params[slot.index]
        return zero(params[slot.index])
    elseif slot.transform isa VectorIdentityTransform
        copyto!(view(destination, parametervalueindices(slot)), view(params, parameterindices(slot)))
        return zero(params[first(parameterindices(slot))])
    elseif slot.transform isa VectorLogTransform
        logabsdet = zero(params[first(parameterindices(slot))])
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            unconstrained_value = params[parameter_index]
            destination[value_index] = exp(unconstrained_value)
            logabsdet += unconstrained_value
        end
        return logabsdet
    elseif slot.transform isa VectorLogitTransform
        logabsdet = zero(params[first(parameterindices(slot))])
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            unconstrained_value = params[parameter_index]
            constrained_value = _bounded_sigmoid(unconstrained_value)
            destination[value_index] = constrained_value
            logabsdet += _logit_logabsdetjac(unconstrained_value)
        end
        return logabsdet
    elseif slot.transform isa LogTransform
        unconstrained_value = params[slot.index]
        destination[slot.value_index] = exp(unconstrained_value)
        return unconstrained_value
    elseif slot.transform isa LogitTransform
        unconstrained_value = params[slot.index]
        constrained_value = to_constrained(slot.transform, unconstrained_value)
        destination[slot.value_index] = constrained_value
        return logabsdetjac(slot.transform, unconstrained_value)
    elseif slot.transform isa BoundedTransform ||
           slot.transform isa LowerBoundedTransform ||
           slot.transform isa UpperBoundedTransform
        unconstrained_value = params[slot.index]
        constrained_value = to_constrained(slot.transform, unconstrained_value)
        destination[slot.value_index] = constrained_value
        return logabsdetjac(slot.transform, unconstrained_value)
    elseif slot.transform isa SimplexTransform
        unconstrained = view(params, parameterindices(slot))
        constrained = view(destination, parametervalueindices(slot))
        _to_constrained_simplex!(constrained, slot.transform, unconstrained)
        return _simplex_logabsdet(constrained)
    elseif slot.transform isa CholeskyCorrTransform
        unconstrained = view(params, parameterindices(slot))
        constrained = view(destination, parametervalueindices(slot))
        return _to_constrained_cholesky_corr!(constrained, slot.transform, unconstrained)
    elseif slot.transform isa CholeskyCovTransform
        unconstrained = view(params, parameterindices(slot))
        constrained = view(destination, parametervalueindices(slot))
        return _to_constrained_cholesky_cov!(constrained, slot.transform, unconstrained)
    end

    throw(ArgumentError("unsupported parameter transform $(typeof(slot.transform))"))
end

function _transform_slot_to_unconstrained!(
    destination::AbstractVector,
    slot::ParameterSlotSpec,
    params::AbstractVector,
)
    (slot.transform isa NoncenteredTransform || slot.transform isa VectorNoncenteredTransform) && throw(
        ErrorException(
            "reparam=:noncentered is accepted by the frontend but its dependent transforms " *
            "are not implemented yet (docs/noncentered-reparam.md, PR-3)",
        ),
    )
    if slot.transform isa IdentityTransform
        destination[slot.index] = params[slot.value_index]
    elseif slot.transform isa VectorIdentityTransform
        copyto!(view(destination, parameterindices(slot)), view(params, parametervalueindices(slot)))
    elseif slot.transform isa VectorLogTransform
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            destination[parameter_index] = log(params[value_index])
        end
    elseif slot.transform isa VectorLogitTransform
        for (parameter_index, value_index) in zip(parameterindices(slot), parametervalueindices(slot))
            constrained_value = params[value_index]
            destination[parameter_index] = log(constrained_value) - log1p(-constrained_value)
        end
    elseif slot.transform isa LogTransform
        destination[slot.index] = log(params[slot.value_index])
    elseif slot.transform isa LogitTransform
        destination[slot.index] = to_unconstrained(slot.transform, params[slot.value_index])
    elseif slot.transform isa BoundedTransform ||
           slot.transform isa LowerBoundedTransform ||
           slot.transform isa UpperBoundedTransform
        destination[slot.index] = to_unconstrained(slot.transform, params[slot.value_index])
    elseif slot.transform isa SimplexTransform
        unconstrained = view(destination, parameterindices(slot))
        constrained = view(params, parametervalueindices(slot))
        _to_unconstrained_simplex!(unconstrained, slot.transform, constrained)
    elseif slot.transform isa CholeskyCorrTransform
        unconstrained = view(destination, parameterindices(slot))
        constrained = view(params, parametervalueindices(slot))
        _to_unconstrained_cholesky_corr!(unconstrained, slot.transform, constrained)
    elseif slot.transform isa CholeskyCovTransform
        unconstrained = view(destination, parameterindices(slot))
        constrained = view(params, parametervalueindices(slot))
        _to_unconstrained_cholesky_cov!(unconstrained, slot.transform, constrained)
    else
        throw(ArgumentError("unsupported parameter transform $(typeof(slot.transform))"))
    end
    return destination
end

function parameterchoicemap(model::TeaModel, params::AbstractVector)
    return _parameterchoicemap(parameterlayout(model), params)
end

# Signature-aware variant (#95 PR-6): the latent slots follow the conditioning
# signature, so a constrained-space vector produced for `(model, constraints)`
# maps back to exactly the conditioned latents' addresses.
function parameterchoicemap(model::TeaModel, params::AbstractVector, constraints::ChoiceMap)
    return _parameterchoicemap(_conditioned_parameter_layout(model, constraints), params)
end

function _parameterchoicemap(layout::ParameterLayout, params::AbstractVector)
    expected = parametervaluecount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    cm = ChoiceMap()
    for slot in layout.slots
        _pushchoice!(cm, _static_address(slot.address), _slot_parameter_values(params, slot))
    end
    return cm
end

function parameter_vector(trace::TeaTrace)
    layout = parameterlayout(trace.model)
    params = Vector{Float64}(undef, parametervaluecount(layout))
    for slot in layout.slots
        _write_slot_value!(params, slot, trace[_static_address(slot.address)])
    end
    return params
end

_has_dependent_transforms(layout::ParameterLayout) = any(
    slot.transform isa NoncenteredTransform || slot.transform isa VectorNoncenteredTransform for
    slot in layout.slots
)

function transform_to_constrained(model::TeaModel, params::AbstractVector, args::Tuple=())
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    constrained = similar(params, parametervaluecount(layout))
    _transform_to_constrained!(constrained, model, params, args)
    return constrained
end

function _transform_to_constrained!(
    destination::AbstractVector,
    model::TeaModel,
    params::AbstractVector,
    args::Tuple=(),
)
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))
    constrained_expected = parametervaluecount(layout)
    length(destination) == constrained_expected ||
        throw(DimensionMismatch("expected constrained destination of length $constrained_expected, got $(length(destination))"))

    if _has_dependent_transforms(layout)
        _dependent_transform_walk!(destination, model, params, args, false)
        return destination
    end
    for slot in layout.slots
        _transform_slot_to_constrained!(destination, slot, params)
    end
    return destination
end

# Signature-aware transform: the constrained/unconstrained split and the
# expected vector length follow the conditioning signature (which addresses are
# observed), so a length mismatch names the conditioning. The three-argument
# forms above keep operating on the model's default (unconditioned) layout.
function transform_to_constrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    resolved = _resolve_signature_plan(model, constraints)
    constrained, _ = _transform_to_constrained_with_logabsdet(model, resolved, params, args, constraints)
    return constrained
end

function transform_to_constrained_with_logabsdet(model::TeaModel, params::AbstractVector, args::Tuple=())
    layout = parameterlayout(model)
    expected = parametercount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    constrained = similar(params, parametervaluecount(layout))
    if _has_dependent_transforms(layout)
        logabsdet = _dependent_transform_walk!(constrained, model, params, args, false)
        return constrained, logabsdet
    end
    logabsdet = expected == 0 ? 0.0 : zero(params[firstindex(params)])
    for slot in layout.slots
        logabsdet += _transform_slot_to_constrained!(constrained, slot, params)
    end
    return constrained, logabsdet
end

function transform_to_unconstrained(model::TeaModel, params::AbstractVector, args::Tuple=())
    layout = parameterlayout(model)
    expected = parametervaluecount(layout)
    length(params) == expected || throw(DimensionMismatch("expected $expected parameters, got $(length(params))"))

    unconstrained = similar(params, parametercount(layout))
    if _has_dependent_transforms(layout)
        _dependent_transform_walk!(unconstrained, model, params, args, true)
        return unconstrained
    end
    for slot in layout.slots
        _transform_slot_to_unconstrained!(unconstrained, slot, params)
    end
    return unconstrained
end

function transform_to_unconstrained(
    model::TeaModel,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    resolved = _resolve_signature_plan(model, constraints)
    return _transform_to_unconstrained(model, resolved, params, args, constraints)
end

# Resolved-plan core of the signature-aware transform: callers that already
# hold the `ResolvedSignaturePlan` (the batched initial-position loop resolves
# it once per batch, issue #156) skip re-deriving the conditioning signature.
# `resolved` is untyped because parameters.jl loads before evaluator.jl.
function _transform_to_unconstrained(
    model::TeaModel,
    resolved,
    params::AbstractVector,
    args::Tuple,
    constraints::ChoiceMap,
)
    layout = resolved.plan.parameter_layout
    expected = parametervaluecount(layout)
    length(params) == expected || throw(
        _signature_length_error(
            model,
            layout,
            constraints,
            expected,
            length(params);
            space="constrained-space parameters",
        ),
    )

    unconstrained = similar(params, parametercount(layout))
    if _has_dependent_transforms(layout)
        _dependent_transform_walk!(
            unconstrained,
            model,
            resolved.plan,
            resolved.compiled,
            params,
            args,
            true,
            constraints,
        )
        return unconstrained
    end
    for slot in layout.slots
        _transform_slot_to_unconstrained!(unconstrained, slot, params)
    end
    return unconstrained
end

transform_to_unconstrained(trace::TeaTrace) =
    transform_to_unconstrained(trace.model, parameter_vector(trace), trace.args)

function initialparameters(model::TeaModel, args::Tuple=(); rng::AbstractRNG=Random.default_rng())
    trace, _ = generate(model, args, choicemap(); rng=rng)
    return parameter_vector(trace)
end
