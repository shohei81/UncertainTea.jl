function _qualify(name::Symbol)
    return GlobalRef(@__MODULE__, name)
end

function _argument_name(arg)
    if arg isa Symbol
        return arg
    elseif arg isa Expr && arg.head == :(::)
        return _argument_name(arg.args[1])
    elseif arg isa Expr && arg.head == :kw
        return _argument_name(arg.args[1])
    end

    throw(ArgumentError("@tea could not determine the argument name for $arg"))
end

function _tea_mode(mode_expr)
    if mode_expr === :static
        return _qualify(:STATIC_MODE)
    elseif mode_expr === :dynamic
        return _qualify(:DYNAMIC_MODE)
    else
        throw(ArgumentError("@tea only supports `static` or `dynamic` modes"))
    end
end

function _qualify_builtin_distribution(name)
    if name in (
        :normal,
        :lognormal,
        :laplace,
        :exponential,
        :gamma,
        :inversegamma,
        :weibull,
        :beta,
        :dirichlet,
        :mvnormal,
        :mvnormaldense,
        :lkjcholesky,
        :bernoulli,
        :bernoullilogit,
        :binomial,
        :geometric,
        :negativebinomial,
        :poisson,
        :studentt,
        :categorical,
        :truncatednormal,
        :truncatedstudentt,
        :mixture,
        :iid,
        :cauchy,
        :halfnormal,
        :halfcauchy,
        :uniform,
        :logistic,
        :gumbel,
    )
        return _qualify(name)
    end
    # Registered user families: splice the builder VALUE captured at macro
    # expansion, so the runtime body works regardless of what the builder is
    # called in the user's module and later re-registration cannot change an
    # already-defined model.
    registration = _registered_user_distribution(name)
    isnothing(registration) || return registration.builder
    return name
end

# Distribution families that may be dot-called as broadcast observations, e.g.
# `{:y} ~ normal.(mu, sigma)`. Any known distribution family other than these is
# rejected at macro time.
const _BROADCAST_DISTRIBUTION_FAMILIES = (:normal,)

const _KNOWN_DISTRIBUTION_FAMILIES = (
    :normal, :lognormal, :laplace, :exponential, :gamma, :inversegamma, :weibull,
    :beta, :dirichlet, :mvnormal, :mvnormaldense, :lkjcholesky, :bernoulli, :bernoullilogit, :binomial, :geometric,
    :negativebinomial, :poisson, :studentt, :categorical, :truncatednormal, :truncatedstudentt, :mixture,
    :cauchy, :halfnormal, :halfcauchy, :uniform, :logistic, :gumbel,
)

# Detects a dot-call distribution observation `family.(args...)` on the RHS of `~`.
# Returns the runtime broadcast-distribution construction expression, or `nothing`
# when `rhs` is not a distribution dot-call. Throws for unsupported dot-called families.
function _rewrite_broadcast_rhs(rhs, ctxsym)
    (rhs isa Expr && rhs.head == :. && length(rhs.args) == 2) || return nothing
    family = rhs.args[1]
    argtuple = rhs.args[2]
    (family isa Symbol && argtuple isa Expr && argtuple.head == :tuple) || return nothing
    family in _KNOWN_DISTRIBUTION_FAMILIES || return nothing
    family in _BROADCAST_DISTRIBUTION_FAMILIES || throw(ArgumentError(
        "broadcast (dot-call) observations currently support only `normal.(...)`, got `$(family).(...)`",
    ))
    arguments = Any[_rewrite_tea_expr(arg, ctxsym) for arg in argtuple.args]
    return Expr(:call, _qualify(:BroadcastNormalDist), arguments...)
end

function _rewrite_tea_expr(expr, ctxsym)
    if !(expr isa Expr)
        return expr
    end

    if expr.head == :call && !isempty(expr.args) && expr.args[1] === :~
        lhs = expr.args[2]
        broadcast_rhs = _rewrite_broadcast_rhs(expr.args[3], ctxsym)
        rhs = isnothing(broadcast_rhs) ? _rewrite_tea_expr(expr.args[3], ctxsym) : broadcast_rhs

        if lhs isa Symbol
            return Expr(:(=), lhs, Expr(:call, _qualify(:choice), ctxsym, QuoteNode(lhs), rhs))
        elseif lhs isa Expr && lhs.head == :braces && length(lhs.args) == 1
            return Expr(:call, _qualify(:choice), ctxsym, lhs.args[1], rhs)
        else
            throw(ArgumentError("unsupported choice syntax in @tea: $lhs"))
        end
    end

    rewritten_args = Any[_rewrite_tea_expr(arg, ctxsym) for arg in expr.args]
    if expr.head == :call && !isempty(rewritten_args) && rewritten_args[1] isa Symbol
        qualified = _qualify_builtin_distribution(rewritten_args[1])
        if qualified !== rewritten_args[1]
            # runtime distribution calls stay centered: the reparam flag lives
            # on the spec/plan only
            rewritten_args = _strip_reparam_arguments(rewritten_args)
        end
        rewritten_args[1] = qualified
    end
    return Expr(expr.head, rewritten_args...)
end

function _address_part_expr(value, literal::Bool)
    if literal
        return :($(_qualify(:AddressLiteralPart))($(QuoteNode(value))))
    end
    return :($(_qualify(:AddressDynamicPart))($(QuoteNode(value))))
end

function _append_address_parts!(parts, expr; symbol_literal::Bool=false)
    if expr isa QuoteNode
        push!(parts, _address_part_expr(expr.value, true))
        return parts
    elseif expr isa Symbol
        push!(parts, _address_part_expr(expr, symbol_literal))
        return parts
    elseif expr isa Expr && expr.head == :call && !isempty(expr.args) && expr.args[1] === :(=>)
        _append_address_parts!(parts, expr.args[2]; symbol_literal=symbol_literal)
        _append_address_parts!(parts, expr.args[3]; symbol_literal=symbol_literal)
        return parts
    elseif expr isa Expr && expr.head == :tuple
        for element in expr.args
            _append_address_parts!(parts, element; symbol_literal=symbol_literal)
        end
        return parts
    elseif expr isa Number || expr isa String || expr isa Char
        push!(parts, _address_part_expr(expr, true))
        return parts
    end

    push!(parts, _address_part_expr(expr, false))
    return parts
end

function _address_spec_expr(lhs)
    parts = Expr[]
    if lhs isa Symbol
        _append_address_parts!(parts, lhs; symbol_literal=true)
    elseif lhs isa Expr && lhs.head == :braces && length(lhs.args) == 1
        _append_address_parts!(parts, lhs.args[1]; symbol_literal=false)
    else
        throw(ArgumentError("unsupported choice syntax in @tea: $lhs"))
    end

    tuple_expr = Expr(:tuple, parts...)
    return :($(_qualify(:AddressSpec))($tuple_expr))
end

function _choice_spec_expr(expr)
    return _choice_spec_expr(expr, (), nothing)
end

function _choice_spec_expr(expr, loop_scopes, binding_override=nothing)
    expr isa Expr && expr.head == :call && expr.args[1] === :~ ||
        throw(ArgumentError("expected a choice expression, got $expr"))

    lhs = expr.args[2]
    rhs = expr.args[3]
    binding_symbol = isnothing(binding_override) ? (lhs isa Symbol ? lhs : nothing) : binding_override
    binding = isnothing(binding_symbol) ? :(nothing) : QuoteNode(binding_symbol)
    address = _address_spec_expr(lhs)
    rhs_spec = _rhs_spec_expr(rhs)
    scopes_expr = Expr(:vect, map(_loop_scope_spec_expr, loop_scopes)...)
    return :($(_qualify(:ChoiceSpec))($binding, $address, $rhs_spec, $scopes_expr))
end

const _REPARAM_ELIGIBLE_FAMILIES = (:normal, :studentt, :laplace, :lognormal)

_reparam_location_scale_positions(family::Symbol) = family === :studentt ? (2, 3) : (1, 2)

# reparam=:auto resolves at macro expansion: noncentered when the location or
# scale argument is a non-literal expression (it may reference other latents,
# which is the funnel-geometry case; a plain model-argument reference also
# resolves noncentered, which is harmless -- both parameterizations are exact).
function _resolve_auto_reparam(family::Symbol, positional)
    location_index, scale_index = _reparam_location_scale_positions(family)
    length(positional) >= scale_index || return :centered
    location = positional[location_index]
    scale = positional[scale_index]
    return (location isa Number && scale isa Number) ? :centered : :noncentered
end

# The flags are only meaningful on the top-level distribution of a `~`; nested
# occurrences (e.g. inside a mixture component) would otherwise diverge
# between the runtime body (keyword stripped) and the static spec.
function _reject_nested_reparam(expr)
    expr isa Expr || return nothing
    if expr.head == :kw && expr.args[1] in (:reparam, :marginalize)
        throw(
            ArgumentError(
                "$(expr.args[1])= is only supported on the top-level distribution call of `~`, found it nested in `$expr`",
            ),
        )
    end
    foreach(_reject_nested_reparam, expr.args)
    return nothing
end

# Strip `reparam=`/`marginalize=` keywords from a distribution-call RHS and
# return the normalized call plus the flags, so macro-time consumers (the
# parameter layout pre-pass) see the same positional arguments as the
# emitted spec.
function _normalized_rhs_call(rhs)
    (rhs isa Expr && rhs.head == :call && !isempty(rhs.args)) || return rhs, :centered, :none
    positional, reparam, marginalize = _split_rhs_keywords(rhs)
    return Expr(:call, rhs.args[1], positional...), reparam, marginalize
end

# Finite-support discrete families eligible for marginalize=:enumerate
# (docs/discrete-enumeration.md). categorical additionally needs a macro-time
# literal probability container so the support size is compile-time.
const _MARGINALIZE_ELIGIBLE_FAMILIES = (:bernoulli, :categorical)

# Split the keyword arguments off a `~` distribution call. Only
# `reparam=:centered|:noncentered|:auto` and `marginalize=:enumerate|:none`
# are recognized; any other keyword is a macro-time error (previously
# keywords were silently spliced into the positional argument list as a
# malformed `Expr(:parameters, ...)`).
function _split_rhs_keywords(rhs::Expr)
    callee = rhs.args[1]
    positional = Any[]
    reparam = :centered
    marginalize = :none
    handle_kw =
        kw -> begin
            (kw isa Expr && kw.head == :kw && kw.args[1] in (:reparam, :marginalize)) || throw(
                ArgumentError(  # also fires for generative (submodel) calls, which never supported keywords
                    "unsupported keyword argument in `~` distribution call `$rhs`; only `reparam=` and `marginalize=` are recognized",
                ),
            )
            value = kw.args[2]
            if kw.args[1] === :reparam
                (value isa QuoteNode && value.value in (:centered, :noncentered, :auto)) || throw(
                    ArgumentError(
                        "reparam must be the literal :centered, :noncentered, or :auto, got `$(value)` in `$rhs`",
                    ),
                )
                reparam = value.value
            else
                (value isa QuoteNode && value.value in (:enumerate, :none)) || throw(
                    ArgumentError(
                        "marginalize must be the literal :enumerate or :none, got `$(value)` in `$rhs`",
                    ),
                )
                marginalize = value.value
            end
            return nothing
        end
    for arg in rhs.args[2:end]
        if arg isa Expr && arg.head == :parameters
            foreach(handle_kw, arg.args)
        elseif arg isa Expr && arg.head == :kw
            handle_kw(arg)
        else
            _reject_nested_reparam(arg)
            push!(positional, arg)
        end
    end
    if reparam !== :centered
        eligible =
            callee in _REPARAM_ELIGIBLE_FAMILIES ||
            (callee === :iid && _iid_reparam_base_eligible(positional))
        eligible || throw(
            ArgumentError(
                "reparam=$(QuoteNode(reparam)) supports the location-scale families " *
                "$(_REPARAM_ELIGIBLE_FAMILIES) (directly or as an iid base), got `$callee`",
            ),
        )
    end
    if marginalize === :enumerate
        callee in _MARGINALIZE_ELIGIBLE_FAMILIES || throw(
            ArgumentError(
                "marginalize=:enumerate supports the finite-support discrete families " *
                "$(_MARGINALIZE_ELIGIBLE_FAMILIES), got `$callee`",
            ),
        )
        if callee === :categorical
            # vector literals only: the runtime categorical constructor does
            # not accept a single tuple argument, and the support size must be
            # known at macro time
            (length(positional) == 1 && positional[1] isa Expr && positional[1].head == :vect) || throw(
                ArgumentError(
                    "marginalize=:enumerate requires a literal probability vector for " *
                    "categorical (the support size must be known at macro time), got `$rhs`",
                ),
            )
        end
    end
    if reparam === :auto
        if callee === :iid
            base = positional[1]
            reparam = _resolve_auto_reparam(base.args[1], base.args[2:end])
        else
            reparam = _resolve_auto_reparam(callee, positional)
        end
    end
    return positional, reparam, marginalize
end

function _iid_reparam_base_eligible(positional)
    isempty(positional) && return false
    base = positional[1]
    base isa Expr && base.head == :call && !isempty(base.args) || return false
    # log-space vector form is not implemented yet; real-line families only
    return base.args[1] in (:normal, :studentt, :laplace)
end

# Drop `reparam=`/`marginalize=` keywords from a distribution call in the
# rewritten runtime body: `generate`/`assess` semantics stay centered and
# forward-sampled, only the spec/plan carry the flags.
function _strip_reparam_arguments(arguments::Vector{Any})
    stripped = Any[]
    for arg in arguments
        if arg isa Expr && arg.head == :parameters
            kept = Any[
                kw for kw in arg.args if
                       !(kw isa Expr && kw.head == :kw && kw.args[1] in (:reparam, :marginalize))
            ]
            isempty(kept) || push!(stripped, Expr(:parameters, kept...))
        elseif arg isa Expr && arg.head == :kw && arg.args[1] in (:reparam, :marginalize)
            continue
        else
            push!(stripped, arg)
        end
    end
    return stripped
end

function _rhs_spec_expr(rhs)
    if rhs isa Expr && rhs.head == :. && length(rhs.args) == 2 &&
       rhs.args[1] isa Symbol && rhs.args[2] isa Expr && rhs.args[2].head == :tuple &&
       rhs.args[1] in _KNOWN_DISTRIBUTION_FAMILIES
        family = rhs.args[1]
        family in _BROADCAST_DISTRIBUTION_FAMILIES || throw(ArgumentError(
            "broadcast (dot-call) observations currently support only `normal.(...)`, got `$(family).(...)`",
        ))
        arguments = Expr(:vect, map(QuoteNode, rhs.args[2].args)...)
        return :($(_qualify(:BroadcastDistributionSpec))($(QuoteNode(family)), $arguments))
    end

    if rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] === :iid
        iid_positional, iid_reparam, _ = _split_rhs_keywords(rhs)
        length(iid_positional) == 2 ||
            throw(ArgumentError("iid expects `iid(distribution_call, n)`"))
        base = iid_positional[1]
        n = iid_positional[2]
        n isa Integer ||
            throw(ArgumentError("iid requires a literal Int count `n`, got `$(n)`"))
        (base isa Expr && base.head == :call && !isempty(base.args) && base.args[1] isa Symbol) ||
            throw(ArgumentError("iid first argument must be a distribution constructor call"))
        arguments = Expr(:vect, QuoteNode(base), QuoteNode(n))
        return :($(_qualify(:DistributionSpec))(
            $(QuoteNode(:iid)),
            $arguments,
            nothing,
            $(QuoteNode(iid_reparam)),
        ))
    end

    if rhs isa Expr && rhs.head == :call && !isempty(rhs.args)
        callee = rhs.args[1]
        positional, reparam, marginalize = _split_rhs_keywords(rhs)
        arguments = Expr(:vect, map(QuoteNode, positional)...)

        if callee === :lkjcholesky
            spec_arguments = positional
            (length(spec_arguments) == 2 && spec_arguments[1] isa Integer && spec_arguments[1] >= 2) ||
                throw(
                    ArgumentError(
                        "lkjcholesky requires a literal integer dimension `d >= 2` as its first argument " *
                        "(for latents and observations alike), got `$rhs`",
                    ),
                )
        end

        if callee isa Symbol && callee in BUILTIN_DISTRIBUTION_FAMILIES
            return :($(_qualify(:DistributionSpec))(
                $(QuoteNode(callee)),
                $arguments,
                nothing,
                $(QuoteNode(reparam)),
                $(QuoteNode(marginalize)),
            ))
        end
        if callee isa Symbol && haskey(USER_DISTRIBUTION_REGISTRY, callee)
            registration = USER_DISTRIBUTION_REGISTRY[callee]
            return :($(_qualify(:DistributionSpec))($(QuoteNode(callee)), $arguments, $(registration.builder)))
        end

        # generative (submodel) calls never supported keywords: the spec drops
        # them but the runtime body would still pass them to the callee, so
        # even the no-op values reparam=:centered / marginalize=:none must be
        # rejected here rather than diverge into a runtime MethodError
        any(arg -> arg isa Expr && arg.head in (:kw, :parameters), rhs.args) && throw(
            ArgumentError(
                "keyword arguments are not supported on generative (submodel) calls in `~`, got `$rhs`",
            ),
        )
        return :($(_qualify(:GenerativeCallSpec))($callee, $arguments))
    end

    return :($(_qualify(:RawChoiceRhsSpec))($(QuoteNode(rhs))))
end

function _loop_scope_spec_expr(scope)
    iterator, iterable = scope
    return :($(_qualify(:LoopScopeSpec))(
        $(QuoteNode(iterator)),
        $(QuoteNode(iterable)),
        $(_expr_has_dynamic_content(iterable)),
    ))
end

function _parameter_layout_expr(choice_nodes)
    slot_exprs = Expr[]
    slot_lookup = Dict{Int,Int}()
    slot_index = 1
    parameter_index = 1
    value_index = 1

    for (choice_index, node) in enumerate(choice_nodes)
        choice_expr, loop_scopes, binding_override = node
        lhs = choice_expr.args[2]
        rhs = choice_expr.args[3]
        binding_symbol = isnothing(binding_override) ? (lhs isa Symbol ? lhs : nothing) : binding_override

        rhs, reparam, marginalize = _normalized_rhs_call(rhs)
        if marginalize === :enumerate && !isempty(loop_scopes)
            throw(
                ArgumentError(
                    "marginalize=:enumerate is not supported on loop-scoped choices " *
                    "(the plan-suffix semantics do not extend into loop bodies; " *
                    "docs/discrete-enumeration.md), found `$lhs`",
                ),
            )
        end
        if !isnothing(binding_symbol) && isempty(loop_scopes) && _supports_parameter_slot(rhs)
            address = _address_spec_expr(lhs)
            transform = _parameter_transform_expr(rhs, reparam)
            parameter_dimension, value_length = _parameter_layout_sizes(rhs)
            push!(
                slot_exprs,
                :($(_qualify(:ParameterSlotSpec))(
                    $choice_index,
                    $(QuoteNode(binding_symbol)),
                    $address,
                    $parameter_index,
                    $parameter_dimension,
                    $value_index,
                    $value_length,
                    $transform,
                )),
            )
            slot_lookup[choice_index] = slot_index
            slot_index += 1
            parameter_index += parameter_dimension
            value_index += value_length
        end
    end

    return :($(_qualify(:ParameterLayout))($(Expr(:vect, slot_exprs...)), $(parameter_index - 1), $(value_index - 1))), slot_lookup
end

function _supported_distribution_family(rhs)
    rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] isa Symbol || return nothing
    family = rhs.args[1]
    family in (
        :normal, :lognormal, :laplace, :exponential, :gamma, :inversegamma, :weibull, :beta, :studentt,
        :cauchy, :halfnormal, :halfcauchy, :logistic, :gumbel,
    ) && return family
    if family === :uniform
        isnothing(_uniform_static_bounds(rhs.args[2:end])) && throw(
            ArgumentError(
                "uniform latents require literal (static) finite bounds; use static Number lower and upper bounds, " *
                "or provide the value as an observation for dynamic bounds",
            ),
        )
        return family
    end
    if family === :dirichlet && !isnothing(_dirichlet_static_size(rhs))
        return family
    end
    if family === :mvnormal && !isnothing(_mvnormal_static_size(rhs))
        return family
    end
    if family === :mvnormaldense && !isnothing(_mvnormaldense_static_size(rhs))
        return family
    end
    if family === :lkjcholesky
        isnothing(_lkjcholesky_static_dim(rhs)) && throw(ArgumentError(
            "lkjcholesky latents require a literal integer dimension `d >= 2` as the first argument",
        ))
        return family
    end
    if family === :truncatednormal || family === :truncatedstudentt
        isnothing(_truncated_static_bounds(family, rhs.args[2:end])) && throw(
            ArgumentError(
                "$family latents require literal (static) bounds; use static Number/Inf lower and upper bounds, " *
                "or provide the value as an observation for dynamic bounds",
            ),
        )
        return family
    end
    if family === :mixture
        _mixture_latent_eligible(rhs.args[2:end]) || throw(
            ArgumentError(
                "mixture latents require every component to be a real-line location-scale family " *
                "(normal, laplace, studentt); use the mixture as an observation for other component families",
            ),
        )
        return family
    end
    return nothing
end

function _supports_parameter_slot(rhs)
    return !isnothing(_supported_distribution_family(rhs))
end

function _dirichlet_static_size(rhs)
    rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] === :dirichlet || return nothing
    return _dirichlet_static_size(rhs.args[2:end])
end

function _mvnormal_static_size(rhs)
    rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] === :mvnormal || return nothing
    return _mvnormal_static_size(rhs.args[2:end])
end

function _mvnormaldense_static_size(rhs)
    rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] === :mvnormaldense || return nothing
    return _mvnormaldense_static_size(rhs.args[2:end])
end

function _lkjcholesky_static_dim(rhs)
    rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] === :lkjcholesky || return nothing
    return _lkjcholesky_static_size(rhs.args[2:end])
end

function _parameter_layout_sizes(rhs)
    family = _supported_distribution_family(rhs)
    isnothing(family) && throw(ArgumentError("unsupported parameter layout size for $rhs"))
    if family === :dirichlet
        size = _dirichlet_static_size(rhs)
        isnothing(size) && throw(ArgumentError("dirichlet parameter slots require a statically known simplex size"))
        return size - 1, size
    elseif family === :mvnormal
        size = _mvnormal_static_size(rhs)
        isnothing(size) && throw(ArgumentError("mvnormal parameter slots require a statically known vector size"))
        return size, size
    elseif family === :mvnormaldense
        size = _mvnormaldense_static_size(rhs)
        isnothing(size) && throw(ArgumentError("mvnormaldense parameter slots require a statically known mean vector size"))
        return size, size
    elseif family === :lkjcholesky
        size = _lkjcholesky_static_dim(rhs)
        isnothing(size) && throw(ArgumentError("lkjcholesky parameter slots require a literal integer dimension"))
        return (size * (size - 1)) ÷ 2, (size * (size + 1)) ÷ 2
    end
    return 1, 1
end

function _parameter_transform_expr(rhs, reparam::Symbol=:centered)
    if reparam === :noncentered
        if rhs isa Expr && rhs.head == :call && rhs.args[1] === :iid
            return :($(_qualify(:VectorNoncenteredTransform))($(rhs.args[3])))
        end
        return :($(_qualify(:NoncenteredTransform))())
    end
    family = _supported_distribution_family(rhs)
    isnothing(family) && throw(ArgumentError("unsupported parameter transform for $rhs"))

    if family === :normal || family === :laplace ||
       family === :cauchy || family === :logistic || family === :gumbel
        return :($(_qualify(:IdentityTransform))())
    elseif family === :halfnormal || family === :halfcauchy
        return :($(_qualify(:LogTransform))())
    elseif family === :uniform
        bounds = _uniform_static_bounds(rhs.args[2:end])
        isnothing(bounds) && throw(ArgumentError("uniform parameter slots require literal (static) finite bounds"))
        return :($(_qualify(:BoundedTransform))($(bounds[1]), $(bounds[2])))
    elseif family === :mvnormal
        size = _mvnormal_static_size(rhs)
        isnothing(size) && throw(ArgumentError("mvnormal parameter slots require a statically known vector size"))
        return :($(_qualify(:VectorIdentityTransform))($size))
    elseif family === :mvnormaldense
        size = _mvnormaldense_static_size(rhs)
        isnothing(size) && throw(ArgumentError("mvnormaldense parameter slots require a statically known mean vector size"))
        return :($(_qualify(:VectorIdentityTransform))($size))
    elseif family === :lkjcholesky
        size = _lkjcholesky_static_dim(rhs)
        isnothing(size) && throw(ArgumentError("lkjcholesky parameter slots require a literal integer dimension"))
        return :($(_qualify(:CholeskyCorrTransform))($size))
    elseif family === :lognormal || family === :exponential || family === :gamma ||
           family === :inversegamma || family === :weibull
        return :($(_qualify(:LogTransform))())
    elseif family === :beta
        return :($(_qualify(:LogitTransform))())
    elseif family === :dirichlet
        size = _dirichlet_static_size(rhs)
        isnothing(size) && throw(ArgumentError("dirichlet parameter slots require a statically known simplex size"))
        return :($(_qualify(:SimplexTransform))($size))
    elseif family === :studentt
        return :($(_qualify(:IdentityTransform))())
    elseif family === :truncatednormal || family === :truncatedstudentt
        bounds = _truncated_static_bounds(family, rhs.args[2:end])
        isnothing(bounds) && throw(ArgumentError("$family parameter slots require literal (static) bounds"))
        lower, upper = bounds
        if isfinite(lower) && isfinite(upper)
            return :($(_qualify(:BoundedTransform))($lower, $upper))
        elseif isfinite(lower)
            return :($(_qualify(:LowerBoundedTransform))($lower))
        elseif isfinite(upper)
            return :($(_qualify(:UpperBoundedTransform))($upper))
        end
        return :($(_qualify(:IdentityTransform))())
    elseif family === :mixture
        return :($(_qualify(:IdentityTransform))())
    end

    throw(ArgumentError("unsupported parameter transform family $family"))
end

function _address_has_dynamic_parts(lhs)
    if lhs isa Symbol
        return false
    elseif lhs isa Expr && lhs.head == :braces && length(lhs.args) == 1
        return _address_expr_has_dynamic_parts(lhs.args[1])
    end

    throw(ArgumentError("unsupported choice syntax in @tea: $lhs"))
end

function _address_expr_has_dynamic_parts(expr)
    if expr isa QuoteNode
        return false
    elseif expr isa Symbol
        return true
    elseif expr isa Expr && expr.head == :call && !isempty(expr.args) && expr.args[1] === :(=>)
        return _address_expr_has_dynamic_parts(expr.args[2]) || _address_expr_has_dynamic_parts(expr.args[3])
    elseif expr isa Expr && expr.head == :tuple
        return any(_address_expr_has_dynamic_parts, expr.args)
    elseif expr isa Number || expr isa String || expr isa Char
        return false
    end

    return true
end

function _expr_has_dynamic_content(expr)
    if expr isa QuoteNode
        return false
    elseif expr isa Symbol
        return true
    elseif expr isa Number || expr isa String || expr isa Char
        return false
    elseif expr isa Expr
        if expr.head == :call
            start = 2
            if !isempty(expr.args) && expr.args[1] isa Symbol && expr.args[1] in (Symbol(":"), :+, :-, :*, :/, :%, :^, :(=>))
                start = 2
            else
                start = 1
            end
            for idx = start:length(expr.args)
                _expr_has_dynamic_content(expr.args[idx]) && return true
            end
            return false
        end

        return any(_expr_has_dynamic_content, expr.args)
    end

    return true
end

function _parse_loop_scope(iteration)
    if iteration isa Expr && iteration.head in (:(=), :in)
        iterator = iteration.args[1]
        iterator isa Symbol || throw(ArgumentError("@tea only supports simple loop iterators, got $iterator"))
        return iterator, iteration.args[2]
    end

    throw(ArgumentError("@tea only supports simple `for x in xs` loops in static analysis"))
end

function _collect_plan_nodes!(expr, nodes::Vector{Tuple}, loop_scopes::Tuple=())
    if !(expr isa Expr)
        return nodes
    end

    if expr.head == :block
        for arg in expr.args
            arg isa LineNumberNode && continue
            _collect_plan_nodes!(arg, nodes, loop_scopes)
        end
        return nodes
    end

    if expr.head == :for && length(expr.args) == 2
        scope = _parse_loop_scope(expr.args[1])
        _collect_plan_nodes!(expr.args[2], nodes, (loop_scopes..., scope))
        return nodes
    end

    if expr.head == :(=) && length(expr.args) == 2
        lhs, rhs = expr.args
        if rhs isa Expr && rhs.head == :call && !isempty(rhs.args) && rhs.args[1] === :~
            binding_override = lhs isa Symbol ? lhs : nothing
            push!(nodes, (:choice, rhs, loop_scopes, binding_override))
        elseif lhs isa Symbol
            push!(nodes, (:deterministic, lhs, rhs, loop_scopes))
        end
        return nodes
    end

    if expr.head == :call && !isempty(expr.args) && expr.args[1] === :~
        push!(nodes, (:choice, expr, loop_scopes, nothing))
        return nodes
    end

    for arg in expr.args
        _collect_plan_nodes!(arg, nodes, loop_scopes)
    end

    return nodes
end

function _execution_plan_steps_expr(plan_nodes, slot_lookup)
    step_exprs = Expr[]
    choice_index = 0

    for node in plan_nodes
        if node[1] === :choice
            choice_index += 1
            choice_expr = node[2]
            loop_scopes = node[3]
            binding_override = node[4]
            lhs = choice_expr.args[2]
            binding_symbol = isnothing(binding_override) ? (lhs isa Symbol ? lhs : nothing) : binding_override
            binding = isnothing(binding_symbol) ? :(nothing) : QuoteNode(binding_symbol)
            address = _address_spec_expr(lhs)
            rhs_spec = _rhs_spec_expr(choice_expr.args[3])
            scopes_expr = Expr(:vect, map(_loop_scope_spec_expr, loop_scopes)...)
            slot_expr = haskey(slot_lookup, choice_index) ? slot_lookup[choice_index] : :(nothing)
            push!(
                step_exprs,
                :($(_qualify(:ChoicePlanStep))(
                    $choice_index,
                    $binding,
                    $address,
                    $rhs_spec,
                    $scopes_expr,
                    $slot_expr,
                )),
            )
        elseif node[1] === :deterministic
            scopes_expr = Expr(:vect, map(_loop_scope_spec_expr, node[4])...)
            push!(
                step_exprs,
                :($(_qualify(:DeterministicPlanStep))(
                    $(QuoteNode(node[2])),
                    nothing,
                    $(QuoteNode(node[3])),
                    $scopes_expr,
                )),
            )
        end
    end

    return Expr(:vect, step_exprs...)
end

function _model_spec_expr(mode_expr, signature, body)
    name = signature.args[1]
    arguments = [_argument_name(arg) for arg in signature.args[2:end]]
    argument_expr = Expr(:vect, map(QuoteNode, arguments)...)
    plan_nodes = Tuple[]
    _collect_plan_nodes!(body, plan_nodes)
    choice_nodes = Tuple{Expr,Tuple,Any}[(node[2], node[3], node[4]) for node in plan_nodes if node[1] === :choice]
    choice_exprs = map(node -> _choice_spec_expr(node[1], node[2], node[3]), choice_nodes)
    choices_expr = Expr(:vect, choice_exprs...)
    shape_specialized = any(
        node -> _address_has_dynamic_parts(node[1].args[2]) || any(scope -> _expr_has_dynamic_content(scope[2]), node[2]),
        choice_nodes,
    )
    parameter_layout_expr, slot_lookup = _parameter_layout_expr(choice_nodes)
    plan_steps_expr = _execution_plan_steps_expr(plan_nodes, slot_lookup)
    return_expr = _return_expr_expr(body)

    return :($(_qualify(:ModelSpec))(
        $(QuoteNode(name)),
        $(QuoteNode(mode_expr)),
        $argument_expr,
        $choices_expr,
        $shape_specialized,
        $parameter_layout_expr,
        $return_expr,
        $plan_steps_expr,
    ))
end

function _return_expr_expr(body)
    body isa Expr && body.head == :block || return QuoteNode(body)

    for arg in reverse(body.args)
        arg isa LineNumberNode && continue
        if arg isa Expr && arg.head == :return
            return QuoteNode(arg.args[1])
        end
        return QuoteNode(arg)
    end

    return QuoteNode(nothing)
end

# True when `expr` contains a `~` choice or an assignment anywhere inside.
function _contains_choice_or_assignment(expr)
    expr isa Expr || return false
    expr.head == :(=) && return true
    expr.head == :call && !isempty(expr.args) && expr.args[1] === :~ && return true
    return any(_contains_choice_or_assignment, expr.args)
end

# Find the first branchful control-flow expression in a model body: any
# `if`/`elseif`/`else` block or ternary `?:` (both parse to `Expr(:if, ...)`),
# and any short-circuit `&&`/`||` that contains a `~` choice or an assignment.
# The linear execution plan has no IR for branches -- it would append BOTH
# branches' choices and assignments to the same plan and silently miscompile
# logjoint -- so static models must reject these at macro expansion, and
# dynamic models must refuse compiled scoring (generate/assess execute the
# real body and stay correct).
function _first_branchful_expr(expr)
    expr isa Expr || return nothing
    (expr.head == :if || expr.head == :elseif) && return expr
    if (expr.head == :&& || expr.head == :||) && _contains_choice_or_assignment(expr)
        return expr
    end
    for arg in expr.args
        found = _first_branchful_expr(arg)
        isnothing(found) || return found
    end
    return nothing
end

function _reject_static_branchful(body)
    found = _first_branchful_expr(body)
    isnothing(found) && return nothing
    if found.head == :&& || found.head == :||
        throw(
            ArgumentError(
                "@tea static models do not support short-circuit `&&`/`||` containing a `~` choice " *
                "or an assignment (found `$(found)`): the static execution plan linearizes the model " *
                "body, so the conditionally-executed choice/assignment would be recorded and scored " *
                "unconditionally, silently corrupting logjoint. Supported instead: deterministic " *
                "computation (including `ifelse(cond, a, b)` for value selection), static `for` loops, " *
                "and moving data-dependent branching outside the model (pass the result as a model argument).",
            ),
        )
    end
    throw(
        ArgumentError(
            "@tea static models do not support `if`/`elseif`/`else` blocks or the ternary `?:` " *
            "operator: the static execution plan linearizes the model body, so the choices and " *
            "assignments of BOTH branches would be recorded and scored unconditionally, silently " *
            "corrupting logjoint. Supported instead: deterministic computation (including " *
            "`ifelse(cond, a, b)` for value selection), static `for` loops, and moving " *
            "data-dependent branching outside the model (pass the result as a model argument).",
        ),
    )
end

function _expand_tea(mode_expr, definition)
    definition isa Expr && definition.head == :function ||
        throw(ArgumentError("@tea expects a function definition"))

    signature = definition.args[1]
    body = definition.args[2]

    signature isa Expr && signature.head == :call ||
        throw(ArgumentError("@tea currently supports standard function definitions only"))

    name = signature.args[1]
    name isa Symbol || throw(ArgumentError("@tea expects a plain function name"))

    # Branchful control flow cannot be represented by the linear execution
    # plan. Static models reject it outright; dynamic models keep the correct
    # runtime (generate/assess) behavior but are flagged so compiled scoring
    # refuses to silently linearize both branches.
    branchful = !isnothing(_first_branchful_expr(body))
    mode_expr === :static && _reject_static_branchful(body)

    ctxsym = gensym(:tea_ctx)
    impl_name = gensym(name)
    impl_signature = Expr(:call, impl_name, ctxsym, signature.args[2:end]...)
    rewritten_body = _rewrite_tea_expr(body, ctxsym)
    mode = _tea_mode(mode_expr)
    spec_expr = _model_spec_expr(mode_expr, signature, body)
    function_definition = Expr(:function, impl_signature, rewritten_body)

    # Argument filler: a function with the model's exact signature (defaults
    # included) that returns the completed argument tuple, so the compiled
    # scoring APIs fill missing trailing arguments with Julia's own
    # default-argument semantics and agree with `generate`.
    filler_name = gensym(Symbol(name, :_argfill))
    filler_signature = Expr(:call, filler_name, map(deepcopy, signature.args[2:end])...)
    filler_body = Expr(:tuple, [_argument_name(arg) for arg in signature.args[2:end]]...)
    filler_definition = Expr(:function, filler_signature, filler_body)

    binding_definition = :(
        $name = $(_qualify(:TeaModel))(
            $mode,
            $(QuoteNode(name)),
            $impl_name,
            $spec_expr;
            argument_filler=($filler_name),
            branchful=($branchful),
        )
    )

    return esc(Expr(:block, function_definition, filler_definition, binding_definition))
end

macro tea(args...)
    if length(args) == 1
        return _expand_tea(:dynamic, args[1])
    elseif length(args) == 2
        return _expand_tea(args[1], args[2])
    else
        throw(ArgumentError("@tea expects `@tea function ... end` or `@tea static function ... end`"))
    end
end
