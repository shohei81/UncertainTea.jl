const Address = Tuple{Vararg{Any}}

mutable struct ChoiceMap
    entries::Vector{Pair{Address,Any}}
    index_by_address::Dict{Address,Int}
    # Monotonic mutation counter (issue #145): the dense observation staging in
    # evaluator.jl snapshots constrained values at gradient-cache construction
    # and assumes the constraints stay immutable for the cache's lifetime. Every
    # `_pushchoice!` bumps this counter (insert AND value replacement), so the
    # staged scorer can verify the assumption in O(1) and fall back to live
    # ChoiceMap lookups when it no longer holds (gibbs mutates its merged
    # constraints in place between gradient evaluations).
    mutation_count::Int
end

ChoiceMap(entries::Vector{Pair{Address,Any}}, index_by_address::Dict{Address,Int}) =
    ChoiceMap(entries, index_by_address, 0)

function ChoiceMap(entries::Vector{Pair{Address,Any}})
    index_by_address = Dict{Address,Int}()
    for (index, entry) in enumerate(entries)
        index_by_address[first(entry)] = index
    end
    return ChoiceMap(entries, index_by_address)
end

ChoiceMap() = ChoiceMap(Pair{Address,Any}[])

"""
    choicemap(entries...) -> ChoiceMap

Build a `ChoiceMap` from `(address, value)` entries — the container that fixes
choice values, most commonly the observations passed to inference:
`choicemap((:y, 0.3))`, `choicemap((:y => 1, 0.3), (:y => 2, -0.1))`. A single
iterable of entries also works, e.g.
`choicemap((:y => i, ys[i]) for i in eachindex(ys))`. `address => value` pairs
are accepted in place of tuples.
"""
function choicemap(entries...)
    cm = ChoiceMap()
    if length(entries) == 1 && !_is_choice_entry(entries[1])
        for entry in entries[1]
            _pushchoice!(cm, entry)
        end
        return cm
    end

    for entry in entries
        _pushchoice!(cm, entry)
    end
    return cm
end

# A single `(address, value)` entry. A 2-tuple whose elements are BOTH Pairs
# is NOT one entry: `(:a => 1.0, :b => 2.0)` is a collection of two
# `address => value` entries (matching vectors of pairs and longer tuples of
# pairs); treating it as one entry would silently flatten the first pair into
# a corrupted address and drop both constraints.
_is_choice_entry(entry) =
    entry isa Pair || (entry isa Tuple && length(entry) == 2 && !(entry[1] isa Pair && entry[2] isa Pair))

function normalize_address(address)
    return (address,)
end

normalize_address(address::Symbol) = (address,)
normalize_address(address::Pair) = (normalize_address(address.first)..., normalize_address(address.second)...)
normalize_address(address::QuoteNode) = normalize_address(address.value)

function normalize_address(address::Tuple)
    flattened = ()
    for part in address
        flattened = (flattened..., normalize_address(part)...)
    end
    return flattened
end

# Concrete addresses assembled from plan specs are already flat unless a
# dynamic part evaluated to a nested Pair or Tuple at runtime.
function _needs_address_normalization(address::Tuple)
    return any(part -> part isa Pair || part isa Tuple || part isa QuoteNode, address)
end

function _normalize_concrete_address(address::Tuple)
    return _needs_address_normalization(address) ? normalize_address(address) : address
end

function _pushchoice!(cm::ChoiceMap, entry::Pair)
    return _pushchoice!(cm, entry.first, entry.second)
end

function _pushchoice!(cm::ChoiceMap, entry::Tuple)
    length(entry) == 2 || throw(ArgumentError("choice entries must have exactly two elements"))
    if entry[1] isa Pair && entry[2] isa Pair
        throw(
            ArgumentError(
                "ambiguous choice entry $(entry): a 2-tuple of Pairs looks like a collection of " *
                "`address => value` entries, not a single `(address, value)` entry; pass the pairs " *
                "directly, e.g. `choicemap(:a => 1.0, :b => 2.0)`",
            ),
        )
    end
    return _pushchoice!(cm, entry[1], entry[2])
end

function _pushchoice!(cm::ChoiceMap, address, value)
    normalized = normalize_address(address)
    index = _choice_index_normalized(cm, normalized)
    cm.mutation_count += 1
    if !isnothing(index)
        cm.entries[index] = normalized => value
        return cm
    end
    push!(cm.entries, normalized => value)
    cm.index_by_address[normalized] = length(cm.entries)
    return cm
end

function _choice_index_normalized(cm::ChoiceMap, normalized::Address)
    index = get(cm.index_by_address, normalized, 0)
    return index == 0 ? nothing : index
end

function _choice_get_normalized(cm::ChoiceMap, normalized::Address)
    index = _choice_index_normalized(cm, normalized)
    isnothing(index) && throw(KeyError(normalized))
    return last(cm.entries[index])
end

function _choice_tryget_normalized(cm::ChoiceMap, normalized::Address)
    index = _choice_index_normalized(cm, normalized)
    isnothing(index) && return false, nothing
    return true, last(cm.entries[index])
end

function Base.getindex(cm::ChoiceMap, address)
    normalized = normalize_address(address)
    return _choice_get_normalized(cm, normalized)
end

function Base.haskey(cm::ChoiceMap, address)
    normalized = normalize_address(address)
    found, _ = _choice_tryget_normalized(cm, normalized)
    return found
end

Base.length(cm::ChoiceMap) = length(cm.entries)
Base.isempty(cm::ChoiceMap) = isempty(cm.entries)
Base.iterate(cm::ChoiceMap, state...) = iterate(cm.entries, state...)

function Base.show(io::IO, cm::ChoiceMap)
    print(io, "ChoiceMap(", length(cm.entries), " entries)")
end
