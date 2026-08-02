module UncertainTeaTablesExt

# Tables.jl interface (issue #340): make `HMCChains` and `PredictiveDraws`
# valid Tables.jl sources so `DataFrame(chains)`, `CSV.write(path, chains)`,
# and every other Tables.jl sink work directly.
#
# Both containers materialize as WIDE column tables:
#   - HMCChains       -> `chain`, `draw`, then one column per constrained
#                        parameter (named as in `parameter_names`, e.g. "mu",
#                        "v[1]"). Rows are chain-major: chain 1's draws in
#                        order, then chain 2's, ... matching the pooled-draw
#                        row order of `constrained_draws`.
#   - PredictiveDraws -> `draw`, then one column per predictive address (named
#                        like the ArviZ export, e.g. "y", "y[3]"). Vector-valued
#                        addresses become columns of vectors.

using UncertainTea
import Tables

# --- HMCChains ----------------------------------------------------------------

Tables.istable(::Type{<:UncertainTea.HMCChains}) = true
Tables.columnaccess(::Type{<:UncertainTea.HMCChains}) = true

function Tables.columns(chains::UncertainTea.HMCChains)
    draws = UncertainTea.posterior_array(chains; space=:constrained)
    num_samples, num_chains, num_params = size(draws)
    names = vcat(
        [:chain, :draw],
        Symbol.(UncertainTea.parameter_names(chains; space=:constrained)),
    )
    allunique(names) || throw(
        ArgumentError(
            "cannot build a table: parameter names collide with the :chain/:draw columns ($(names))",
        ),
    )
    chain_column = repeat(1:num_chains; inner=num_samples)
    draw_column = repeat(1:num_samples; outer=num_chains)
    # vec of the (num_samples, num_chains) slice is column-major, i.e. exactly
    # the chain-major row order declared above.
    parameter_columns = [vec(draws[:, :, p]) for p = 1:num_params]
    return NamedTuple{Tuple(names)}((chain_column, draw_column, parameter_columns...))
end

# --- PredictiveDraws ----------------------------------------------------------

Tables.istable(::Type{UncertainTea.PredictiveDraws}) = true
Tables.columnaccess(::Type{UncertainTea.PredictiveDraws}) = true

function Tables.columns(draws::UncertainTea.PredictiveDraws)
    addresses = UncertainTea.addresses(draws)
    names = vcat([:draw], [Symbol(UncertainTea._observation_display_name(a)) for a in addresses])
    allunique(names) || throw(
        ArgumentError(
            "cannot build a table: predictive address names collide ($(names))",
        ),
    )
    columns = Any[collect(1:length(draws))]
    for address in addresses
        push!(columns, draws[address])
    end
    return NamedTuple{Tuple(names)}(Tuple(columns))
end

end # module
