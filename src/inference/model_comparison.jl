# WAIC and PSIS-LOO model comparison from a pointwise log-likelihood matrix.
#
# References:
#   Vehtari, Gelman & Gabry (2017), "Practical Bayesian model evaluation using
#     leave-one-out cross-validation and WAIC", Stat. Comput. 27:1413-1432.
#   Zhang & Stephens (2009), "A New and Efficient Estimation Method for the
#     Generalized Pareto Distribution", Technometrics 51(3):316-325.

struct WAICResult
    elpd::Float64
    p_eff::Float64
    se::Float64
    pointwise::Vector{Float64}
end

struct LOOResult
    elpd::Float64
    p_eff::Float64
    se::Float64
    pointwise::Vector{Float64}
    pareto_k::Vector{Float64}
end

function _mc_logsumexp(values::AbstractVector)
    isempty(values) && throw(ArgumentError("logsumexp requires a non-empty vector"))
    max_value = maximum(values)
    isfinite(max_value) || return max_value
    total = 0.0
    for value in values
        total += exp(value - max_value)
    end
    return max_value + log(total)
end

function _mc_mean(x::AbstractVector)
    return sum(x) / length(x)
end

function _mc_var(x::AbstractVector)
    n = length(x)
    n > 1 || return 0.0
    m = _mc_mean(x)
    total = 0.0
    for xi in x
        total += (xi - m)^2
    end
    return total / (n - 1)
end

"""
    _gpd_fit(exceedances) -> (k, sigma)

Fit a generalized Pareto distribution to `exceedances` (positive values over a threshold)
using the Zhang & Stephens (2009) profile-likelihood estimator. Returns the shape `k` and
scale `sigma`. A weakly-informative prior is applied to `k`.
"""
function _gpd_fit(exceedances::AbstractVector)
    x = sort!(collect(float.(exceedances)))
    n = length(x)
    n >= 2 || return (Inf, NaN)

    prior_bs = 3.0
    prior_k = 10.0
    m = 30 + floor(Int, sqrt(n))
    quartile_index = clamp(floor(Int, n / 4 + 0.5), 1, n)
    x_quartile = x[quartile_index]
    x_max = x[n]

    b = Vector{Float64}(undef, m)
    len_scale = Vector{Float64}(undef, m)
    for i = 1:m
        bi = (1 - sqrt(m / (i - 0.5))) / (prior_bs * x_quartile) + 1 / x_max
        b[i] = bi
        acc = 0.0
        for xi in x
            acc += log1p(-bi * xi)
        end
        ki = acc / n
        len_scale[i] = n * (log(-bi / ki) - ki - 1)
    end

    # Posterior weights over the theta (b) grid: softmax of the profile log-likelihood.
    max_ls = maximum(len_scale)
    weight_sum = 0.0
    weights = Vector{Float64}(undef, m)
    for i = 1:m
        weights[i] = exp(len_scale[i] - max_ls)
        weight_sum += weights[i]
    end
    b_post = 0.0
    for i = 1:m
        b_post += b[i] * (weights[i] / weight_sum)
    end

    acc = 0.0
    for xi in x
        acc += log1p(-b_post * xi)
    end
    k_raw = acc / n
    sigma = -k_raw / b_post
    k = (n * k_raw + prior_k * 0.5) / (n + prior_k)
    return (k, sigma)
end

# Inverse CDF (quantile) of the generalized Pareto distribution.
function _gpd_inv(probs::AbstractVector, k::Real, sigma::Real)
    x = Vector{Float64}(undef, length(probs))
    if !(sigma > 0)
        fill!(x, NaN)
        return x
    end
    for i in eachindex(probs)
        p = probs[i]
        if p <= 0
            x[i] = 0.0
        elseif p >= 1
            x[i] = k < 0 ? -sigma / k : Inf
        elseif abs(k) < 1e-12
            x[i] = -sigma * log1p(-p)
        else
            x[i] = sigma * expm1(-k * log1p(-p)) / k
        end
    end
    return x
end

# PSIS smoothing of a single observation's raw log importance ratios.
# Returns the normalized smoothed log weights and the fitted Pareto shape k.
function _psis_smooth(log_ratios::AbstractVector)
    n = length(log_ratios)
    lw = collect(float.(log_ratios))
    tail_len = ceil(Int, min(0.2 * n, 3 * sqrt(n)))

    if tail_len <= 4 || n <= 5
        lw .-= _mc_logsumexp(lw)
        return lw, Inf
    end

    perm = sortperm(lw)                       # ascending
    tail_positions = perm[(n-tail_len+1):n]  # indices of the largest tail_len ratios
    cutoff = lw[perm[n-tail_len]]              # largest value below the tail
    tail_values = lw[tail_positions]             # ascending (perm was sorted)
    # Exponentiate relative to the largest log ratio so the smoothing is
    # invariant to a common offset and cannot overflow (issue #82).
    shift = lw[perm[n]]
    exp_cutoff = exp(cutoff - shift)
    exceedances = exp.(tail_values .- shift) .- exp_cutoff

    k, sigma = _gpd_fit(exceedances)
    if isfinite(k) && sigma > 0
        probs = [(i - 0.5) / tail_len for i = 1:tail_len]
        smoothed = log.(_gpd_inv(probs, k, sigma) .+ exp_cutoff) .+ shift
        for (idx, pos) in enumerate(tail_positions)
            lw[pos] = smoothed[idx]
        end
    end

    # Truncate at the largest raw log weight.
    max_raw = maximum(log_ratios)
    for i in eachindex(lw)
        if lw[i] > max_raw
            lw[i] = max_raw
        end
    end

    lw .-= _mc_logsumexp(lw)
    return lw, k
end

"""
    waic(ll::AbstractMatrix) -> WAICResult

Compute the widely applicable information criterion from an `S x N` pointwise
log-likelihood matrix (`S` draws, `N` observations).
"""
function waic(ll::AbstractMatrix)
    n_draws, n_obs = size(ll)
    log_s = log(n_draws)
    elpd_i = Vector{Float64}(undef, n_obs)
    p_i = Vector{Float64}(undef, n_obs)
    for i = 1:n_obs
        col = view(ll, :, i)
        lppd_i = _mc_logsumexp(col) - log_s
        p_i[i] = _mc_var(col)
        elpd_i[i] = lppd_i - p_i[i]
    end
    elpd = sum(elpd_i)
    p_eff = sum(p_i)
    se = sqrt(n_obs * _mc_var(elpd_i))
    return WAICResult(elpd, p_eff, se, elpd_i)
end

"""
    psis_loo(ll::AbstractMatrix) -> LOOResult

Compute Pareto-smoothed importance sampling leave-one-out cross-validation from an
`S x N` pointwise log-likelihood matrix.
"""
function psis_loo(ll::AbstractMatrix)
    n_draws, n_obs = size(ll)
    log_s = log(n_draws)
    elpd_i = Vector{Float64}(undef, n_obs)
    pareto_k = Vector{Float64}(undef, n_obs)
    lppd = 0.0
    for i = 1:n_obs
        col = view(ll, :, i)
        log_ratios = -col
        logw, k = _psis_smooth(log_ratios)
        elpd_i[i] = _mc_logsumexp(col .+ logw)
        pareto_k[i] = k
        lppd += _mc_logsumexp(col) - log_s
    end
    elpd = sum(elpd_i)
    p_loo = lppd - elpd
    se = sqrt(n_obs * _mc_var(elpd_i))
    return LOOResult(elpd, p_loo, se, elpd_i, pareto_k)
end

# The four-argument convenience wrappers (`waic`/`psis_loo`/`loo` from an
# inference result) live in result_interface.jl: they are typed on the
# posterior-draws interface union, which is only complete once every result
# type has been defined.

"""
    loo(model, args, constraints, result; kwargs...) -> LOOResult
    loo(ll::AbstractMatrix) -> LOOResult

Leave-one-out cross-validation via Pareto-smoothed importance sampling; an
alias for `psis_loo`. The four-argument form computes the pointwise
log-likelihood matrix from an inference `result` first (anything implementing
`constrained_draws`: `HMCChains`, `GibbsChain`, `ADVIResult`, `LaplaceResult`,
weighted particle results, ...; keyword arguments are forwarded to
`constrained_draws`); the matrix form takes an `S x N` (draws x observations)
log-likelihood matrix directly.

`constraints` is required — it defines which choices are the observations that
LOO leaves out. The same `(model, args, constraints, result)` argument list
also drives [`predict`](@ref) and the log-likelihood-emitting form of
[`to_arviz_dict`](@ref), so one triple covers the whole posterior workflow. A
`result` that is not an inference result (or a log-likelihood matrix) fails
with a `MethodError`.
"""
loo(ll::AbstractMatrix) = psis_loo(ll)

function Base.show(io::IO, result::WAICResult)
    print(io, "WAICResult(elpd=", round(result.elpd; digits=2),
        ", p_eff=", round(result.p_eff; digits=2),
        ", se=", round(result.se; digits=2),
        ", n=", length(result.pointwise), ")")
end

function Base.show(io::IO, result::LOOResult)
    max_k = isempty(result.pareto_k) ? 0.0 : maximum(result.pareto_k)
    print(io, "LOOResult(elpd=", round(result.elpd; digits=2),
        ", p_eff=", round(result.p_eff; digits=2),
        ", se=", round(result.se; digits=2),
        ", n=", length(result.pointwise),
        ", max_pareto_k=", round(max_k; digits=2), ")")
end
