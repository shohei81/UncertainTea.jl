# Elliptical slice sampling (Murray, Adams & MacKay 2010, issue #294): the
# go-to GRADIENT-FREE, TUNING-FREE sampler for a latent vector with a
# multivariate-Gaussian prior -- exactly the GP latent-function models added in
# issue #280, where strong prior correlations make random-walk moves poor and
# every NUTS leapfrog costs a gradient. Each update draws an auxiliary
# nu ~ N(0, Sigma), then slice-samples the angle of the ellipse
# f' = f cos(phi) + nu sin(phi) under the LIKELIHOOD alone; every point on the
# ellipse has equal prior density, so only log-likelihood evaluations are
# needed and every iteration accepts (the bracket shrinks until it does).

struct EllipticalSliceResult{S<:AbstractMatrix}
    samples::S               # dimension x num_samples (post-warmup)
    num_warmup::Int
    # likelihood evaluations per stored sample (the sampler's only cost knob;
    # 1 contraction ~= 2 evaluations)
    average_likelihood_evaluations::Float64
end

"""
    elliptical_slice(loglikelihood, prior_scale_tril; num_samples, num_warmup=0,
                     initial_params=nothing, rng=Random.default_rng())

Elliptical slice sampling for a target `p(f) ∝ N(f; 0, L L') · exp(loglikelihood(f))`,
where `prior_scale_tril` is the lower-triangular Cholesky factor `L` of the
prior covariance (e.g. `gp_cholesky(X, kernel, jitter)` for a GP prior).
Gradient-free and tuning-free: each iteration draws one auxiliary prior sample
and shrinks an angle bracket until the move is accepted, so `loglikelihood` is
the only model code required — compose it from `UncertainTea.logpdf` calls for
non-Gaussian likelihoods (classification, counts). Returns an
`EllipticalSliceResult` with `dimension × num_samples` post-warmup draws.
"""
function elliptical_slice(
    loglikelihood,
    prior_scale_tril::AbstractMatrix;
    num_samples::Int,
    num_warmup::Int=0,
    initial_params=nothing,
    rng::AbstractRNG=Random.default_rng(),
)
    num_samples >= 1 || throw(ArgumentError("elliptical_slice requires num_samples >= 1"))
    num_warmup >= 0 || throw(ArgumentError("elliptical_slice requires num_warmup >= 0"))
    n = size(prior_scale_tril, 1)
    size(prior_scale_tril, 2) == n ||
        throw(ArgumentError("elliptical_slice prior_scale_tril must be square"))
    f = isnothing(initial_params) ? prior_scale_tril * randn(rng, n) : collect(float.(initial_params))
    length(f) == n || throw(
        ArgumentError("elliptical_slice initial state length $(length(f)) must match the prior dimension $n"),
    )
    current_loglik = loglikelihood(f)
    isfinite(current_loglik) ||
        throw(ArgumentError("elliptical_slice initial state has a non-finite log-likelihood"))

    samples = Matrix{Float64}(undef, n, num_samples)
    evaluations = 0
    total_iterations = num_warmup + num_samples
    proposal = similar(f)
    for iteration = 1:total_iterations
        nu = prior_scale_tril * randn(rng, n)
        log_y = current_loglik + log(rand(rng))
        phi = 2 * pi * rand(rng)
        phi_min, phi_max = phi - 2 * pi, phi
        while true
            @. proposal = f * cos(phi) + nu * sin(phi)
            proposal_loglik = loglikelihood(proposal)
            evaluations += 1
            if proposal_loglik > log_y
                f, proposal = proposal, f
                current_loglik = proposal_loglik
                break
            end
            # shrink the bracket toward phi = 0 (the current state) and retry;
            # phi = 0 reproduces f exactly, so termination is guaranteed
            if phi < 0
                phi_min = phi
            else
                phi_max = phi
            end
            phi = phi_min + (phi_max - phi_min) * rand(rng)
        end
        if iteration > num_warmup
            samples[:, iteration-num_warmup] = f
        end
    end
    return EllipticalSliceResult(samples, num_warmup, evaluations / total_iterations)
end
