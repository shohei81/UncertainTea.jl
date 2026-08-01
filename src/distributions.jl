include("distributions/core.jl")
include("distributions/registration.jl")
# single-source scalar logpdf kernels + analytic partials (issue #285); the CPU
# logpdf methods, backend scoring, and batched gradients all consume these
include("distributions/scalar_kernels.jl")
include("distributions/continuous.jl")
include("distributions/discrete.jl")
include("distributions/structured.jl")
include("distributions/gaussian_process.jl")
include("distributions/sparse_gaussian_process.jl")
include("distributions/hidden_markov.jl")
include("distributions/ordered_logistic.jl")
include("distributions/truncated.jl")
