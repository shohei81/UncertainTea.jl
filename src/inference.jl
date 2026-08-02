include("inference/types_and_hmc.jl")
include("inference/nuts_core.jl")
include("inference/api.jl")
include("inference/elliptical_slice.jl")
# Last on purpose: the posterior-draws interface union covers every result
# type defined above (issue #337).
include("inference/result_interface.jl")
