module Cloudy

include(joinpath("Kernels","KernelTensors.jl"))
include(joinpath("Kernels","KernelFunctions.jl"))
include(joinpath("ParticleDistributions","ParticleDistributions.jl"))
include(joinpath("Sources","Sources.jl"))
include(joinpath("BasisFunctions.jl"))
include(joinpath("Galerkin.jl"))
include(joinpath("Collocation.jl"))

end
