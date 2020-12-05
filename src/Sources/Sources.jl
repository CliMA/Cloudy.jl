"""
  Microphysical source functions

"""
module Sources

using ..ParticleDistributions
using ..KernelFunctions

# methods that compute source terms from microphysical parameterizations
export get_coalescence_integral


"""
get_coalescence_integral(x::Array{FT}, kern::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int
)

Returns the collision-coalescence integral at points `x`.
"""
function get_coalescence_integral(x::Array{FT}, kern::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int) where {FT<:Real}
  # monte carlo samples
  y = sort(sample(pdist, n_samples))

  out = zeros(size(x))
  # source contribution to collision integral
  for (i, xx) in enumerate(x)
    for yy in  y[y .<= xx]
      out[i] += 0.5 * pdist(xx - yy) * kern(xx - yy, yy)
    end
  end

  # sink contribution to collision integral
  out -= pdist.(x) .* sum(kern.(x, y'), dims=2)

  # normalize to get correct monte carlo average
  out *= pdist.n / length(y) 

  return out
end

end #module Sources.jl
