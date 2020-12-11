"""
  Microphysical source functions

"""
module Sources

using ..ParticleDistributions
using ..KernelFunctions

# methods that compute source terms from microphysical parameterizations
export get_coalescence_integral
export get_coalescence_integral_moment


"""
get_coalescence_integral(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int)

Returns the collision-coalescence integral at points `x`.
"""
function get_coalescence_integral(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int) where {FT<:Real}
  # monte carlo samples
  y = sort(sample(pdist, n_samples))

  out = zeros(size(x))
  # source contribution to collision integral
  for (i, xx) in enumerate(x)
    for yy in  y[y .<= xx]
      out[i] += 0.5 * pdist(xx - yy) * kernel(xx - yy, yy)
    end
  end

  # sink contribution to collision integral
  out -= pdist.(x) .* sum(kernel.(x, y'), dims=2)

  # normalize to get correct monte carlo average
  out *= pdist.n / length(y) 

  return out
end


"""
get_coalescence_integral_moment(k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int)

Returns the collision-coalescence integral at moment of order `k`.
"""
function get_coalescence_integral_moment(k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int) where {FT<:Real}
  # monte carlo samples
  x = zeros(n_samples)
  y = zeros(n_samples)
  for i in 1:n_samples
    x[i], y[i] = sample(pdist, 2)
  end

  # collision integral for the k-th moment
  out = 0.0
  for i in 1:n_samples
    out += 0.5 * kernel(x[i], y[i]) * ((x[i]+y[i])^k - x[i]^k - y[i]^k)
  end

  # normalize to get correct monte carlo average
  out *= (pdist.n)^2 / n_samples

  return out
end


end #module Sources.jl