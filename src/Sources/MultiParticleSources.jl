"""
  Microphysical source functions

"""
module MultiParticleSources

using HCubature
using NIntegration
using NLsolve
using Distributions
using SpecialFunctions: gamma
using ..ParticleDistributions
using ..KernelFunctions
# methods that compute source terms from microphysical parameterizations
export get_coalescence_integral
export get_coalescence_integral_moment
export constant_coalescence_efficiency

"""
get_coalescence_integral(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int)

Returns the collision-coalescence integral at points `x`.
"""
function get_coalescence_integrals(x::Array{FT}, kernel::KernelFunction{FT}, 
  pdists::Array{ParticleDistribution{FT}}, n_samples::Int) where {FT<:Real}
  
  out_plus = zeros((size(x),len(p_dists)))
  out_minus = zeros((size(x),len(p_dists)))
  for p in 1:len(pdists)
    # monte carlo samples
    y = sort(sample(pdists[p], n_samples))

    # source contribution to collision integral
    for (i, xx) in enumerate(x)
      for yy in  y[y .<= xx]
        out_plus[i, p] += 0.5 * pdist(xx - yy) * kernel(xx - yy, yy)
      end
    end

    # sink contribution to collision integral
    out_minus[:,p] += pdist.(x) .* sum(kernel.(x, y'), dims=2)

    # normalize to get correct monte carlo average
    out_plus *= pdists[p].n / length(y) 
    out_minus *= pdists[p].n / length(y) 
  end
  return (out_plus, out_minus)
end


"""
get_coalescence_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
    n_samples_mc::Int, coalescence_efficiency::Function)

Returns the collision-coalescence integral at moment of order `k`, using
Monte Carlo integration
"""
function get_coalescence_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdists::Array{ParticleDistribution{FT}},
    n_samples_mc::Int, coalescence_efficiency::Function
    ) where {FT<:Real}

  mom_out_plus = zeros((len(p_dists), len(p_dists)))
  mom_out_minus = zeros((len(p_dists), len(p_dists)))

  # # monte carlo samples
  # x_samples = zeros((n_samples_mc, len(p_dists)))
  # for p in 1:len(pdists)
  #   for i in 1:n_samples_mc
  #     x_samples[i, p] = ParticleDistributions.sample(pdists[p], 1)
  #   end
  # end

  for p1 in 1:len(pdists)
    # monte carlo samples for inner integral
    y = sort(sample(pdists[p1], n_samples_mc))
    (out_plus, out_minus) = get_coalescence_integrals(y, kernel, pdists, n_samples_mc)
    
    for p2 in 1:len(pdists)
        x = sort(sample(pdists[p2], n_samples_mc))
        # collision integrals for the k-th moment
        for i in 1:n_samples_mc
          mom_out_plus[p1, p2] += out_plus[i, p1] * pdists[p2].(x[i]) * x[i]^k
          mom_out_minus[p1, p2] += out_minus[i, p1] * pdists[p2].(x[i]) * x[i]^k
        end

        # normalize to get correct monte carlo average
        mom_out_plus[p1, p2] *= (pdists[p1].n * pdists[p2].n) / n_samples_mc
        mom_out_minus[p1, p2] *= (pdists[p1].n * pdists[p2].n) / n_samples_mc
    end
  end

  return (mom_out_plus, mom_out_minus)
end


# """
# get_coalescence_integral_moment(
#     k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}
#     )

# Returns the collision-coalescence integral at moment of order `k`, using
# quadrature
# """
# function get_coalescence_integral_moment(
#     k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}
#     ) where {FT<:Real}

#   function coal_integrand(x)
#       integrand = 0.5 * pdist(x[1]) * pdist(x[2]) + kernel(x[1], x[2]) * ((x[1] + x[2])^k -x[1]^k - x[2]^k)
#       return integrand
#   end
#   max_mass = ParticleDistributions.moment(pdist, 1.0)
#   out = hcubature(coal_integrand, [0, 0], [max_mass, max_mass]; rtol=1e-4)

#   return out[1]
# end

function constant_coalescence_efficiency(x, y, val)
    return val
end

end #module Sources.jl
