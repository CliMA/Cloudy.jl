"""
  Microphysical source functions

"""
module MultiParticleSources

using QuadGK
using HCubature
#using NIntegration
#using NLsolve
#using Distributions
using SpecialFunctions: gamma
using ..ParticleDistributions
using ..KernelFunctions
# methods that compute source terms from microphysical parameterizations
# export get_coalescence_integrals
# export get_coalescence_integral_moment
# export constant_coalescence_efficiency
export get_coalescence_integral_moment_qrs

"""
get_coalescence_integral(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int)

Returns the collision-coalescence integral at points `x`.
"""
function get_coalescence_integral_moment_qrs(
  moment::Int, kernel::KernelFunction{FT}, pdists::Array{PD,1}
  ) where {FT<:Real, PD<:ParticleDistribution{FT}}

  function weighting_fn(x, pdist1, pdist2)
    return pdist1(x) / (pdist1(x) + pdist2(x))
  end

  function q_integrand_inner(x, y, j, k)
    integrand = 0.5 * kernel(x - y, y) * pdists[j](x-y) * pdists[k](y)
    return integrand
  end

  function q_integrand_outer(x, j, k)
    outer = x^moment * quadgk(yy -> q_integrand_inner(x, yy, j, k), 0.0, x; rtol=1e-4)[1]
    return outer[0]
  end

  function r_integrand(xy, j, k)
    integrand = xy[1]^moment * kernel(xy[1], xy[2]) * pdists[j](xy[1]) * pdists[k](xy[2])
    return integrand
  end

  function s_integrands(x, j, k)
    integrandj = weighting_fn(x, pdists[j], pdists[k]) * q_integrand_outer(x, j, k)
    integrandk = (1 - weighting_fn(x, pdists[j], pdists[k])) * q_integrand_outer(x, j, k)
    if k > j
      (integrandj, integrandk) = (integrandk, integrandj)
    end
    print(integrandj, integrandk)
    return (integrandj, integrandk)
  end
  
  Ndist = length(pdists)
  Q = zeros((Ndist, Ndist))
  R = zeros((Ndist, Ndist))
  S = zeros((Ndist, 2))
  
  for j in 1:Ndist
    max_mass = ParticleDistributions.moment(pdists[j+1], 1.0)
    if j < Ndist
      s = x -> s_integrands(x, j, j+1)
      S[j,1] = quadgk(x -> s[1](x), 0.0, max_mass; rtol=1e-4)[1]
      S[j,2] = quadgk(x -> s[2](x), 0.0, max_mass; rtol=1e-4)[1]
    end
    for k in j+1:Ndist
      max_mass = ParticleDistributions.moment(pdists[k], 1.0)
      Q[j,k] = quadgk(x -> q_integrand_outer(x, j, k), 0.0, max_mass; rtol=1e-4)[1]
      R[j,k] = hcubature(xy -> r_integrand(xy, j, k), [0.0, 0.0], [max_mass, max_mass]; rtol=1e-8, maxevals=1000)[0]
    end
  end

  return (Q, R, S)
end
# function get_coalescence_integrals(x::Array{FT,2}, kernel::KernelFunction{FT}, 
#   pdists::Array{PD,1}, n_samples::Int) where {FT<:Real, PD<:ParticleDistribution{FT}}
#   out_plus = zeros((size(x)[2], length(pdists)))
#   out_minus = zeros((size(x)[2], length(pdists)))
#   for p in 1:length(pdists)
#     # monte carlo samples
#     y = sort(ParticleDistributions.sample(pdists[p], n_samples))

#     # source contribution to collision integral
#     for (i, xx) in enumerate(x[p,:])
#       for yy in  y[y .<= xx]
#         out_plus[i, p] += 0.5 * pdists[p](xx - yy) * kernel(xx - yy, yy)
#       end
#     end

#     # sink contribution to collision integral
#     out_minus[:,p] += pdists[p].(x[p,:]) .* sum(kernel.(x[p,:], y'), dims=2)

#     # normalize to get correct monte carlo average
#     out_plus[:,p] *= pdists[p].n / length(y) 
#     out_minus[:,p] *= pdists[p].n / length(y) 
#   end
#   return (out_plus, out_minus)
# end

# """
# get_coalescence_integral(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int)

# Returns the collision-coalescence integral at points `x`.
# """
# function get_coalescence_integrals(x::Array{FT,1}, kernel::KernelFunction{FT}, 
#   pdists::Array{PD,1}, n_samples::Int) where {FT<:Real, PD<:ParticleDistribution{FT}}
#   out_plus = zeros(length(pdists))
#   out_minus = zeros(length(pdists))
#   for p in 1:length(pdists)
#     # monte carlo samples
#     y = sort(ParticleDistributions.sample(pdists[p], n_samples))

#     # source contribution to collision integral
#     for (i, xx) in enumerate(x)
#       for yy in  y[y .<= xx]
#         out_plus[p] += 0.5 * pdists[p](xx - yy) * kernel(xx - yy, yy)
#       end
#     end

#     # sink contribution to collision integral
#     out_minus[p] += (pdists[p].(x)' * sum(kernel.(x, y'), dims=2))[1][1]

#     # normalize to get correct monte carlo average
#     out_plus[p] *= pdists[p].n / length(y) 
#     out_minus[p] *= pdists[p].n / length(y) 
#   end
#   return (out_plus, out_minus)
# end


# """
# get_coalescence_integral_moment(
#     k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
#     n_samples_mc::Int, coalescence_efficiency::Function)

# Returns the collision-coalescence integral at moment of order `k`, using
# Monte Carlo integration
# """
# function get_coalescence_integral_moment(
#     k::Int, kernel::KernelFunction{FT}, pdists::Array{PD,1},
#     n_samples_mc::Int
#     ) where {FT<:Real, PD<:ParticleDistribution{FT}}

#   mom_out_plus = zeros((length(pdists), length(pdists)))
#   mom_out_minus = zeros((length(pdists), length(pdists)))

#   for p1 in 1:length(pdists)
#     # monte carlo samples for inner integral
#     y = sort(sample(pdists[p1], n_samples_mc))
#     (out_plus, out_minus) = get_coalescence_integrals(y, kernel, pdists, n_samples_mc)
#     print(out_minus, out_plus)
#     # TODO: add a weighting function
#     for p2 in 1:length(pdists)
#         x = sort(sample(pdists[p2], n_samples_mc))
#         # collision integrals for the k-th moment
#         for i in 1:n_samples_mc
#           mom_out_plus[p1, p2] += out_plus[i] * pdists[p2].(x[i]) * x[i]^k
#           mom_out_minus[p1, p2] += out_minus[i] * pdists[p2].(x[i]) * x[i]^k
#         end

#         # normalize to get correct monte carlo average
#         mom_out_plus[p1, p2] *= (pdists[p1].n * pdists[p2].n) / n_samples_mc
#         mom_out_minus[p1, p2] *= (pdists[p1].n * pdists[p2].n) / n_samples_mc
#     end
#   end

#   return (mom_out_plus, mom_out_minus)
# end

end #module MultiParticleSources.jl
