"""
  Microphysical source functions
  TODO: Figure out a better way of setting integral limits
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
using Distributions: pdf
# methods that compute source terms from microphysical parameterizations
# export get_coalescence_integrals
# export get_coalescence_integral_moment
# export constant_coalescence_efficiency
export get_coalescence_integral_moment_qrs!

FT = Float64

function weighting_fn(x, k, pdists)
  denom = 0.0
  num = 0.0
  for j=1:length(pdists)
    denom += pdf(pdists[k].dist, x)
    if j<= k
      num += pdf(pdists[k].dist, x)
    end
  end
  if denom == 0.0
    return 0.0
  else
    return num / denom
  end
end

function q_integrand_inner(x, y, j, k, kernel, pdists)
  integrand = 0.5 * kernel(x - y, y) * (pdists[j](x-y) * pdists[k](y) + pdists[k](x-y) * pdists[j](y))
  return integrand
end

function q_integrand_outer(x, j, k, kernel, pdists, moment_order)
  outer = x.^moment_order * quadgk(yy -> q_integrand_inner(x, yy, j, k, kernel, pdists), 0.0, x)[1]
  return outer
end

function r_integrand(x, y, j, k, kernel, pdists, moment_order)
  integrand = x.^moment_order * kernel(x, y) * pdists[j](x) * pdists[k](y)
  return integrand
end

function s_integrand_inner(x, k, kernel, pdists, moment_order)
  integrand_inner = y -> 0.5 * kernel(x - y, y) * pdists[k](x-y) * pdists[k](y)
  integrand_outer = x.^moment_order * quadgk(yy -> integrand_inner(yy), 0.0, x)[1]
  return integrand_outer
end

function s_integrand1(x, k, kernel, pdists, moment_order)
  integrandj = weighting_fn(x, k, pdists) * s_integrand_inner(x, k, kernel, pdists, moment_order)
  return integrandj
end

function s_integrand2(x, k, kernel, pdists, moment_order)
  integrandk = (1 - weighting_fn(x, k, pdists)) * s_integrand_inner(x, k, kernel, pdists, moment_order)
  return integrandk
end

"""
get_coalescence_integral_moment_qrs(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int)

Returns the collision-coalescence integral at points `x`.
Q: source term to particle k via collisions with smaller particles j
"""
function get_coalescence_integral_moment_qrs!(
  moment_order, kernel, pdists, Q, R, S)

  Ndist = length(pdists)
  
  for j in 1:Ndist
    max_mass = ParticleDistributions.max_mass(pdists[j])
    s1 = x -> s_integrand1(x, j, kernel, pdists, moment_order)
    s2 = x -> s_integrand2(x, j, kernel, pdists, moment_order)
    S[j,1] = quadgk(s1, 0.0, max_mass; rtol=1e-8)[1]
    S[j,2] = quadgk(s2, 0.0, max_mass; rtol=1e-8)[1]
    for k in 1:Ndist
      max_mass = max(ParticleDistributions.max_mass(pdists[j]), ParticleDistributions.max_mass(pdists[k]))
      R[j,k] = hcubature(xy -> r_integrand(xy[1], xy[2], j, k, kernel, pdists, moment_order), [0.0, 0.0], [max_mass, max_mass]; rtol=1e-8, maxevals=1000)[1]
      if j < k
        Q[j,k] = quadgk(x -> q_integrand_outer(x, j, k, kernel, pdists, moment_order), 0.0, max_mass; rtol=1e-8)[1]
      else
        Q[j,k] = 0.0
      end
    end
  end
end

# TODO: do we need a sorting function?

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
