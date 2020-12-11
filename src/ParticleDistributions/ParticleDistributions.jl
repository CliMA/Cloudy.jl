"""
  particle mass distribution module

Particle mass distribution functions for microphysical process modeling:
  - computation of densities of distributions
  - sampling from densities
"""
module ParticleDistributions

using Distributions: Distribution, Gamma, Exponential, MixtureModel, pdf, components
using ForwardDiff
using DocStringExtensions
using SpecialFunctions: polygamma, gamma_inc, gamma
using Random: rand

# particle mass distributions available for microphysics
export ParticleDistribution
export GammaParticleDistribution
export ExponentialParticleDistribution
export AdditiveGammaParticleDistribution
export AdditiveExponentialParticleDistribution

# methods that query particle mass distributions
export sample
export moment
export density_gradient
export normal_mass_constraint

"""
  ParticleDistribution{FT}

A particle mass distribution function, which can be initialized
for various subtypes of assumed shapes in the microphysics parameterization.
"""
abstract type ParticleDistribution{FT} end


"""
  NormalizingFlow{FT} <: ParticleDistribution{FT}

Represents a normalizing flow.

# Constructors
  NormalizingFlow(n::Real, dist::Distribution, trafo)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct NormalizingFlow{FT} <: ParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "underlying probability distribution"
  dist::Distribution
  "Flow transformation"
  trafo
end


"""
  GammaParticleDistribution{FT} <: ParticleDistribution{FT}

Represents particle mass distribution function of gamma shape.

# Constructors
  GammaParticleDistribution(n::Real, θ::Real, k::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GammaParticleDistribution{FT} <: ParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "shape parameter"
  k::FT
  "scale parameter"
  θ::FT
  "underlying probability distribution"
  dist::Distribution

  function GammaParticleDistribution(n::FT, k::FT, θ::FT) where {FT<:Real}
    dist = Gamma(k, θ)
    new{FT}(n, k, θ, dist)
  end
end


"""
  ExponentialParticleDistribution{FT} <: ParticleDistribution{FT}

Represents particle mass distribution function of exponential shape.

# Constructors
  ExponentialParticleDistribution(n::Real, θ::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ExponentialParticleDistribution{FT} <: ParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "scale parameter"
  θ::FT
  "underlying probability distribution"
  dist::Distribution

  function ExponentialParticleDistribution(n::FT, θ::FT) where {FT<:Real}
    dist = Exponential(θ)
    new{FT}(n, θ, dist)
  end
end


"""
  AdditiveGammaParticleDistribution{FT} <: ParticleDistribution{FT}

Represents particle mass distribution function of gamma shape.

# Constructors
  AdditiveGammaParticleDistribution(n1::Real, n2::Real, θ1::Real, θ2::Real, k1::Real, k2::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AdditiveGammaParticleDistribution{FT} <: ParticleDistribution{FT}
  "total normalization constant (e.g., total droplet number concentration)"
  n::FT
  "1st normalization constant (e.g., droplet number concentration in 1st dist)"
  n1::FT
  "2nd normalization constant (e.g., droplet number concentration in 2nd dist)"
  n2::FT
  "1st shape parameter"
  k1::FT
  "2nd shape parameter"
  k2::FT
  "1st scale parameter"
  θ1::FT
  "2nd scale parameter"
  θ2::FT
  "underlying probability distribution"
  dist::Distribution

  function AdditiveGammaParticleDistribution(n1::FT, n2::FT, k1::FT, k2::FT, θ1::FT, θ2::FT) where {FT<:Real}
    dist = MixtureModel([Gamma(k1, θ1), Gamma(k2, θ2)], [n1/(n1 + n2), n2/(n1 + n2)])
    new{FT}(n1 + n2, n1, n2, k1, k2, θ1, θ2, dist)
  end
end


"""
  AdditiveExponentialParticleDistribution{FT} <: ParticleDistribution{FT}

Represents particle mass distribution function of exponential shape.

# Constructors
  AdditiveExponentialParticleDistribution(n1::Real, n2::Real, θ1::Real, θ2::Real, k1::Real, k2::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AdditiveExponentialParticleDistribution{FT} <: ParticleDistribution{FT}
  "total normalization constant (e.g., total droplet number concentration)"
  n::FT
  "1st normalization constant (e.g., droplet number concentration in 1st dist)"
  n1::FT
  "2nd normalization constant (e.g., droplet number concentration in 2nd dist)"
  n2::FT
  "1st scale parameter"
  θ1::FT
  "2nd scale parameter"
  θ2::FT
  "underlying probability distribution"
  dist::Distribution

  function AdditiveExponentialParticleDistribution(n1::FT, n2::FT, θ1::FT, θ2::FT) where {FT<:Real}
    dist = MixtureModel([Exponential(θ1), Exponential(θ2)], [n1/(n1 + n2), n2/(n1 + n2)])
    new{FT}(n1 + n2, n1, n2, θ1, θ2, dist)
  end
end


"""
  (pdist::ParticleDistribution{FT}(x::FT)

  - `x` - is an array of points to evaluate the density of `pdist` at
Returns the particle mass density evaluated at `x`.
"""
function (pdist::ParticleDistribution{FT})(x::FT) where {FT<:Real}
  return pdist.n * pdf.(pdist.dist, x)
end


"""
  sample(pdist::ParticleDistribution{FT}, n_samples::Int)

  - `pdist` - is a particle mass distribution
  - `n_samples` - is the number of samples to be drawn
Returns samples from the implied probability distribution.
"""
function sample(pdist::ParticleDistribution{FT}, n_samples::Int) where {FT<:Real}
  return rand(pdist.dist, n_samples)
end


"""
  moment(pdist::ParticleDistribution{FT}, q::Int, a::FT, b::FT)

  - `pdist` - is a particle mass distribution
  - `q` - is the order of the moment
  - `a` - is the lower boundary for the incomplete integral
  - `b` - is the lower boundary for the incomplete integral
Returns the incomplete moment.
"""
function moment(pdist::GammaParticleDistribution{FT}, q::Int, a::FT, b::FT) where {FT<:Real}
  n = pdist.n
  k = pdist.k
  θ = pdist.θ
  
  g1, __ = gamma_inc(k+q, a/θ, 0)
  __, g2 = gamma_inc(k+q, b/θ, 0)

  return n*θ^q*gamma(k+q)/gamma(k)*(1 - g1 - g2)
end

function moment(pdist::ExponentialParticleDistribution{FT}, q::Int, a::FT, b::FT) where {FT<:Real}
  n = pdist.n
  θ = pdist.θ
  
  g1, __ = gamma_inc(1+q, a/θ, 0)
  __, g2 = gamma_inc(1+q, b/θ, 0)

  return n*θ^q*gamma(1+q)*(1 - g1 - g2)
end


"""
  density_gradient(pdist::ParticleDistribution{FT}, x::Array{FT})

  - `pdist` - is a particle mass distribution
  - `x` - is an array of points to evaluate the gradient of `pdist` at
Returns the gradient of `pdist` wrt its parameters.
"""
function density_gradient(pdist::GammaParticleDistribution{FT}, x::Array{FT}) where {FT<:Real}
  out = zeros(length(x), 3)
  dens_val = pdist.(x)
  for (i, xx) in enumerate(x)
    out[i,1] = 1 / pdist.n
    out[i,2] = log(xx/pdist.θ) - polygamma(0, pdist.k)
    out[i,3] = (xx / pdist.θ - pdist.k) / pdist.θ  
  end
  out[:,1] .*= dens_val
  out[:,2] .*= dens_val
  out[:,3] .*= dens_val
  return out
end

function density_gradient(pdist::ExponentialParticleDistribution{FT}, x::Array{FT}) where {FT<:Real}
  out = zeros(length(x), 2)
  dens_val = pdist.(x)
  for (i, xx) in enumerate(x)
    out[i,1] = 1 / pdist.n
    out[i,2] = (xx / pdist.θ - 1) / pdist.θ 
  end
  out[:,1] .*= dens_val
  out[:,2] .*= dens_val
  return out
end

function density_gradient(pdist::AdditiveGammaParticleDistribution{FT}, x::Array{FT}) where {FT<:Real}
  out = zeros(length(x), 6)
  dens_val1 = pdist.n1 * pdf.(components(pdist.dist)[1], x)
  dens_val2 = pdist.n2 * pdf.(components(pdist.dist)[2], x)
  for (i, xx) in enumerate(x)
    out[i,1] = 1 / pdist.n1
    out[i,2] = 1 / pdist.n2
    out[i,3] = log(xx/pdist.θ1) - polygamma(0, pdist.k1)
    out[i,4] = log(xx/pdist.θ2) - polygamma(0, pdist.k2)
    out[i,5] = (xx / pdist.θ1 - pdist.k1) / pdist.θ1
    out[i,6] = (xx / pdist.θ2 - pdist.k2) / pdist.θ2  
  end
  out[:,1] .*= dens_val1
  out[:,2] .*= dens_val2
  out[:,3] .*= dens_val1
  out[:,4] .*= dens_val2
  out[:,5] .*= dens_val1
  out[:,6] .*= dens_val2
  return out
end


function density_gradient(pdist::AdditiveExponentialParticleDistribution{FT}, x::Array{FT}) where {FT<:Real}
  out = zeros(length(x), 4)
  dens_val1 = pdist.n1 * pdf.(components(pdist.dist)[1], x)
  dens_val2 = pdist.n2 * pdf.(components(pdist.dist)[2], x)
  for (i, xx) in enumerate(x)
    out[i,1] = 1 / pdist.n1
    out[i,2] = 1 / pdist.n2
    out[i,3] = (xx / pdist.θ1 - 1.0) / pdist.θ1
    out[i,4] = (xx / pdist.θ2 - 1.0) / pdist.θ2  
  end
  out[:,1] .*= dens_val1
  out[:,2] .*= dens_val2
  out[:,3] .*= dens_val1
  out[:,4] .*= dens_val2
  return out
end

"""
  normal_mass_costraint(pdist::ParticleDistribution{FT})

  - `pdist` - is a particle mass distribution
Returns an unnormalized normal vector wrt to the mass conservation constraint.
"""
function normal_mass_constraint(pdist::GammaParticleDistribution{FT}) where {FT<:Real}
  return [pdist.k*pdist.θ, pdist.n*pdist.θ, pdist.k*pdist.n]
end

function normal_mass_constraint(pdist::ExponentialParticleDistribution{FT}) where {FT<:Real}
  return [pdist.θ, pdist.n]
end

function normal_mass_constraint(pdist::AdditiveGammaParticleDistribution{FT}) where {FT<:Real}
  return [pdist.k1*pdist.θ1, pdist.k2*pdist.θ2, pdist.n1*pdist.θ1, pdist.n2*pdist.θ2, pdist.k1*pdist.n1, pdist.k2*pdist.n2]
end

function normal_mass_constraint(pdist::AdditiveExponentialParticleDistribution{FT}) where {FT<:Real}
  return [pdist.θ1, pdist.θ2, pdist.n1, pdist.n2]
end

end #module ParticleDistributions.jl