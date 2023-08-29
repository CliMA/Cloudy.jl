"""
  particle mass distribution module

Particle mass distribution functions for microphysical process modeling:
  - computation of densities of distributions
  - sampling from densities
"""
module SuperParticleDistributions

using Distributions: Distribution, Gamma, Exponential, pdf
using DocStringExtensions
using SpecialFunctions: gamma

# particle mass distributions available for microphysics
export ParticleDistribution
export GammaParticleDistribution
export ExponentialParticleDistribution

# methods that query particle mass distributions
export get_moments
export update_dist_from_moments!

"""
  ParticleDistribution{FT}

A particle mass distribution function, which can be initialized
for various subtypes of assumed shapes in the microphysics parameterization.
"""
abstract type ParticleDistribution{FT} end


"""
  GammaParticleDistribution{FT} <: ParticleDistribution{FT}

Represents particle mass distribution function of gamma shape.

# Constructors
  GammaParticleDistribution(n::Real, θ::Real, k::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct GammaParticleDistribution{FT, D <: Distribution} <: ParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "shape parameter"
  k::FT
  "scale parameter"
  θ::FT
  "underlying probability distribution"
  dist::D

  function GammaParticleDistribution(n::FT, k::FT, θ::FT) where {FT<:Real}
    if n < 0
        throw(DomainError(n, "n must be nonnegative"))
    end
    dist = Gamma(k, θ)
    return new{FT, typeof(dist)}(n, k, θ, dist)
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
mutable struct ExponentialParticleDistribution{FT, D <: Distribution} <: ParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "scale parameter"
  θ::FT
  "underlying probability distribution"
  dist::D

  function ExponentialParticleDistribution(n::FT, θ::FT) where {FT<:Real}
    if n < 0
      throw(DomainError(n, "n must be nonnegative"))
    end
    dist = Exponential(θ)
    return new{FT, typeof(dist)}(n, θ, dist)
  end
end

"""
  (pdist::ParticleDistribution{FT}(x::FT)

  - `x` - is an array of points to evaluate the density of `pdist` at
Returns the particle mass density evaluated at `x`.
"""
function (pdist::ParticleDistribution{FT})(x::FT) where {FT<:Real}
  return FT(pdist.n * pdf(pdist.dist, x))
end

"""
    moments(pdist::GammaParticleDistribution{FT})
Returns the first three (0, 1, 2) moments of the distribution
"""
function get_moments(pdist::GammaParticleDistribution{FT}) where {FT<:Real}
  return [pdist.n, pdist.n * pdist.k * pdist.θ, pdist.n*pdist.k*(pdist.k+1)*pdist.θ^2]
end

"""
    moments(pdist::ExponentialParticleDistribution{FT})
Returns the first two (0, 1) moments of the distribution
"""
function get_moments(pdist::ExponentialParticleDistribution{FT}) where {FT<:Real}
  return [pdist.n, pdist.n * pdist.θ]
end

"""
    update_dist_from_moments!(pdist::GammaParticleDistribution{FT}, moments::Array{FT})

Updates parameters of the gamma distribution given the first three moments
"""
function update_dist_from_moments!(pdist::GammaParticleDistribution{FT}, moments::Array{FT}) where {FT<:Real}
  if length(moments) != 3
    throw(ArgumentError("must specify exactly 3 moments for gamma distribution"))
  end
  pdist.n = moments[1]
  pdist.k = (moments[2]/moments[1])/(moments[3]/moments[2]-moments[2]/moments[1])
  pdist.θ = moments[3]/moments[2]-moments[2]/moments[1]
  pdist.dist = Gamma(pdist.k, pdist.θ)
end

"""
    update_dist_from_moments!(pdist::ExponentialParticleDistribution{FT}, moments::Array{FT})

Updates parameters of the gamma distribution given the first three moments
"""
function update_dist_from_moments!(pdist::ExponentialParticleDistribution{FT}, moments::Array{FT}) where {FT<:Real}
  if length(moments) != 2
    throw(ArgumentError("must specify exactly 2 moments for exponential distribution"))
  end
  pdist.n = moments[1]
  pdist.θ = moments[2]/moments[1]
  pdist.dist = Exponential(pdist.θ)
end

end