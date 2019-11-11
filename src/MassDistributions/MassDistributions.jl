"""
  particle mass distribution module

Particle mass distribution functions for microphysical process modeling:
  - computation of moments of distributions
  - computation of densities of distributions
  - updating of distribution parameters given a set of moments
"""
module MassDistributions

using SpecialFunctions: gamma

# mass distributions available for microphysics
export MassDistributionFunction
export Gamma
export Exponential

# methods that query particle mass distributions
export compute_moment
export compute_density

# methods that update mass distributions
export update_params!


"""
  MassDistributionFunction{FT}

A particle mass distribution function, which can be initialized
for various subtypes of assumed shapes in the microphysics parameterizations.
"""
abstract type MassDistributionFunction{FT} end


"""
  Gamma{FT} <: MassDistributionFunction

Represents particle mass distribution function of gamma shape.

# Constructors
  Gamma(n::Real, θ::Real, k::Real)

# Fields

"""
mutable struct Gamma{FT} <: MassDistributionFunction{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "scale parameter"
  θ::FT
  "shape parameter"
  k::FT

  function Gamma(n::FT, θ::FT, k::FT) where {FT<:Real}
    if n < 0 || θ <= 0 || k <= 0
      error("n needs to be nonnegative. θ and k need to be positive.")
    end
    new{FT}(n, θ, k)
  end
end


"""
  Exponential{FT}

Represents particle mass distribution function of exponential shape.

# Constructors
  Exponential(n::Real, θ::Real)

# Fields

"""

mutable struct Exponential{FT} <: MassDistributionFunction{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "scale parameter"
  θ::FT

  function Exponential(n::FT, θ::FT) where {FT<:Real}
    if n < 0 || θ <= 0
      error("n needs to be nonnegative. θ needs to be positive.")
    end
    new{FT}(n, θ)
  end
end


"""
  compute_moment(dist, q)

  - `dist` - distribution of which the moment `q` is taken
  - `q` - is the potentially real-valued order of the moment
Returns the q-th moment of a particle mass distribution function.
"""
function compute_moment(dist::Gamma, q::FT) where {FT<:Real}
  # moment_of_dist = n * θ^q * Γ(k+q) / Γ(k)
  dist.n * dist.θ.^q * gamma(dist.k + q) / gamma(dist.k)
end

function compute_moment(dist::Exponential, q::FT) where {FT<:Real}
  # moment_of_dist = n * θ^q * Γ(q+1)
  dist.n * dist.θ.^q * gamma(q + 1)
end


"""
  compute_density(dist, x)

  - `dist` - is the particle mass distribution
  - `x` - is the point to evaluate the density of `dist` at
Returns the particle mass density evaluated at point `x`.
"""
function compute_density(dist::Gamma{FT}, x::FT) where {FT<:Real}
  # moment_of_dist = n / θ^k / Γ(k) * x^(k-1) * exp(-x/θ)
  if x < 0
    error("Density can only be evaluated at positive values.")
  end
  dist.n * x^(dist.k - 1) / dist.θ^dist.k / gamma(dist.k) * exp(-x / dist.θ)
end

function compute_density(dist::Exponential{FT}, x::FT) where {FT<:Real}
  # moment_of_dist = n / θ * exp(-x/θ)
  if x < 0
    error("Density can only be evaluated at positive values.")
  end
  dist.n / dist.θ * exp(-x / dist.θ)
end


"""
  update_params(dist, mom)

  - `dist` - is a mass distirbution function
  - `mom` - is an array of moments
Updates the internal parameters of the mass distribution function `dist`.
"""
function update_params!(dist::Gamma{FT}, mom::Array{FT}) where {FT <: Real}
  if length(mom) != length(fieldnames(typeof(dist)))
    error("Number of moments must be consistent with distribution type.")
  end
  if mom[1] < 0 || mom[2] <= 0 || (mom[3] <= 0)
    error("0th moment needs to be >=0. 1st & 2nd moments need to >0.")
    end
  if mom[3] / mom[1] - (mom[2] / mom[1])^2 <= 0
    # Check for M_2 / M_0 - (M_1 / M_0)^2 >= 0
    error("Variance implied by moments needs to be nonnegative.")
  end
  # n = M_0
  dist.n = mom[1]

  # θ = (M_2 M_0 - M_1^2) / M_1 / M_0
  dist.θ = (mom[3] * mom[1] - mom[2]^2) / mom[2] / mom[1]

  # k = M_1^2 / (M_2 M_0 - M_1^2)
  dist.k = mom[2]^2 / (mom[3] * mom[1] - mom[2]^2)

  nothing
end

function update_params!(dist::Exponential{FT}, mom::Array{FT}) where {FT <: Real}
  if length(mom) != length(fieldnames(typeof(dist)))
    error("Number of moments must match number of distribution parameters.")
  end
  if mom[1] < 0 || mom[2] <= 0
    error("0th moment needs to be >=0. 1st moment needs to >0.")
  end
  # n = M_0
  dist.n = mom[1]

  # θ = M_1 / M_0
  dist.θ = mom[2] / mom[1]

  nothing
end

end #module MassDistributions.jl
