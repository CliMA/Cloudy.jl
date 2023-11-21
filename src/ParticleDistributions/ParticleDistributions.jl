"""
  particle mass distribution module

Particle mass distribution functions for microphysical process modeling:
  - computation of moments of distributions
  - computation of densities of distributions
  - creating distributions given a set of parameters
  - creating distributions given a set of moments
"""
module ParticleDistributions

using SpecialFunctions: gamma, gamma_inc
using DocStringExtensions
using QuadGK

import NumericalIntegration as NI

# particle mass distributions available for microphysics
export AbstractParticleDistribution
export PrimitiveParticleDistribution
export ExponentialPrimitiveParticleDistribution
export GammaPrimitiveParticleDistribution
export MonodispersePrimitiveParticleDistribution
export LognormalPrimitiveParticleDistribution

# methods that query particle mass distributions
export moment
export get_moments
export density
export normed_density
export nparams
export update_dist_from_moments!
export moment_source_helper

# setters and getters
export get_params


"""
  AbstractParticleDistribution{FT}

A particle mass distribution function, which can be initialized
for various subtypes of assumed shapes in the microphysics parameterization.
"""
abstract type AbstractParticleDistribution{FT} end


"""
  PrimitiveParticleDistribution{FT}

A particle mass distribution that has support on the positive real
axis and analytic expressions for moments and partial moments.
"""
abstract type PrimitiveParticleDistribution{FT} <: AbstractParticleDistribution{FT} end


"""
  ExponentialPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents particle mass distribution function of exponential shape.

# Constructors
  ExponentialPrimitiveParticleDistribution(n::Real, θ::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct ExponentialPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "scale parameter"
  θ::FT

  function ExponentialPrimitiveParticleDistribution(n::FT, θ::FT) where {FT<:Real}
    if n < 0 || θ <= 0
      error("n needs to be nonnegative. θ needs to be positive.")
    end

    new{FT}(n, θ)
  end
end


"""
  GammaPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents particle mass distribution function of gamma shape.

# Constructors
  GammaPrimitiveParticleDistribution(n::Real, θ::Real, k::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct GammaPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "scale parameter"
  θ::FT
  "shape parameter"
  k::FT

  function GammaPrimitiveParticleDistribution(n::FT, θ::FT, k::FT) where {FT<:Real}
    if n < 0 || θ <= 0 || k <= 0
      error("n needs to be nonnegative. θ and k need to be positive.")
    end
    new{FT}(n, θ, k)
  end
end

"""
  MonodispersePrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents monodisperse particle size distribution function.

# Constructors
  MonodispersePrimitiveParticleSizeDistribution(n::Real, m::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct MonodispersePrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "particle diameter"
  θ::FT
  
  function MonodispersePrimitiveParticleDistribution(n::FT, θ::FT) where {FT<:Real}
    if n < 0 || θ <= 0
      error("n needs to be nonnegative. θ needs to be positive.")
    end
    new{FT}(n, θ)
  end
end

"""
  LognormalPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents lognormal particle size distribution function.

# Constructors
  LognormalPrimitiveParticleSizeDistribution(n::Real, μ::Real, σ::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct LognormalPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
  "normalization constant (e.g., droplet number concentration)"
  n::FT
  "logarithmic mean size"
  μ::FT
  "logarithmic standard deviation"
  σ::FT
  
  function LognormalPrimitiveParticleDistribution(n::FT, μ::FT, σ::FT) where {FT<:Real}
    if n < 0 || σ <= 0
      error("n needs to be nonnegative. σ needs to be positive.")
    end
    new{FT}(n, μ, σ)
  end
end

"""
  (pdist::ParticleDistribution{FT}(x::FT)

  - `x` - is an array of points to evaluate the density of `pdist` at
Returns the particle mass density evaluated at `x`.
"""
function (pdist::AbstractParticleDistribution{FT})(x::FT) where {FT<:Real}
  return density(pdist, x)
end

"""
  moment_func(dist)

  `dist` - particle mass distribution function
Returns a function that computes the moments of `dist`.
"""
function moment_func(dist::ExponentialPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # moment_of_dist = n * θ^q * Γ(q+1)
  function f(q)
    dist.n .* dist.θ.^q .* gamma.(q .+ FT(1))
  end
  return f
end

function moment_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # moment_of_dist = n * θ^q * Γ(q+k) / Γ(k)
  function f(q)
    dist.n .* dist.θ.^q .* gamma.(q .+ dist.k) / gamma.(dist.k)
  end
  return f
end

function moment_func(dist::MonodispersePrimitiveParticleDistribution{FT}) where {FT<:Real}
  # moment_of_dist = n * θ^(q)
  function f(q)
   dist.n .* dist.θ.^q
  end
  return f
end

function moment_func(dist::LognormalPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # moment_of_dist = n * exp(q * μ + 1/2 * q^2 * σ^2)
  function f(q)
    dist.n .* exp.(q .* dist.μ + q.^2 .* dist.σ.^2 ./ 2)
  end
  return f 
end

"""
  moment(dist, q)

  - `dist` - distribution of which the partial moment `q` is taken
  - `q` - is a potentially real-valued order of the moment
Returns the q-th moment of a particle mass distribution function.
"""
function moment(dist::AbstractParticleDistribution{FT}, q::FT) where {FT<:Real}
  moment_func(dist)(q)
end

"""
    get_moments(pdist::GammaParticleDistribution{FT})
Returns the first P (0, 1, 2) moments of the distribution where P is the innate
numer of prognostic moments
"""
function get_moments(pdist::GammaPrimitiveParticleDistribution{FT}) where {FT<:Real}
  return [pdist.n, pdist.n * pdist.k * pdist.θ, pdist.n*pdist.k*(pdist.k+1)*pdist.θ^2]
end

function get_moments(pdist::LognormalPrimitiveParticleDistribution{FT}) where {FT<:Real}
  return [pdist.n, pdist.n * exp(pdist.μ + pdist.σ^2 / 2), 
    pdist.n * exp(2.0 * pdist.μ + 2.0 * pdist.σ^2)]
end

function get_moments(pdist::ExponentialPrimitiveParticleDistribution{FT}) where {FT<:Real}
  return [pdist.n, pdist.n * pdist.θ]
end

function get_moments(pdist::MonodispersePrimitiveParticleDistribution{FT}) where {FT<:Real}
  return [pdist.n, pdist.n * pdist.θ]
end

"""
  density_func(dist)

  - `dist` - is a particle mass distribution
Returns the particle mass density function.
"""
function density_func(dist::ExponentialPrimitiveParticleDistribution{FT}) where {FT<:Real}
  function f(x)
    dist.n ./ dist.θ .* exp.(-x ./ dist.θ)
  end
  return f
end

function density_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # density = n / θ^k / Γ(k) * x^(k-1) * exp(-x/θ)
  function f(x)
    dist.n .* x.^(dist.k .- 1) ./ dist.θ.^dist.k ./ gamma.(dist.k) .* exp.(-x ./ dist.θ)
  end
  return f
end

function density_func(dist::LognormalPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # density = n * 1 / (x σ √2π) exp((-ln(x) - μ)^2 / 2σ^2 )
  function f(x)
    dist.n .* exp.(-(log.(x) - dist.μ).^2 ./ (2*dist.σ^2) ) ./ (x .* dist.σ .* sqrt(2 * π)) 
  end
  return f
end

function density_func(dist::MonodispersePrimitiveParticleDistribution{FT}) where {FT<:Real}
  # density = n δ(θ); here we return a rectangular pulse only for visualizations: n/(2Δx) * [H(x-θ+Δx) - H(x-θ-Δx)]
  # where 2Δx = 2θ/10 is the pulse width and H represents the heaviside function
  function f(x)
    return (abs(x-dist.θ) < dist.θ/10.0) ? dist.n / (2 * dist.θ / 10.0) : FT(0)
  end
  return f
end

"""
  normed_density_func(dist)

  - `dist` - is a particle mass distribution
Returns the normalized particle mass density function.
"""
function normed_density_func(dist::ExponentialPrimitiveParticleDistribution{FT}) where {FT<:Real}
  function f(x)
    1 ./ dist.θ .* exp.(-x ./ dist.θ)
  end
  return f
end

function normed_density_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # density = n / θ^k / Γ(k) * x^(k-1) * exp(-x/θ)
  function f(x)
    x.^(dist.k .- 1) ./ dist.θ.^dist.k ./ gamma.(dist.k) .* exp.(-x ./ dist.θ)
  end
  return f
end

function normed_density_func(dist::LognormalPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # density = n * 1 / (x σ √2π) exp((-ln(x) - μ)^2 / 2σ^2 )
  function f(x)
    exp.(-(log.(x) - dist.μ).^2 ./ (2*dist.σ^2) ) ./ (x .* dist.σ .* sqrt(2 * π)) 
  end
  return f
end

"""
  density(dist, x)

  - `dist` - is a particle mass distribution
  - `x` - is a point to evaluate the density of `dist` at
Returns the particle mass density evaluated at point `x`.
"""
function density(dist::AbstractParticleDistribution{FT}, x::FT) where {FT<:Real}
  if any(x .< zero(x))
    error("Density can only be evaluated at nonnegative values.")
  end
  density_func(dist)(x)
end

"""
  normed_density(dist, x)

  - `dist` - is a particle mass distribution
  - `x` - is a point to evaluate the density of `dist` at
Returns the particle normalized mass density evaluated at point `x`.
"""
function normed_density(dist::AbstractParticleDistribution{FT}, x::FT) where {FT<:Real}
  if any(x .< zero(x))
    error("Density can only be evaluated at nonnegative values.")
  end
  normed_density_func(dist)(x)
end


"""
  nparams(dist)

  - `dist` - is a particle mass distribution
Returns the number of settable parameters of dist.
"""
function nparams(dist::PrimitiveParticleDistribution{FT}) where {FT<:Real}
  length(propertynames(dist))
end

"""
  get_params(dist)

  - `dist` - is a particle mass distribution
Returns the names and values of settable parameters for a dist.
"""
function get_params(dist::PrimitiveParticleDistribution{FT}) where {FT<:Real}
  params = Array{Symbol, 1}(collect(propertynames(dist)))
  values = Array{FT, 1}([getproperty(dist, p) for p in params])
  return params, values
end

"""
  check_moment_consistency(m::Array{FT})

  - `m` - is an array of moments

Checks if moments are nonnegative and whether even-ordered central moments implied
by moments vector are all positive.
"""
function check_moment_consistency(m::Array{FT}) where {FT<:Real}
  n_mom = length(m)
  # check if  moments are nonnegative
  if any(m .< 0.0)
    error("all moments need to be nonnegative.")
  end

  # check if even-ordered central moments are positive (e.g., variance, etc.)
  # non-positivity  would be inconsistent with a well-defined distribution.
  for order in 2:2:n_mom-1
    cm = 0.0
    for i in 0:order
      cm += binomial(order, i) * (-1)^i * (m[2] / m[1])^i * (m[order-i+1] / m[1])
    end
    if cm < 0.0
      error("order-$order central moment needs to be nonnegative.")
    end
  end

  nothing
end

"""
    update_dist_from_moments!(pdist::GammaPrimitiveParticleDistribution{FT}, moments::Array{FT})

Updates parameters of the gamma distribution given the first three moments
"""
function update_dist_from_moments!(pdist::GammaPrimitiveParticleDistribution{FT}, moments::Array{FT}; param_range = Dict("θ" => (eps(FT), Inf), "k" => (eps(FT), Inf))) where {FT<:Real}
  @assert length(moments) == 3
  if moments[1] > eps(FT) && moments[2] > eps(FT) && moments[3] > eps(FT)
    pdist.k = max(param_range["k"][1], min(param_range["k"][2], (moments[2]/moments[1])/(moments[3]/moments[2]-moments[2]/moments[1])))
    pdist.θ = max(param_range["θ"][1], min(param_range["θ"][2], moments[3]/moments[2]-moments[2]/moments[1]))
    pdist.n = moments[2]/(pdist.k * pdist.θ)
  else #don't change θ and k
    pdist.n = FT(0)
  end
end

"""
    update_dist_from_moments!(pdist::LognormalPrimitiveParticleDistribution{FT}, moments::Array{FT})

Updates parameters of the lognormal distribution given the first three moments
"""
function update_dist_from_moments!(pdist::LognormalPrimitiveParticleDistribution{FT}, moments::Array{FT}; param_range = Dict("μ" => (-Inf, Inf), "σ" => (eps(FT), Inf))) where {FT<:Real}
  @assert length(moments) == 3
  if moments[1] > eps(FT) && moments[2] > eps(FT) && moments[3] > eps(FT)
    pdist.μ = max(param_range["μ"][1], min(param_range["μ"][2], log(moments[2]^2 / moments[1]^(3/2) / moments[3]^(1/2))))
    pdist.σ = max(param_range["σ"][1], min(param_range["σ"][2], sqrt(log(moments[1]*moments[3]/moments[2]^2))))
    pdist.n = moments[2]/exp(pdist.μ + 1/2 * pdist.σ^2)
  else #don't change μ and σ
    pdist.n = FT(0)
  end
end

"""
    update_dist_from_moments!(pdist::ExponentialPrimitiveParticleDistribution{FT}, moments::Array{FT})

Updates parameters of the exponential distribution given the first two moments
"""
function update_dist_from_moments!(pdist::ExponentialPrimitiveParticleDistribution{FT}, moments::Array{FT}; param_range = Dict("θ" => (eps(FT), Inf))) where {FT<:Real}
  @assert length(moments) == 2
  if moments[1] > eps(FT) && moments[2] > eps(FT)
    pdist.θ = max(param_range["θ"][1], min(param_range["θ"][2], moments[2]/moments[1]))
    pdist.n = moments[2] / pdist.θ
  else #don't change θ
    pdist.n = FT(0)
  end
end

"""
    update_dist_from_moments!(pdist::MonodispersePrimitiveParticleDistribution{FT}, moments::Array{FT})

Updates parameters of the monodisperse distribution given the first two moments
"""
function update_dist_from_moments!(pdist::MonodispersePrimitiveParticleDistribution{FT}, moments::Array{FT}; param_range = Dict("θ" => (eps(FT), Inf))) where {FT<:Real}
  @assert length(moments) == 2
  if moments[1] > eps(FT) && moments[2] > eps(FT)
    pdist.θ = max(param_range["θ"][1], min(param_range["θ"][2], moments[2]/moments[1]))
    pdist.n = moments[2] / pdist.θ
  else #don't change θ
    pdist.n = FT(0)
  end
end

"""
  moment_source_helper(dist, p1, p2, x_threshold)

  - `dist` - AbstractParticleDistribution
  - `p1` - power of particle mass
  - `p2` - power of particle mass
  - `x_threshold`- particle mass threshold

Returns ∫_0^x_threshold ∫_0^(x_threshold-x') x^p1 x'^p2 f(x) f(x') dx dx' for computations of the source of moments of the distribution below
the given threshold x_threshold. For MonodispersePrimitiveParticleDistribution The integral can be computed analytically: 
∫_0^x_threshold ∫_0^(x_threshold-x') x^p1 x'^p2 f(x) f(x') dx dx = n^2 * θ^(p1+p2) if θ < x_threshold/2, and equals zero otherwise. For
ExponentialPrimitiveParticleDistribution and GammaPrimitiveParticleDistribution the two-dimensional integral reduces to a one-dimensional 
integral over incomplete gamma functions.
"""
function moment_source_helper(dist::MonodispersePrimitiveParticleDistribution{FT}, p1::FT, p2::FT, x_threshold::FT) where {FT<:Real}
  n, θ = get_params(dist)[2]
  source = (θ < x_threshold/2) ? n^2 * θ^(p1+p2) : 0
  return source
end

function moment_source_helper(
  dist::ExponentialPrimitiveParticleDistribution{FT}, 
  p1::FT, 
  p2::FT, 
  x_threshold::FT; 
  x_lowerbound = 1e-5, 
  n_bins = 50
  ) where {FT<:Real}
  n, θ = get_params(dist)[2]

  f(x) = x^p1 * exp(-x/θ) * gamma_inc(p2 + 1, (x_threshold - x)/θ)[1] * gamma(p2 + 1)
  
  logx = range(log(x_lowerbound), log(x_threshold), n_bins+1)
  x = exp.(logx)
  y = [x[1:end-1] .* f.(x[1:end-1]); FT(0)]

  return n^2 * θ^(p2 - 1) * NI.integrate(logx, y, NI.SimpsonEvenFast())
end

function moment_source_helper(
  dist::GammaPrimitiveParticleDistribution{FT}, 
  p1::FT, 
  p2::FT, 
  x_threshold::FT; 
  x_lowerbound = 1e-5, 
  n_bins = 50
  ) where {FT<:Real}
  n, θ, k = get_params(dist)[2]

  f(x) = x^(p1 + k - 1) * exp(-x/θ) * gamma_inc(p2 + k, (x_threshold - x)/θ)[1] * gamma(p2 + k)

  logx = range(log(x_lowerbound), log(x_threshold), n_bins+1)
  x = exp.(logx)
  y = [x[1:end-1] .* f.(x[1:end-1]); FT(0)]

  return n^2 * θ^(p2 - k) / gamma(k)^2 * NI.integrate(logx, y, NI.SimpsonEvenFast())
end

function moment_source_helper(
  dist::LognormalPrimitiveParticleDistribution{FT}, 
  p1::FT, 
  p2::FT, 
  x_threshold::FT; 
  x_lowerbound = 1e-5, 
  n_bins = 50,
  ) where {FT<:Real}

  g(x, y) = x.^(p1) .* y.^(p2) .* density_func(dist).(x) .* density_func(dist).(y)
  f(y) = quadgk(xx -> g(xx,y), x_lowerbound, x_threshold - y)[1]
  
  return quadgk(f, x_lowerbound, x_threshold)[1]
end

end #module ParticleDistributions.jl
