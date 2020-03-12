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

import LinearAlgebra: norm
import Optim: optimize, LBFGS

# particle mass distributions available for microphysics
export ParticleDistribution
export Primitive
export Exponential
export Gamma
export Mixture

# methods that query particle mass distributions
export moment
export density

# setters and getters
export update_params_from_moments


"""
  ParticleDistribution{FT}

A particle mass distribution function, which can be initialized
for various subtypes of assumed shapes in the microphysics parameterization.
"""
abstract type ParticleDistribution{FT} end


"""
  Primitive{FT}

A particle mass distribution that has support on the positive real
axis and analytic expressions for moments and partial moments.
"""
abstract type Primitive{FT} <: ParticleDistribution{FT} end


"""
  Exponential{FT} <: Primitive{FT}

Represents particle mass distribution function of exponential shape.

# Constructors
  Exponential(n::Real, θ::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Exponential{FT} <: Primitive{FT}
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
  Gamma{FT} <: Primitive{FT}

Represents particle mass distribution function of gamma shape.

# Constructors
  Gamma(n::Real, θ::Real, k::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Gamma{FT} <: Primitive{FT}
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
  Mixture{FT} <: ParticleDistribution{FT}

A particle mass distribution function that is a mixture of
primitive or truncated subdistribution functions.

# Constructors
  Mixture(dists::ParticleDistribution{Real}...)
  Mixture(dist_arr::Array{ParticleDistribution{FT}})

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Mixture{FT} <: ParticleDistribution{FT}
  "array of distributions"
  subdists::Array{ParticleDistribution{FT}}

  function Mixture(dists::ParticleDistribution{FT}...) where {FT<:Real}
    if length(dists) < 2
      error("need at least two subdistributions to form a mixture.")
    end

    new{FT}(collect(dists))
  end
end

function Mixture(dist_arr::Array{ParticleDistribution{FT}}) where {FT<:Real}
  Mixture(dist_arr...)
end


"""
  moment_func(dist)

  `dist` - particle mass distribution function
Returns a function that computes the moments of `dist`.
"""
function moment_func(dist::Exponential{FT}) where {FT<:Real}
  # moment_of_dist = n * θ^q * Γ(q+1)
  # can reuse the moments of gamma distribution

  moment_func_gamma = moment_func(Gamma(FT(1), FT(1), FT(1)))
  function f(n, θ, q)
    moment_func_gamma(n, θ, FT(1), q)
  end
  return f
end

function moment_func(dist::Gamma{FT}) where {FT<:Real}
  # moment_of_dist = n * θ^q * Γ(q+k) / Γ(k)
  function f(n, θ, k, q)
    n .* θ.^q .* gamma.(q .+ k) / gamma.(k)
  end
  return f
end

function moment_func(dist::Mixture{FT}) where {FT<:Real}
  # mixture moment is sum of moments
  num_pars = [nparams(d) for d in dist.subdists]
  mom_funcs = [moment_func(d) for d in dist.subdists]
  function f(params...)
    inputs = collect(params)
    dist_params = inputs[1:end-1]
    q = inputs[end]
    i = 1
    output = zero(q)
    for (n, mom_func) in zip(num_pars, mom_funcs)
      output += mom_func(dist_params[i:i+n-1]..., q)
      i += n
    end
    return output
  end
  return f
end


"""
  moment(dist, q)

  - `dist` - distribution of which the partial moment `q` is taken
  - `q` - is a potentially real-valued order of the moment
Returns the q-th moment of a particle mass distribution function.
"""
function moment(dist::ParticleDistribution{FT}, q::FT) where {FT<:Real}
  moment_func(dist)(reduce(vcat, get_params(dist)[2])..., q)
end


"""
  density_func(dist)

  - `dist` - is a particle mass distribution
Returns the particle mass density function.
"""
function density_func(dist::Exponential{FT}) where {FT<:Real}
  # density = n / θ * exp(-x/θ)

  # can reuse the density of gamma distribution
  density_func_gamma = density_func(Gamma(FT(1), FT(1), FT(1)))
  function f(n, θ, x)
    density_func_gamma(n, θ, FT(1), x)
  end
  return f
end

function density_func(dist::Gamma{FT}) where {FT<:Real}
  # density = n / θ^k / Γ(k) * x^(k-1) * exp(-x/θ)
  function f(n, θ, k, x)
    n .* x.^(k .- 1) ./ θ.^k ./ gamma.(k) .* exp.(-x ./ θ)
  end
  return f
end

function density_func(dist::Mixture{FT}) where {FT<:Real}
  # mixture density is sum of densities of subdists
  num_pars = [nparams(d) for d in dist.subdists]
  dens_funcs = [density_func(d) for d in dist.subdists]
  function f(params...)
    inputs = collect(params)
    dist_params = inputs[1:end-1]
    x = inputs[end]
    i = 1
    output = 0.0
    for (n, dens_func)  in zip(num_pars, dens_funcs)
      output += dens_func(dist_params[i:i+n-1]..., x)
      i += n
    end
    return output
  end
  return f
end


"""
  density(dist, x)

  - `dist` - is a particle mass distribution
  - `x` - is a point to evaluate the density of `dist` at
Returns the particle mass density evaluated at point `x`.
"""
function density(dist::ParticleDistribution{FT}, x::FT) where {FT<:Real}
  if any(x .< zero(x))
    error("Density can only be evaluated at nonnegative values.")
  end
  density_func(dist)(reduce(vcat, get_params(dist)[2])..., x)
end


"""
  nparams(dist)

  - `dist` - is a particle mass distribution
Returns the number of settable parameters of dist.
"""
function nparams(dist::Primitive{FT}) where {FT<:Real}
  length(propertynames(dist))
end

function nparams(dist::Mixture{FT}) where {FT<:Real}
  sum([nparams(d) for d in dist.subdists])
end


"""
  get_params(dist)

  - `dist` - is a particle mass distribution
Returns the names and values of settable parameters for a dist.
"""
function get_params(dist::Primitive{FT}) where {FT<:Real}
  params = Array{Symbol, 1}(collect(propertynames(dist)))
  values = Array{FT, 1}([getproperty(dist, p) for p in params])
  return params, values
end

function get_params(dist::Mixture{FT}) where {FT<:Real}
  params, values = Array{Array{Symbol, 1}}([]), Array{Array{FT, 1}}([])
  for (i, d) in enumerate(dist.subdists)
    params_sub, values_sub = get_params(d)
    append!(params, [params_sub])
    append!(values, [values_sub])
  end
  return params, values
end


"""
  update_params(dist, params)

  - `dist` - is a particle mass distribution
Returns a new distribution of same type as input with `params` as parameters.
If dist is of type `Mixture` then subdistributions are updated.
"""
function update_params(dist::Exponential{FT}, values::Array{FT}) where {FT<:Real}
  Exponential(values...)
end

function update_params(dist::Gamma{FT}, values::Array{FT}) where {FT<:Real}
  Gamma(values...)
end

function update_params(dist::Mixture{FT}, values::Array{FT}) where {FT<:Real}
  if length(values) != nparams(dist)
    error("length of values must match number of params of dist.")
  end

  # create new subdistributions one dist at a time
  i = 1
  dist_arr = Array{ParticleDistribution{FT}}([])
  for d in dist.subdists
    n = nparams(d)
    push!(dist_arr, update_params(d, values[i:i+n-1]))
    i += n
  end

  Mixture(dist_arr)
end


function update_params_from_moments(dist::ParticleDistribution{FT}, m::Array{FT}) where {FT<:Real}
  if length(m) != nparams(dist)
    error("Number of moments must be consistent with distribution type.")
  end
  check_moment_consistency(m)

  # function whose minimum defines new parameters of dist given m
  function g(x)
    norm(m - moment_func(dist)(exp.(x)..., collect(0:length(m)-1)))
  end

  # minimize g to find parameter values so that moments of dist are close to m
  r = optimize(
    g,
    log.(reduce(vcat, get_params(dist)[2])) .+ 1e-6, # initial value for new dist parameters is old dist parameters
    LBFGS()
  );

  update_params(dist, exp.(r.minimizer))
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

end #module ParticleDistributions.jl
