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

import NonlinearSolve as NLS
import LinearAlgebra: norm
import Optim: optimize, LBFGS
import NumericalIntegration as NI

# particle mass distributions available for microphysics
export AbstractParticleDistribution
export PrimitiveParticleDistribution
export ExponentialPrimitiveParticleDistribution
export GammaPrimitiveParticleDistribution
export MonodispersePrimitiveParticleDistribution
export AdditiveParticleDistribution
export MonodisperseAdditiveParticleDistribution
export ExponentialAdditiveParticleDistribution
export GammaAdditiveParticleDistribution

# methods that query particle mass distributions
export moment
export get_moments
export density
export nparams
export update_params
export moments_to_params
export update_params_from_moments
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
  AdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}

A particle mass distribution function that is a mixture of
primitive or truncated subdistribution functions.

# Constructors
  AdditiveParticleDistribution(dists::AbstractParticleDistribution{Real}...)
  AdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}})

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}
  "array of distributions"
  subdists::Array{AbstractParticleDistribution{FT}}

  function AdditiveParticleDistribution(dists::AbstractParticleDistribution{FT}...) where {FT<:Real}
    if length(dists) < 2
      error("need at least two subdistributions to form a mixture.")
    end

    new{FT}(collect(dists))
  end
end

function AdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}}) where {FT<:Real}
  AdditiveParticleDistribution(dist_arr...)
end

"""
  ExponentialAdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}

A particle mass distribution function that is a mixture of
Exponential distributions

# Constructors
  ExponentialAdditiveParticleDistribution(dists::AbstractParticleDistribution{Real}...)
  ExponentialAdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}})

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ExponentialAdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}
  "array of exponential distributions"
  subdists::Array{AbstractParticleDistribution{FT}}

  function ExponentialAdditiveParticleDistribution(dists::AbstractParticleDistribution{FT}...) where {FT<:Real}
    if length(dists) < 2
      error("need at least two subdistributions to form a mixture.")
    end
    if !all(typeof(dists[i])==ExponentialPrimitiveParticleDistribution{FT} for i in 1:length(dists))
      error("all subdistributions need to be of type ExponentialPrimitiveParticleDistribution")
    end
    new{FT}(collect(dists))
  end
end

function ExponentialAdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}}) where {FT<:Real}
  ExponentialAdditiveParticleDistribution(dist_arr...)
end


"""
  GammaAdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}

A particle mass distribution function that is a mixture of
Gamma distributions

# Constructors
  GammaAdditiveParticleDistribution(dists::AbstractParticleDistribution{Real}...)
  GammaAdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}})

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GammaAdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}
  "array of Gamma distributions"
  subdists::Array{AbstractParticleDistribution{FT}}

  function GammaAdditiveParticleDistribution(dists::AbstractParticleDistribution{FT}...) where {FT<:Real}
    if length(dists) < 2
      error("need at least two subdistributions to form a mixture.")
    end
    if !all(typeof(dists[i])==GammaPrimitiveParticleDistribution{FT} for i in 1:length(dists))
      error("all subdistributions need to be of type GammaPrimitiveParticleDistribution")
    end
    new{FT}(collect(dists))
  end
end

function GammaAdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}}) where {FT<:Real}
  GammaAdditiveParticleDistribution(dist_arr...)
end

"""
  MonodisperseAdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}

A particle mass distribution function that is a mixture of
Monodisperse distributions

# Constructors
  MonodisperseAdditiveParticleDistribution(dists::AbstractParticleDistribution{Real}...)
  MonodisperseAdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}})

# Fields
$(DocStringExtensions.FIELDS)
"""
struct MonodisperseAdditiveParticleDistribution{FT} <: AbstractParticleDistribution{FT}
  "array of Monodisperse distributions"
  subdists::Array{AbstractParticleDistribution{FT}}

  function MonodisperseAdditiveParticleDistribution(dists::AbstractParticleDistribution{FT}...) where {FT<:Real}
    if length(dists) < 2
      error("need at least two subdistributions to form a mixture.")
    end
    if !all(typeof(dists[i])==MonodispersePrimitiveParticleDistribution{FT} for i in 1:length(dists))
      error("all subdistributions need to be of type MonodispersePrimitiveParticleDistribution")
    end
    new{FT}(collect(dists))
  end
end

function MonodisperseAdditiveParticleDistribution(dist_arr::Array{AbstractParticleDistribution{FT}}) where {FT<:Real}
  MonodisperseAdditiveParticleDistribution(dist_arr...)
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
  # can reuse the moments of gamma distribution

  moment_func_gamma = moment_func(GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1)))
  function f(n, θ, q)
    moment_func_gamma(n, θ, FT(1), q)
  end
  return f
end

function moment_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT<:Real}
  # moment_of_dist = n * θ^q * Γ(q+k) / Γ(k)
  function f(n, θ, k, q)
    n .* θ.^q .* gamma.(q .+ k) / gamma.(k)
  end
  return f
end

function moment_func(dist::MonodispersePrimitiveParticleDistribution{FT}) where {FT<:Real}
  # moment_of_dist = n * θ^(q)
  function f(n, θ, q)
   n .* θ.^q
  end
  return f
end

function moment_func(dist::Union{AdditiveParticleDistribution{FT}, ExponentialAdditiveParticleDistribution{FT}, GammaAdditiveParticleDistribution{FT}, MonodisperseAdditiveParticleDistribution{FT}}) where {FT<:Real}
  # mixture moment is sum of moments of subdistributions
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
function moment(dist::AbstractParticleDistribution{FT}, q::FT) where {FT<:Real}
  moment_func(dist)(reduce(vcat, get_params(dist)[2])..., q)
end

"""
    moments(pdist::GammaParticleDistribution{FT})
Returns the first P (0, 1, 2) moments of the distribution where P is the innate
numer of prognostic moments
"""
function get_moments(pdist::GammaPrimitiveParticleDistribution{FT}) where {FT<:Real}
  return [pdist.n, pdist.n * pdist.k * pdist.θ, pdist.n*pdist.k*(pdist.k+1)*pdist.θ^2]
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

function density_func(dist::Union{AdditiveParticleDistribution{FT}, ExponentialAdditiveParticleDistribution{FT}, GammaAdditiveParticleDistribution{FT}}) where {FT<:Real}
  # mixture density is sum of densities of subdistributions
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
function density(dist::AbstractParticleDistribution{FT}, x::FT) where {FT<:Real}
  if any(x .< zero(x))
    error("Density can only be evaluated at nonnegative values.")
  end
  density_func(dist)(x)
end


"""
  nparams(dist)

  - `dist` - is a particle mass distribution
Returns the number of settable parameters of dist.
"""
function nparams(dist::PrimitiveParticleDistribution{FT}) where {FT<:Real}
  length(propertynames(dist))
end

function nparams(dist::Union{AdditiveParticleDistribution{FT}, ExponentialAdditiveParticleDistribution{FT}, GammaAdditiveParticleDistribution{FT}, MonodisperseAdditiveParticleDistribution{FT}}) where {FT<:Real}
  sum([nparams(d) for d in dist.subdists])
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

function get_params(dist::Union{AdditiveParticleDistribution{FT}, ExponentialAdditiveParticleDistribution{FT}, GammaAdditiveParticleDistribution{FT}, MonodisperseAdditiveParticleDistribution{FT}}) where {FT<:Real}
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
If dist is of type `AdditiveParticleDistribution` then subdistributions are updated.
"""
function update_params(dist::ExponentialPrimitiveParticleDistribution{FT}, values::Array{FT}) where {FT<:Real}
  ExponentialPrimitiveParticleDistribution(values...)
end

function update_params(dist::GammaPrimitiveParticleDistribution{FT}, values::Array{FT}) where {FT<:Real}
  GammaPrimitiveParticleDistribution(values...)
end

function update_params(dist::MonodispersePrimitiveParticleDistribution{FT}, values::Array{FT}) where {FT<:Real}
  MonodispersePrimitiveParticleDistribution(values...)
end

function update_params(dist::AdditiveParticleDistribution{FT}, values::Array{FT}) where {FT<:Real}
  if length(values) != nparams(dist)
    error("length of values must match number of params of dist.")
  end

  # create new subdistributions one dist at a time
  i = 1
  dist_arr = Array{AbstractParticleDistribution{FT}}([])
  for d in dist.subdists
    n = nparams(d)
    push!(dist_arr, update_params(d, values[i:i+n-1]))
    i += n
  end

  AdditiveParticleDistribution(dist_arr)
end

function update_params(dist::ExponentialAdditiveParticleDistribution{FT}, values::Array{FT}) where {FT<:Real}
  if length(values) != nparams(dist)
    error("length of values must match number of params of dist.")
  end

  # create new subdistributions one dist at a time
  i = 1
  dist_arr = Array{AbstractParticleDistribution{FT}}([])
  for d in dist.subdists
    n = nparams(d)
    push!(dist_arr, update_params(d, values[i:i+n-1]))
    i += n
  end

  ExponentialAdditiveParticleDistribution(dist_arr)
end

function update_params(dist::GammaAdditiveParticleDistribution{FT}, values::Array{FT}) where {FT<:Real}
  if length(values) != nparams(dist)
    error("length of values must match number of params of dist.")
  end

  # create new subdistributions one dist at a time
  i = 1
  dist_arr = Array{AbstractParticleDistribution{FT}}([])
  for d in dist.subdists
    n = nparams(d)
    push!(dist_arr, update_params(d, values[i:i+n-1]))
    i += n
  end

  GammaAdditiveParticleDistribution(dist_arr)
end

function update_params(dist::MonodisperseAdditiveParticleDistribution{FT}, values::Array{FT}) where {FT<:Real}
  if length(values) != nparams(dist)
    error("length of values must match number of params of dist.")
  end

  # create new subdistributions one dist at a time
  i = 1
  dist_arr = Array{AbstractParticleDistribution{FT}}([])
  for d in dist.subdists
    n = nparams(d)
    push!(dist_arr, update_params(d, values[i:i+n-1]))
    i += n
  end

  MonodisperseAdditiveParticleDistribution(dist_arr)
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
function update_dist_from_moments!(pdist::GammaPrimitiveParticleDistribution{FT}, moments::Array{FT}) where {FT<:Real}
  if length(moments) != 3
    throw(ArgumentError("must specify exactly 3 moments for gamma distribution"))
  end
  pdist.n = moments[1]
  pdist.k = (moments[2]/moments[1])/(moments[3]/moments[2]-moments[2]/moments[1])
  pdist.θ = moments[3]/moments[2]-moments[2]/moments[1]
end

"""
    update_dist_from_moments!(pdist::ExponentialPrimitiveParticleDistribution{FT}, moments::Array{FT})

Updates parameters of the gamma distribution given the first three moments
"""
function update_dist_from_moments!(pdist::ExponentialPrimitiveParticleDistribution{FT}, moments::Array{FT}) where {FT<:Real}
  if length(moments) != 2
    throw(ArgumentError("must specify exactly 2 moments for exponential distribution"))
  end
  pdist.n = moments[1]
  pdist.θ = moments[2]/moments[1]
end

"""
  update_params_from_moments(ODE_parameters, target_moments)

  - `ODE_parameters` - ODE parameters, a Dict containing a key ":dist", whose
                       value is the distribution at the previous time step.
                       dist is a ParticleDistribution; it is used to calculate
                       the diagnostic moments and also to dispatch to the
                       moments-to-parameters mapping (done by the function
                       moments_to_params) for the given type of distribution
  - `target_moments` - array of moments used to update the distribution

Uses the given moments to (approximately) determine the corresponding
distribution parameters, then updates the distribution with these parameters.
The moments-to-parameters mapping is done differently depending on the
ParticleDistribution type - e.g., it can be done analytically (for priimtive
ParticleDistributions), it can involve solving an optimization problem (for
general ParticleDistributions), or it can involve solving a nonlinear system of
equations (for GammaAdditiveParticleDistributions and 
ExponentialAdditiveParticleDistributions).
For distributions where none of these methods work well, an alternative
solution would be to train a regression model (e.g., a random forest) to learn
the moments-to-parameters map and to write a moments_to_params function which
uses the predictions of that model.
"""
function update_params_from_moments(ODE_parameters, target_moments::Array{FT}) where {FT<:Real}
  # Extract the solution at the previous time step
  dist_prev = ODE_parameters[:dist]
  # Update parameters from the given target moments
  moments_to_params(dist_prev, target_moments)
end

function update_params_from_moments(ODE_parameters, target_moments::Array{FT}, param_range::Dict{String, Tuple{FT, FT}}) where {FT<:Real}
  # Extract the solution at the previous time step
  dist_prev = ODE_parameters[:dist]
  # Update parameters from the given target moments
  moments_to_params(dist_prev, target_moments; param_range = param_range)
end

function moments_to_params(dist::GammaPrimitiveParticleDistribution{FT}, target_moments::Array{FT}; param_range = Dict("θ" => (1e-5, 1e5), "k" => (eps(FT), FT(5)))) where {FT<:Real}
  if length(target_moments) != nparams(dist)
    error("Number of moments must be consistent with distribution type.")
  end

  M0 = target_moments[1]
  M1 = target_moments[2]
  M2 = target_moments[3]
  if M0 < eps(FT) || M1 < eps(FT) || M2 < eps(FT)
    n = FT(0)
    θ = FT(1)
    k = FT(2)
  else
    θ = max(param_range["θ"][1], min(param_range["θ"][2], -(M1^2 - M0*M2)/(M0*M1)))
    k = max(param_range["k"][1], min(param_range["k"][2], -M1^2/(M1^2 - M0*M2)))
    n = M1/(θ*k)
  end

  update_params(dist, [n, θ, k])

end

function moments_to_params(dist::ExponentialPrimitiveParticleDistribution{FT}, target_moments::Array{FT}; param_range = Dict("θ" => (1e-5, 1e5))) where {FT<:Real}
  if length(target_moments) != nparams(dist)
    error("Number of moments must be consistent with distribution type.")
  end

  M0 = target_moments[1]
  M1 = target_moments[2]
  if M0 < eps(FT) || M1 < eps(FT)
    n = FT(0)
    θ = FT(1)
  else
    θ = max(param_range["θ"][1], min(param_range["θ"][2], M1/M0))
    n = M1/θ
  end 

  update_params(dist, [n, θ])

end

function moments_to_params(dist::MonodispersePrimitiveParticleDistribution{FT}, target_moments::Array{FT}; param_range = Dict("θ" => (1e-5, 1e5))) where {FT<:Real}
  if length(target_moments) != nparams(dist)
    error("Number of moments must be consistent with distribution type.")
  end

  M0 = target_moments[1]
  M1 = target_moments[2]
  if M0 < eps(FT) || M1 < eps(FT)
    n = FT(0)
    θ = FT(1)
  else
    θ = max(param_range["θ"][1], min(param_range["θ"][2], M1/M0))
    n = M1/m
  end 

  update_params(dist, [n, θ])

end

function moments_to_params(dist::AbstractParticleDistribution{FT}, target_moments::Array{FT}) where {FT<:Real}
  n_moments = length(target_moments)
  if n_moments != nparams(dist)
    error("Number of moments must be consistent with distribution type.")
  end
  check_moment_consistency(target_moments)

  # Function whose minimum defines new parameters of dist given target_moments
  function g(x)
    norm(target_moments - moment_func(dist)(exp.(x)..., collect(0:n_moments-1)))
  end

  # Minimize g to find parameter values so that moments of dist are close to
  # target_moments; use distribution parameters at previous time step as an
  # initial guess for the parameters at the current time step
  r = optimize(
    g,
    log.(reduce(vcat, get_params(dist)[2])) .+ 1e-6,
    LBFGS()
  )

  update_params(dist, exp.(r.minimizer))
end

function moments_to_params(dist::Union{ExponentialAdditiveParticleDistribution, GammaAdditiveParticleDistribution, MonodisperseAdditiveParticleDistribution}, target_moments::Array{FT}) where {FT<:Real}
  # number of subdistributions
  m = length(dist.subdists)
  # number of parameters per subdistribution
  n_params_subdist = nparams(dist.subdists[1])

  start_params = reduce(vcat, get_params(dist)[2])
  n_vars = length(start_params) # number of unknowns in the system

  # Set up system of equations relating moments to the unknown parameters.
  c(x, p) = construct_system(x, dist, p)

  # Use NewtonRaphson to solve for the unknowns, using the distribution parameters at
  # the previous time step as an initial guess for the parameters at the
  # current time step.
  sol = zeros(n_vars)
  try
    model = NLS.NonlinearProblem(c, start_params, target_moments)
    sol = NLS.solve(model, NLS.NewtonRaphson())
  catch
    model = NLS.NonlinearProblem(c, start_params .* rand(n_vars), target_moments)
    sol = NLS.solve(model, NLS.NewtonRaphson())
  end

  # Get parameters in the correct order for use as input to update_params
  params = vcat([sol[i*n_params_subdist+1:(i+1)*n_params_subdist] for i in 0:m-1]...)
  # Update distribution with the new parameters
  update_params(dist, params)
end

"""
  construct_system(x, dist, M::Array{FT})

  - `x` - array of unknowns (= the distribution parameters)
  - `dist` - AbstractParticleDistribution
  - `M`- array of moments

Constructs a system of equations relating the moments (M) to the unknown
parameters (x) of the given distribution
"""
function construct_system(x, dist::GammaAdditiveParticleDistribution{FT}, M::Array{FT}) where {FT<:Real}
  n_equations = length(M)

  # x: [n_1, θ_1, k_1, ..., n_m, θ_m, k_m]
  # Construct system of n_equations equations relating moments to parameters
  # M_i = n[1] * θ[1]^i * (k[1] + i-1)*(k[1] + i-2)*...*k[1]
  #         + ...
  #         + n[m] * θ[m]^i * (k[m] + i-1)*(k[m] + i-2)*...*k[m]
  p1 = ones(FT, 1, n_equations)
  p2 = collect(reshape(0:n_equations-1, 1, n_equations))
  k_prod(k, n) = (n == 0) ? 1 : k_prod(k, n-1) * (k + n-1)
  g(n, θ, k) = sum(n.^p1 .* θ.^p2 .* k_prod.(k, p2), dims = 1)[:]

  return log.(M) - log.(g(x[1:3:end], x[2:3:end], x[3:3:end]))
end

function construct_system(x, dist::ExponentialAdditiveParticleDistribution{FT}, M::Array{FT}) where {FT<:Real}
  n_equations = length(M)

  # x: [n_1, θ_1, ..., n_m, θ_m]
  # Construct system of n_equations equations relating moments to parameters
  # M_i = i! * ∑(n[j] * θ[j]^i)
  c1 = factorial.(0:n_equations-1)
  p1 = ones(FT, 1, n_equations)
  p2 = collect(reshape(0:n_equations-1, 1, n_equations))
  g(n, θ) = c1 .* sum(n.^p1 .* θ.^p2, dims = 1)[:]

  return M - g(x[1:2:end], x[2:2:end])
end

function construct_system(x, dist::MonodisperseAdditiveParticleDistribution{FT}, M::Array{FT}) where {FT<:Real}
  n_equations = length(M)

  # x: [n_1, θ_1, ..., n_m, θ_m]
  # Construct system of n_equations equations relating moments to parameters
  # M_i = ∑(n[j] * θ[j]^i) 
  p1 = ones(FT, 1, n_equations)
  p2 = collect(reshape(0:n_equations-1, 1, n_equations))
  g(n, θ) = sum(n.^p1 .* θ.^p2, dims = 1)[:]
    
  return M - g(x[1:2:end], x[2:2:end])
end

"""
  moment_source_helper(dist, a, b, x_star)

  - `dist` - AbstractParticleDistribution
  - `p1` - power of particle mass
  - `p2` - power of particle mass
  - `x_star`- particle mass threshold

Returns ∫_0^x_star ∫_0^(x_star-x') x^p1 x'^p2 f(x) f(x') dx dx' for computations of the source of moments of the distribution below
the given threshold x_star.
"""
function moment_source_helper(dist::MonodispersePrimitiveParticleDistribution{FT}, p1::FT, p2::FT, x_star::FT) where {FT<:Real}
  n, θ = get_params(dist)[2]
  source = (θ < x_star/2) ? n^2 * θ^(p1+p2) : 0
  return source
end

function moment_source_helper(dist::ExponentialPrimitiveParticleDistribution{FT}, p1::FT, p2::FT, x_star::FT) where {FT<:Real}
  n, θ = get_params(dist)[2]

  f(x) = x^p1 * exp(-x/θ) * gamma_inc(p2 + 1, (x_star - x)/θ)[1] * gamma(p2 + 1)
  
  logx = range(log(1e-5), log(x_star), 51)
  x = exp.(logx)
  y = [x[1:end-1] .* f.(x[1:end-1]); FT(0)]

  return n^2 * θ^(p2 - 1) * NI.integrate(logx, y, NI.SimpsonEvenFast())
end

function moment_source_helper(dist::GammaPrimitiveParticleDistribution{FT}, p1::FT, p2::FT, x_star::FT) where {FT<:Real}
  n, θ, k = get_params(dist)[2]

  f(x) = x^(p1 + k - 1) * exp(-x/θ) * gamma_inc(p2 + k, (x_star - x)/θ)[1] * gamma(p2 + k)

  logx = range(log(1e-5), log(x_star), 51)
  x = exp.(logx)
  y = [x[1:end-1] .* f.(x[1:end-1]); FT(0)]

  return n^2 * θ^(p2 - k) / gamma(k)^2 * NI.integrate(logx, y, NI.SimpsonEvenFast())
end

end #module ParticleDistributions.jl
