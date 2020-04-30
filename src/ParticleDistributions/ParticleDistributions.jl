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
using NLPModels
using NLPModelsIpopt

# NLPModelsKnitro is a thin KNITRO wrapper for NLP (Nonlinear Programming)
# models. KNITRO is a commercial solver but a demo version can be
# downloaded from the Artelys website at
# https://www.artelys.com/solvers/knitro/
using Requires
@init @require NLPModelsKnitro = "bec4dd0d-7755-52d5-9a02-22f0ffc7efcb" begin
  using .NLPModelsKnitro
end

import LinearAlgebra: norm
import Optim: optimize, LBFGS

# particle mass distributions available for microphysics
export ParticleDistribution
export Primitive
export Exponential
export Gamma
export Mixture
export ExponentialMixture
export GammaMixture

# methods that query particle mass distributions
export moment
export density
export nparams
export update_params
export update_params_from_moments

# setters and getters
export get_params


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
  ExponentialMixture{FT} <: ParticleDistribution{FT}

A particle mass distribution function that is a mixture of
Exponential distributions

# Constructors
  ExponentialMixture(dists::ParticleDistribution{Real}...)
  ExponentialMixture(dist_arr::Array{ParticleDistribution{FT}})

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ExponentialMixture{FT} <: ParticleDistribution{FT}
  "array of exponential distributions"
  subdists::Array{ParticleDistribution{FT}}

  function ExponentialMixture(dists::ParticleDistribution{FT}...) where {FT<:Real}
    if length(dists) < 2
      error("need at least two subdistributions to form a mixture.")
    end
    if !all(typeof(dists[i])==Exponential{FT} for i in 1:length(dists))
      error("all subdistributions need to be of type Exponential")
    end
    new{FT}(collect(dists))
  end
end

function ExponentialMixture(dist_arr::Array{ParticleDistribution{FT}}) where {FT<:Real}
  ExponentialMixture(dist_arr...)
end


"""
  GammaMixture{FT} <: ParticleDistribution{FT}

A particle mass distribution function that is a mixture of
Gamma distributions

# Constructors
  GammaMixture(dists::ParticleDistribution{Real}...)
  GammaMixture(dist_arr::Array{ParticleDistribution{FT}})

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GammaMixture{FT} <: ParticleDistribution{FT}
  "array of Gamma distributions"
  subdists::Array{ParticleDistribution{FT}}

  function GammaMixture(dists::ParticleDistribution{FT}...) where {FT<:Real}
    if length(dists) < 2
      error("need at least two subdistributions to form a mixture.")
    end
    if !all(typeof(dists[i])==Gamma{FT} for i in 1:length(dists))
      error("all subdistributions need to be of type Gamma")
    end
    new{FT}(collect(dists))
  end
end

function GammaMixture(dist_arr::Array{ParticleDistribution{FT}}) where {FT<:Real}
  GammaMixture(dist_arr...)
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


function moment_func(dist::Union{Mixture{FT}, ExponentialMixture{FT}, GammaMixture{FT}}) where {FT<:Real}
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

function density_func(dist::Union{Mixture{FT}, ExponentialMixture{FT}, GammaMixture{FT}}) where {FT<:Real}
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

function nparams(dist::Union{Mixture{FT}, ExponentialMixture{FT}, GammaMixture{FT}}) where {FT<:Real}
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

function get_params(dist::Union{Mixture{FT}, ExponentialMixture{FT}, GammaMixture{FT}}) where {FT<:Real}
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

function update_params(dist::ExponentialMixture{FT}, values::Array{FT}) where {FT<:Real}
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

  ExponentialMixture(dist_arr)
end

function update_params(dist::GammaMixture{FT}, values::Array{FT}) where {FT<:Real}
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

  GammaMixture(dist_arr)
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
equations (for GammaMixtures and ExponentialMixtures).
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

function moments_to_params(dist::Gamma{FT}, target_moments::Array{FT}) where {FT<:Real}
  if length(target_moments) != nparams(dist)
    error("Number of moments must be consistent with distribution type.")
  end
  check_moment_consistency(target_moments)

  # target_moments[1] == M0, target_moments[2] == M1, target_moments[3] == M2
  n = target_moments[1]
  θ = (target_moments[1]*target_moments[3] - target_moments[2]^2) / target_moments[2]
  k = target_moments[2]^2 / (target_moments[1]*target_moments[3] - target_moments[2]^2)

  update_params(dist, [n, θ, k])

end

function moments_to_params(dist::Exponential{FT}, target_moments::Array{FT}) where {FT<:Real}
  if length(target_moments) != nparams(dist)
    error("Number of moments must be consistent with distribution type.")
  end
  check_moment_consistency(target_moments)

  # target_moments[1] == M0, target_moments[2] == M1
  n = target_moments[1]
  θ = target_moments[2] / target_moments[1]

  update_params(dist, [n, θ])

end

function moments_to_params(dist::ParticleDistribution{FT}, target_moments::Array{FT}) where {FT<:Real}
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

function moments_to_params(dist::Union{ExponentialMixture, GammaMixture}, target_moments::Array{FT}) where {FT<:Real}
  # number of equations
  n_equations = length(target_moments)
  # number of subdistributions
  m = length(dist.subdists)
  # number of parameters per subdistribution
  n_params_subdist = nparams(dist.subdists[1])

  start_params = reduce(vcat, get_params(dist)[2])
  start_params_ordered = vcat([start_params[i:n_params_subdist:end] for i in 1:n_params_subdist]...)
  n_vars = length(start_params_ordered) # number of unknowns in the system

  # Set up system of equations relating moments to the unknown parameters.
  c(x) = construct_system(x, dist, target_moments)

  # Use Ipopt as an alternative solver. By far not as good as Knitro though.
  # TODO: Allow the user to choose the solver.
  # Use Ipopt to solve for the unknowns. ADNLPModel is an AbstractNLPModel
  # using ForwardDiff to compute the derivatives
  model = ADNLPModel(x -> 0.0, start_params_ordered; c=c,
                     lcon=zeros(n_equations), ucon=zeros(n_equations),
                     lvar=zeros(n_vars))
  stats = ipopt(model, print_level=0)
  sol = stats.solution

  # Get parameters in the correct order for use as input to update_params
  params_ordered = vcat([sol[i:m:end] for i in 1:m]...)
  # Update distribution with the new parameters
  update_params(dist, params_ordered)
end

@init @require NLPModelsKnitro = "bec4dd0d-7755-52d5-9a02-22f0ffc7efcb" begin
  function moments_to_params(dist::Union{ExponentialMixture, GammaMixture}, target_moments::Array{FT}) where {FT<:Real}
    # number of equations
    n_equations = length(target_moments)
    # number of subdistributions
    m = length(dist.subdists)
    # number of parameters per subdistribution
    n_params_subdist = nparams(dist.subdists[1])

    start_params = reduce(vcat, get_params(dist)[2])
    start_params_ordered = vcat([start_params[i:n_params_subdist:end] for i in 1:n_params_subdist]...)
    n_vars = length(start_params_ordered) # number of unknowns in the system

    # Set up system of equations relating moments to the unknown parameters.
    c(x) = construct_system(x, dist, target_moments)
    # Use Knitro to solve for the unknowns, formulating the problem as a
    # nonlinear least-squares problem, and using the distribution parameters at
    # the previous time step as an initial guess for the parameters at the
    # current time step
    # Note: Problem formulation includes positivity constraints on all parameters
    model = ADNLSModel(c, start_params_ordered, n_equations, lvar=zeros(n_vars))
    stats = knitro(model, x0=model.meta.x0, outlev=0)
    sol = stats.solution

    # Get parameters in the correct order for use as input to update_params
    params_ordered = vcat([sol[i:m:end] for i in 1:m]...)
    # Update distribution with the new parameters
    update_params(dist, params_ordered)
  end
end


"""
  construct_system(x, dist, M::Array{FT})

  - `x` - array of unknowns (= the distribution parameters)
  - `dist` - ParticleDistribution
  - `M`- array of moments

Constructs a system of equations relating the moments (M) to the unknown
parameters (x) of the given distribution
"""
function construct_system(x, dist::GammaMixture{FT}, M::Array{FT}) where {FT<:Real}
  n_equations = length(M)
  m = length(dist.subdists)

  # x: [n_1, ..., n_m, θ_1, ..., θ_m, k_1, ..., k_m]
  # Construct system of n_equations equations relating moments to parameters
  # M[i] = n[1] * θ[1]^i * (k[1] + i-1)*(k[1] + i-2)*...*k[1]
  #         + ...
  #         + n[m] * θ[m]^i * (k[m] + i-1)*(k[m] + i-2)*...*k[m]
  F = []
  # M0 (==M[1]) is the sum of all n_j: M[1] = ∑n_j
  F_0 = M[1] - sum(x[1:m])
  push!(F, F_0)
  for i in 1:(n_equations-1)
    F_i = M[i+1]
    for j in 1:m
      # x[j+2*m] is k_j
      factor = prod([(x[j+2*m]+counter) for counter in 0:(i-1)])
      # x[j] is n_j,  x[j+m] is θ_j
      F_i -= x[j] * x[j+m]^i * factor
    end
    push!(F, F_i)
  end

  return F
end

function construct_system(x, dist::ExponentialMixture{FT}, M::Array{FT}) where {FT<:Real}
  n_equations = length(M)
  m = length(dist.subdists)

  # x: [n_1, ..., n_m, θ_1, ..., θ_m]
  # Construct system of n_equations equations relating moments to parameters
  # M[i] = i! * ∑(n[j] * θ[j]^i) 
  F = []
  # M0 (==M[1]) is the sum of all n_j: M[1] = ∑n_j
  F_0 = M[1] - sum(x[1:m])
  push!(F, F_0)
  for i in 1:(n_equations-1)
    # x[j] is n_j, x[j+m] is θ_j
    F_i = M[i+1] - factorial(i) * sum([x[j] * x[j+m]^i for j in 1:m])
    push!(F, F_i)
  end

  return F
end

end #module ParticleDistributions.jl
