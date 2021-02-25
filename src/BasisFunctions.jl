module BasisFunctions

using QuadGK
using SpecialFunctions: gamma

export AbstractBasisFunc
export PrimitiveUnivariateBasisFunc
export CompactBasisFunc
export GlobalBasisFunc
export GaussianBasisFunction
export LognormalBasisFunction
export GammaBasisFunction
export CompactBasisFunction1
export CompactBasisFunctionUneven
export CompactBasisFunctionLog
export basis_func
export evaluate_rbf
export get_moment
export get_params
export get_support


"""
  AbstractBasisFunc{FT}

A basis function over R^d, which can take a variety of forms.
"""
abstract type AbstractBasisFunc{FT} end

"""
    PrimitiveUnivariateBasisFunc{FT}

A 1D basis function over R, which can take a variety of forms.
"""
abstract type PrimitiveUnivariateBasisFunc{FT} <: AbstractBasisFunc{FT} end

abstract type CompactBasisFunc{FT} <: PrimitiveUnivariateBasisFunc{FT} end

abstract type GlobalBasisFunc{FT} <: PrimitiveUnivariateBasisFunc{FT} end

"""
   GaussianBasisFunction{FT}

A normal distribution.
"""
struct GaussianBasisFunction{FT} <: GlobalBasisFunc{FT}
    "center of the basis function"
    μ::FT
    "width of the basis function"
    σ::FT

    function GaussianBasisFunction(μ::FT, σ::FT) where {FT <: Real}
        if σ <= 0
          error("σ needs to be positive")
        end
      
        new{FT}(μ, σ)
    end
end


"""
   LognormalBasisFunction{FT}

A lognormal distribution.
"""
struct LognormalBasisFunction{FT} <: GlobalBasisFunc{FT}
    "mean of log(x)"
    μ::FT
    "std dev of log(x)"
    σ::FT

    function LognormalBasisFunction(μ::FT, σ::FT) where {FT <: Real}
        if σ <= 0
          error("σ needs to be positive")
        end
      
        new{FT}(μ, σ)
    end
end

"""
   GammaBasisFunction{FT}

A normal distribution.
"""
struct GammaBasisFunction{FT} <: GlobalBasisFunc{FT}
    "shape parameter"
    k::FT
    "scale parameter"
    θ::FT

    function GammaBasisFunction(k::FT, θ::FT) where {FT <: Real}
        if θ <= 0
          error("θ needs to be positive")
        end
      
        new{FT}(k, θ)
    end
end

"""
   CompactBasisFunction{FT}

A compactly supported basis function of degree 7, which is twice differentiable
"""
struct CompactBasisFunction1{FT} <: CompactBasisFunc{FT}
    "mean/center"
    μ::FT
    "scale parameter"
    θ::FT

    function CompactBasisFunction1(μ::FT, θ::FT) where {FT <: Real}
        if θ <= 0
          error("θ needs to be positive")
        end
      
        new{FT}(μ, θ)
    end
end

"""
   CompactBasisFunction{FT}

A compactly supported basis function of degree 7, which is twice differentiable
"""
struct CompactBasisFunctionUneven{FT} <: CompactBasisFunc{FT}
    "mean/center"
    μ::FT
    "LH scale parameter"
    θ_L::FT
    "RH scale parameter"
    θ_R::FT

    function CompactBasisFunctionUneven(μ::FT, θ_L::FT, θ_R::FT) where {FT <: Real}
        if θ_L <= 0
          error("θ needs to be positive")
        end

        if θ_R <= 0
          error("θ needs to be positive")
        end
      
        new{FT}(μ, θ_L, θ_R)
    end
end

"""
   CompactBasisFunction{FT}

A compactly supported basis function of degree 7, which is twice differentiable
"""
struct CompactBasisFunctionLog{FT} <: CompactBasisFunc{FT}
    "mean/center"
    μ::FT
    "scale parameter"
    θ::FT

    function CompactBasisFunctionLog(μ::FT, θ::FT) where {FT <: Real}
        if θ <= 0
          error("θ needs to be positive")
        end
      
        new{FT}(μ, θ)
    end
end


"""
  basis_func(dist)

  `rbf` - Radial Basis Function
Returns a function that computes the moments of `dist`.
"""
function basis_func(rbf::GaussianBasisFunction{FT}) where {FT <: Real}
    p = get_params(rbf)[2]
    μ = p[1]
    σ = p[2]
    function f(μ, σ, x)
        exp(-((x-μ)/σ)^2/2)/σ/sqrt(2*pi)
    end
    g = x-> f(μ, σ, x)
    return g
end

function basis_func(rbf::LognormalBasisFunction{FT}) where {FT <: Real}
  p = get_params(rbf)[2]
  μ = p[1]
  σ = p[2]
  function f(μ, σ, x)
      1/x/σ/sqrt(2*pi)*exp(-(log(x)-μ)^2/2/σ^2)
  end
  g = x-> f(μ, σ, x)
  return g
end

function basis_func(rbf::GammaBasisFunction{FT}) where {FT <: Real}
  p = get_params(rbf)[2]
  k = p[1]
  θ = p[2]
  function f(k, θ, x)
      x^(k-1)*exp(-x/θ)/θ^k/gamma(k)
  end
  g = x-> f(k, θ, x)
  return g
end

function basis_func(rbf::CompactBasisFunction1{FT}) where {FT <: Real}
  p = get_params(rbf)[2]
  μ = p[1]
  θ = p[2]
  function f(μ,θ,x)
    r = abs(x-μ)/θ
    if r > 1
      0
    else
      12/35*(1-r)^4*(4+16*r+12*r^2+3*r^3)
    end
  end
  g = x -> f(μ,θ,x)
  return g
end

function basis_func(rbf::CompactBasisFunctionUneven{FT}) where {FT <: Real}
  p = get_params(rbf)[2]
  μ = p[1]
  θ_L = p[2]
  θ_R = p[3]
  function f(μ,θ_L,θ_R,x)
    dx = x - μ
    if dx > 0
      r = dx / θ_R
    else
      r = -dx / θ_L
    end

    if r > 1
      0
    else
      12/35*(1-r)^4*(4+16*r+12*r^2+3*r^3)
    end
  end
  g = x -> f(μ,θ_L,θ_R,x)
  return g
end

function basis_func(rbf::CompactBasisFunctionLog{FT}) where {FT <: Real}
  p = get_params(rbf)[2]
  μ = p[1]
  θ = p[2]
  function f(μ,θ,x)
    r = abs(log(x)-μ)/θ
    if r > 1
      0
    else
      12/35*(1-r)^4*(4+16*r+12*r^2+3*r^3)/x/θ
    end
  end
  g = x -> f(μ,θ,x)
  return g
end

"""
  get_params(basis_func)

  - `basis_func` - is a basis function
Returns the names and values of settable parameters for a dist.
"""
function get_params(basis_func::AbstractBasisFunc{FT}) where {FT<:Real}
  params = Array{Symbol, 1}(collect(propertynames(basis_func)))
  values = Array{FT, 1}([getproperty(basis_func, p) for p in params])
  return params, values
end

function evaluate_rbf(basis::Array{RBF,1}, c::Array{FT}, x::Array{FT}) where {FT<:Real, RBF <: PrimitiveUnivariateBasisFunc}
  Nb = length(basis)
  if (length(c) != Nb)
    error("Number of coefficients must match number of basis functions")
  end

  approx = zeros(FT, length(x))
  for i=1:Nb
    approx += c[i]*basis_func(basis[i]).(x)
  end

  return approx
end

function evaluate_rbf(basis::Array{RBF,1}, c::Array{FT}, x::FT) where {FT<:Real, RBF <: PrimitiveUnivariateBasisFunc}
  Nb = length(basis)
  if (length(c) != Nb)
    error("Number of coefficients must match number of basis functions")
  end

  approx = 0
  for i=1:Nb
    approx += c[i]*basis_func(basis[i])(x)
  end

  return approx
end

function evaluate_rbf(basisfun::RBF, x::FT) where {FT<:Real, RBF <: PrimitiveUnivariateBasisFunc}
  return basis_func(basisfun)(x)
end


""" 
Calculate the qth moment of the basis function, or of the whole basis 
"""
function get_moment(rbf::RBF, q::FT; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real, RBF <: GlobalBasisFunc}
  integrand = x-> basis_func(rbf)(x)*x^q
  mom = quadgk(integrand, xstart, xstop)[1]
  return mom
end

function get_moment(rbf::RBF, q::FT; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real, RBF <: CompactBasisFunc}
  integrand = x-> basis_func(rbf)(x)*x^q
  supp = get_support(rbf)
  mom = quadgk(integrand, max(xstart, supp[1]), min(xstop, supp[2]))[1]
  return mom
end

function get_moment(basis::Array{RBF, 1}, q::FT; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real, RBF <: PrimitiveUnivariateBasisFunc}
  Nb = length(basis)
  moms = zeros(FT, Nb)
  for i=1:Nb
    moms[i] = get_moment(basis[i], q, xstart=xstart, xstop=xstop)
  end

  return moms
end

function get_moment(rbf::LognormalBasisFunction, q::FT; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real}
    params = get_params(rbf)
    mu = params[2][1]
    sigma = params[2][2]
    mom = exp(q*mu+q^2*sigma^2/2)
    return mom
end

function get_moment(rbf::GammaBasisFunction, q::FT; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real}
    params = get_params(rbf)
    k = params[2][1]
    theta = params[2][2]
    moms = gamma(k+q)/gamma(k)*theta^q
  return moms
end

function get_moment(rbf::GaussianBasisFunction, q::FT; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real}
  params = get_params(rbf)
  mu = params[2][1]
  sigma = params[2][2]
  if q == 0.0
    return 1.0
  elseif q == 1.0
    return mu
  elseif q == 2.0
    return mu^2 + sigma^2
  else
    integrand = x-> basis_func(rbf)(x)*x^q
    mom = quadgk(integrand, xstart, xstop)[1]
    return mom
  end
end

function get_support(rbf::CompactBasisFunctionLog{FT}) where {FT <: Real}
  params = get_params(rbf)[2]
  μ = params[1]
  θ = params[2]
  xmin = exp(μ - θ)
  xmax = exp(μ + θ)
  return (xmin, xmax)
end

function get_support(rbf::CompactBasisFunction1{FT}) where {FT <: Real}
  params = get_params(rbf)
  μ = params[1]
  θ = params[2]
  xmin = μ - θ
  xmax = μ + θ
  return (xmin, xmax)
end

function get_support(rbf::CompactBasisFunctionUneven{FT}) where {FT <: Real}
  params = get_params(rbf)
  μ = params[1]
  θ_L = params[2]
  θ_R = params[3]
  xmin = μ - θ_L
  xmax = μ + θ_R
  return (xmin, xmax)
end

end