module BasisFunctions

using QuadGK
export AbstractBasisFunc
export PrimitiveUnivariateBasisFunc
export GaussianBasisFunction
export basis_func
export evaluate_rbf
export get_moment


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

"""
    PrimitiveUnivariateBasisFunc{FT}

A 1D basis function over R, which can take a variety of forms.
"""
struct GaussianBasisFunction{FT} <: PrimitiveUnivariateBasisFunc{FT}
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
  basis_func(dist)

  `rbf` - Gaussian Basis Function
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

function evaluate_rbf(basis::Array{PrimitiveUnivariateBasisFunc,1}, c::Array{FT}, x::Array{FT}) where {FT<:Real}
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

function evaluate_rbf(basis::Array{PrimitiveUnivariateBasisFunc,1}, c::Array{FT}, x::FT) where {FT<:Real}
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

function get_moment(basis::Array{PrimitiveUnivariateBasisFunc, 1}, q::FT; xstart::FT = eps(), xstop::FT = 1000.0) where {FT <: Real}
  Nb = length(basis)
  moms = zeros(FT, Nb)
  for i=1:Nb
    integrand = x-> basis_func(basis[i])(x)*x^q
    moms[i] = quadgk(integrand, xstart, xstop)[1]
  end

  return moms
end

end