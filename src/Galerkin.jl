module Galerkin

using Cloudy.BasisFunctions
using Cloudy.KernelTensors
using QuadGK
using Cubature
using LinearAlgebra

export get_rbf_inner_products
export get_IC_vec
export get_kernel_rbf_sink
export get_kernel_rbf_source

function get_rbf_inner_products(basis::Array{PrimitiveUnivariateBasisFunc, 1}; xstart::FT = eps(), xstop::FT = 1000.0) where {FT<:Real}
    # A_ij = <basis[i], basis[j]>
    Nb = length(basis)
    A = zeros(FT, Nb, Nb)
    for i=1:Nb
        for j=i:Nb
            integrand = x-> basis_func(basis[i])(x)*basis_func(basis[j])(x)
            A[i,j] = quadgk(integrand, xstart, xstop)[1]
            A[j,i] = A[i,j]
        end
    end
    
    return A
end

function get_IC_vec(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, A::Array{FT}; xstart::FT = eps(), xstop::FT = 1000.0) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = <u0, basis[i]>
    
    # calculate the b_i vector
    Nb = length(basis)
    b = Array{FT}(undef, Nb)
    for i=1:Nb
        integrand = x-> u0.(x)*basis_func(basis[i])(x)
        b[i] = quadgk(integrand, xstart, xstop)[1]
    end

    # calculate c0
    c0 = A \ b
    return c0
end

function get_kernel_rbf_sink(basis::Array{PrimitiveUnivariateBasisFunc, 1}, kernel::Function; xstart::FT = eps(), xstop::FT = 1000.0) where {FT <: Real}
    # N_ijk = <basis[k](x), basis[j](x'), K(x, x'), basis[i](x) dx' dx 
    Nb = length(basis)
    N = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = x -> basis_func(basis[k])(x[1]) * basis_func(basis[j])(x[2]) * kernel(x) * basis_func(basis[i])(x[1])
                N[i,j,k] = hcubature(integrand, [xstart, xstart], [xstop, xstop])[1]
            end
        end
    end
    return N
end

function get_kernel_rbf_source(basis::Array{PrimitiveUnivariateBasisFunc, 1}, kernel::Function; xstart::FT = eps(), xstop::FT = 1000.0) where {FT <: Real}
    # M_ijk = 1/2 <basis[k](x-x'), basis[j](x'), K(x-x', x'), basis[i](x) dx' dx 
    Nb = length(basis)
    M = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = x -> inner_integrand(basis[k], basis[j], kernel, x)*basis_func(basis[i])(x)
                M[i, j, k] = quadgk(integrand, xstart, xstop)[1]
            end
        end
    end

    return M
end

function inner_integrand(basis_k::PrimitiveUnivariateBasisFunc, basis_j::PrimitiveUnivariateBasisFunc, kernel::Function, x::FT; xstart::FT = eps()) where {FT <: Real}
    inner_fn = y -> 1/2*basis_func(basis_k)(x - y) * basis_func(basis_j)(y) * kernel([x-y, x])
    inner_int = quadgk(inner_fn, xstart, x)[1]
    return inner_int
end

end