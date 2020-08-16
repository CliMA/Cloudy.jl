module Collocation

using Cloudy.BasisFunctions
using QuadGK
using NonNegLeastSquares
using LinearAlgebra

export get_rbf_inner_products
export get_IC_vec
export get_kernel_rbf_sink
export get_kernel_rbf_source
export get_mass_cons_term
export collision_coalescence
export get_constants_vec

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
    # must enforce a positivty constraint
    # calculate the b_i vector
    Nb = length(basis)
    b = Array{FT}(undef, Nb)
    for i=1:Nb
        integrand = x-> u0.(x)*basis_func(basis[i])(x)
        b[i] = quadgk(integrand, xstart, xstop)[1]
    end

    # calculate c0, enforcing positivity
    #c0 = A \ b
    c0 = nonneg_lsq(A, b)
    return c0[:,1]
end

"""
mass conserving form
"""
function get_IC_vec(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, A::Array{FT}, J::Array{FT,1}; xstart::FT = eps(), xstop::FT = 1000.0) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = <u0, basis[i]>
    # must enforce a positivty constraint
    # calculate the b_i vector
    Nb = length(basis)
    b = Array{FT}(undef, Nb+1)
    for i=1:Nb
        integrand = x-> u0.(x)*basis_func(basis[i])(x)
        b[i] = quadgk(integrand, xstart, xstop)[1]
    end
    mass = quadgk(x->u0.(x)*x, xstart, xstop)[1]
    b[Nb+1] = mass
    A2 = vcat(A, J')

    # calculate c0, enforcing positivity
    #c0 = A \ b
    c0 = nonneg_lsq(A2, b)
    return c0[:,1], mass
end

function get_kernel_rbf_sink(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT}, kernel::Function; xstart::FT = eps(), xstop::FT = 1000.0) where {FT <: Real}
    # N_ijk = <basis[k](x), basis[j](x'), K(x, x'), basis[i](x) dx' dx 
    Nb = length(basis)
    N = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = y -> basis_func(basis[k])(y)*kernel([rbf_locs[i], y])
                N[i,j,k] = basis_func(basis[j])(rbf_locs[i]) * quadgk(integrand, xstart, xstop)[1]
            end
        end
    end
    return N
end

function get_kernel_rbf_source(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT}, kernel::Function; xstart::FT = eps()) where {FT <: Real}
    # M_ijk = 1/2 <basis[k](x-x'), basis[j](x'), K(x-x', x'), basis[i](x) dx' dx 
    Nb = length(basis)
    M = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = y-> basis_func(basis[j])(rbf_locs[i] - y)*basis_func(basis[k])(y)* kernel([rbf_locs[i] - y, y])
                M[i, j, k] = quadgk(integrand, xstart, rbf_locs[i])[1]
            end
        end
    end

    return M
end

function get_mass_cons_term(basis::Array{PrimitiveUnivariateBasisFunc, 1}; xstart::FT = eps(), xstop::FT=1000.0) where {FT <: Real}
    Nb = length(basis)
    J = zeros(FT, Nb)
    for i=1:Nb
        integrand = x-> basis_func(basis[i])(x) * x
        J[i] = quadgk(integrand, xstart, xstop)[1]
    end
    
    return J
end

function collision_coalescence(nj::Array{FT,1}, A::Array{FT,2}, M::Array{FT,3}, N::Array{FT,3}) where {FT <: Real}
    Nb = length(nj)
    # first calculate c(t)
    c = nonneg_lsq(A, nj)[:,1]

    # time rate of change: dn/dt|_xj, t
    dndt = zeros(FT, Nb)
    for i=1:Nb
        dndt[i] = (1/2*c'*M[i,:,:]*c - c'*N[i,:,:]*c)
    end

    return dndt
end

"""
mass conserving form
"""
function collision_coalescence(nj::Array{FT,1}, A::Array{FT,2}, M::Array{FT,3}, N::Array{FT,3}, J::Array{FT,1}, mass::FT) where {FT <: Real}
    Nb = length(nj)
    # first calculate c(t)
    A2 = vcat(A, J')
    nj2 = vcat(nj, mass)

    c = nonneg_lsq(A2, nj2)[:,1]

    # time rate of change: dn/dt|_xj, t
    dndt = zeros(FT, Nb)
    for i=1:Nb
        dndt[i] = (1/2*c'*M[i,:,:]*c - c'*N[i,:,:]*c)
    end

    return dndt
end

function get_constants_vec(nj::Array{FT, 1}, A::Array{FT}, J::Array{FT,1}; xstart::FT = eps(), xstop::FT = 1000.0) where {FT<:Real}
    Nb = length(nj)
    # first calculate c(t)
    A2 = vcat(A, J')
    nj2 = vcat(nj, mass)

    c = nonneg_lsq(A2, nj2)[:,1]

    return c
end

end