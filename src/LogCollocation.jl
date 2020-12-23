module LOGCollocation

using Cloudy.BasisFunctions
using QuadGK
using NonNegLeastSquares
using LinearAlgebra
using Convex
using SCS
using Random, Distributions
using Clustering

export select_rbf_locs
export select_rbf_shapes
export get_rbf_inner_products
export get_IC_vec
export get_kernel_rbf_sink
export get_kernel_rbf_source
export get_mass_cons_term
export collision_coalescence
export get_constants_vec

"""
Logarithmically spaced between xmin and xmax
"""
function select_rbf_locs(xmin::FT, xmax::FT, nlocs::Int64) where {FT <: Real}
    locs = exp.(range(log(xmin), stop=log(xmax), length=nlocs) |> collect)
    return locs
end

"""
Given a set of RBF locations, set up shape parameters based on distance between
adjacent rbfs 
"""
function select_rbf_shapes(rbf_locs::Array{FT}; smoothing_factor::FT = 2.0) where {FT <: Real}
    rbf_pts = append!([0.0], rbf_locs)
    rbf_sigma = zeros(length(rbf_locs))
    rbf_sigma[1] = rbf_locs[1]/smoothing_factor
    rbf_sigma[2:end] = (rbf_locs[2:end] - rbf_pts[1:end-2])/smoothing_factor
    return rbf_sigma
end

""" 
For conversion between collocation point and constants vector
In logspace, Φ_ij = ϕ_k(z_j) where z_j are the centers of the basis distributions
"""
function get_rbf_inner_products(basis::Array{PrimitiveUnivariateBasisFunc, 1}; fake::FT = 0.0) where {FT <: Real}
    # Φ_ij = Φ_i(x_j)
    Nb = length(basis)
    Φ = zeros(FT, Nb, Nb)
    for j=1:Nb
        zj = get_moment(basis[j], 1.0)
        for i=1:Nb
            Φ[j,i] = evaluate_rbf(basis[i], zj)
        end
    end

    return Φ
end

"""
Calculating the initial condition vector: non-mass conserving form
zj are centers of the basis functions, and correspond to ln(x_j)
"""
function get_IC_vec(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, A::Array{FT}) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = u0(xi)
    Nb = length(basis)
    b = Array{FT}(undef, Nb)
    for i=1:Nb
        zi = get_moment(basis[i], 1.0)
        b[i] = u0(exp(zi))
    end
    c0 = get_constants_vec(b, A)
    return c0
end

"""
Calculating the initial condition vector: mass conserving form
 - zj are centers of the basis functions, and correspond to ln(x_j)
 - mass is calculated via an explicit integral in log space
"""
function get_IC_vec(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, A::Array{FT}, J::Array{FT,1}; xstart::FT = eps(), xstop::FT = 1e6) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = u0(xi)
    Nb = length(basis)
    b = Array{FT}(undef, Nb)
    for i=1:Nb
        zi = get_moment(basis[i], 1.0)
        b[i] = u0(exp(zi))
    end
    mass = quadgk(x->u0.(x)*x, xstart, xstop)[1]
    
    c0 = get_constants_vec(b, A, J, mass)
    return (c0, mass)
end

""" 
Represents consumption of particles of size x_j
Calculated via explicit integrals in log space
"""

function get_kernel_rbf_sink(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT}, kernel::Function; zstart::FT = -1e6, zstop::FT = 1e6) where {FT <: Real}
    # N_ijk = <basis[k](x), basis[j](x'), K(x, x'), basis[i](x) dx' dx 
    Nb = length(basis)
    N = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = ξ -> basis_func(basis[k])(ξ)*kernel([exp(rbf_locs[i]), exp(ξ)])
                N[i,j,k] = basis_func(basis[j])(log(rbf_locs[i])) * quadgk(integrand, zstart, zstop)[1]
            end
        end
    end
    return N
end

function get_kernel_rbf_source(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT}, kernel::Function; zstart::FT = -1e6) where {FT <: Real}
    # M_ijk = 1/2 <basis[k](x-x'), basis[j](x'), K(x-x', x'), basis[i](x) dx' dx 
    Nb = length(basis)
    M = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = ξ -> basis_func(basis[j])(log(exp(rbf_locs[i]) - exp(ξ)))*basis_func(basis[k])(ξ)* kernel([rbf_locs[i] - exp(ξ), exp(ξ)])
                M[i, j, k] = quadgk(integrand, zstart, rbf_locs[i])[1]
            end
        end
    end

    return M
end

"""
First moment of each basis function
"""
function get_mass_cons_term(basis::Array{PrimitiveUnivariateBasisFunc, 1}; zstart::FT = -1e6, zstop::FT=1e6) where {FT <: Real}
    Nb = length(basis)
    J = zeros(FT, Nb)
    for i=1:Nb
        integrand = z -> basis_func(basis[i])(z) * exp(2*z)
        J[i] = quadgk(integrand, zstart, zstop)[1]
    end
    
    return J
end

"""
Collision coalescnece rate of change: non-mass conserving form
"""
function collision_coalescence(nj::Array{FT,1}, A::Array{FT,2}, M::Array{FT,3}, N::Array{FT,3}) where {FT <: Real}
    Nb = length(nj)
    # first calculate c(t)
    c = get_constants_vec(nj, A)

    # time rate of change: dn/dt|_xj, t
    dndt = zeros(FT, Nb)
    for i=1:Nb
        dndt[i] = (1/2*c'*M[i,:,:]*c - c'*N[i,:,:]*c)
    end

    return dndt
end

"""
Collision coalescnece rate of change: mass conserving form
"""
function collision_coalescence(nj::Array{FT,1}, A::Array{FT,2}, M::Array{FT,3}, N::Array{FT,3}, J::Array{FT,1}, mass::FT) where {FT <: Real}
    Nb = length(nj)
    # first calculate c(t); 
    c = get_constants_vec(nj, A, J, mass)

    # time rate of change: dn/dt|_xj, t
    dndt = zeros(FT, Nb)
    for i=1:Nb
        dndt[i] = (1/2*c'*M[i,:,:]*c - c'*N[i,:,:]*c)
    end

    return dndt
end

""" 
Retrieves constants vector from a set of values at the collocation points with conservation
of the first moment
"""
function get_constants_vec(nj::Array{FT, 1}, A::Array{FT}, J::Array{FT,1}, mass::FT) where {FT<:Real}
    Nb = length(nj)
    # calculate c(t)

    # here x is the coefficients
    x = Variable(Nb)
    objective = sumsquares(A*x - nj)
    constraint1 = x >= 0
    constraint2 = J'*x == mass
    problem = minimize(objective, constraint1, constraint2)
    solve!(problem, SCS.Optimizer(verbose=false), verbose=false)
    c = x.value

    return c[:,1]
end

""" 
Retrieves constants vector from a set of values at the collocation points without mass conservation
"""
function get_constants_vec(nj::Array{FT, 1}, A::Array{FT}) where {FT<:Real}
    return nonneg_lsq(A, nj)[:,1]
end

end