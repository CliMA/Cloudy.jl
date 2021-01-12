module LOGCollocation

using Cloudy.BasisFunctions
using QuadGK
using NonNegLeastSquares
using LinearAlgebra
using Convex
using SCS
using Random, Distributions
using Clustering

export select_rbf_locsa
export select_rbf_shapes
export get_rbf_inner_products
export get_IC_vec
export get_kernel_rbf_sink
export get_kernel_rbf_source
export get_mass_cons_term
export collision_coalescence
export get_constants_vec

"""
Logarithmically spaced between 0 and 1
"""
function select_rbf_locs(nlocs::Int64; xmin::FT=eps(), xmax=1.00) where {FT <: Real}
    locs = exp.(range(log(xmin), stop=log(xmax), length=nlocs+1) |> collect)
    return locs[1:end-1]
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
In logspace, Φ_ij = ϕ_j(z_i) where z_i are the centers of the basis distributions
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

function get_mass_cons_term(basis::Array{PrimitiveUnivariateBasisFunc, 1}, xmin::FT, ζ::FT; zstart::FT = 0.0, zstop::FT=1.0) where {FT <: Real}
    Nb = length(basis)
    J = zeros(FT, Nb)
    for i=1:Nb
        integrand = w -> basis_func(basis[i])(w) * ζ^w
        J[i] = quadgk(integrand, xstart, xstop)[1]*xmin
    end
    
    return J
end

"""
Calculating the initial condition vector: non-mass conserving form
zj are centers of the basis functions in their natural space
"""
function get_IC_vec(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, ζ::FT, xmin::FT, A::Array{FT}) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = N(zi)
    Nb = length(basis)
    b = Array{FT}(undef, Nb)
    for i=1:Nb
        zi = get_moment(basis[i], 1.0)
        xi = xmin*(ζ^zi)
        b[i] = u0(xi)*xi*log(ζ)
    end
    c0 = get_constants_vec(b, A)
    return c0
end

"""
Calculating the initial condition vector: mass conserving form
 - zj are centers of the basis functions, and correspond to ln(x_j)
 - mass is calculated via an explicit integral in log space
"""
function get_IC_vec(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, ζ::FT, xmin::FT, A::Array{FT}, J::Array{FT,1}) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = u0(xi)
    Nb = length(basis)
    b = Array{FT}(undef, Nb)
    for i=1:Nb
        zi = get_moment(basis[i], 1.0)
        xi = xmin*(ζ^zi)
        b[i] = u0(xi)*xi*log(ζ)
    end
    mass = quadgk(x->u0.(x)*x, xmin, xmin*ζ)[1]
    
    c0 = get_constants_vec(b, A, J, mass)
    return (c0, mass)
end

""" 
Represents consumption of particles of size x_j
Calculated via explicit integrals in log space
"""

function get_kernel_rbf_sink(basis::Array{PrimitiveUnivariateBasisFunc, 1}, ζ::FT, xmin::FT, rbf_locs_z::Array{FT}, kernel::Function; zstart::FT = 0.0, zstop::FT = 1.0) where {FT <: Real}
    # N_ijk = <basis[k](x), basis[j](x'), K(x, x'), basis[i](x) dx' dx 
    Nb = length(basis)
    N = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = ξ -> basis_func(basis[k])(ξ)*kernel([xmin*ζ^rbf_locs_z[i], xmin*ζ^ξ])
                N[i,j,k] = basis_func(basis[j])(rbf_locs_z[i]) * quadgk(integrand, zstart, zstop)[1]
            end
        end
    end
    return N
end

function get_kernel_rbf_source(basis::Array{PrimitiveUnivariateBasisFunc, 1}, ζ::FT, xmin::FT, rbf_locs_z::Array{FT}, kernel::Function; zstart::FT = 0.0) where {FT <: Real}
    # M_ijk = 1/2 <basis[k](x-x'), basis[j](x'), K(x-x', x'), basis[i](x) dx' dx 
    Nb = length(basis)
    M = zeros(FT, Nb, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = ξ -> basis_func(basis[j])(log(ζ^(rbf_locs_z[i]) - ζ^ξ)/log(ζ))/(ζ^(rbf_locs_z[i]) - ζ^ξ) * 
                                        basis_func(basis[k])(ξ)* kernel([log(ζ^(rbf_locs_z[i]) - ζ^ξ)/log(ζ), xmin*ζ^ξ])
                M[i, j, k] = ζ^rbf_locs_z[i] * quadgk(integrand, zstart, rbf_locs_z[i]-log(2)/log(ζ))[1]
            end
        end
    end

    return M
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
        dndt[i] = (c'*M[i,:,:]*c - c'*N[i,:,:]*c)
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