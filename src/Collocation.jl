module Collocation

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
Sample points from the initial distribution and select locations via clustering; 
add additional locations up to the maximum
"""
function select_rbf_locs(u0::Function, xmax::FT, nlocs::Int64; nsample::Int64 = 1000, xmin::FT = eps()) where {FT <: Real}
    Random.seed!(123)
    sampler = Uniform(xmin, xmax)
    accepter = Uniform(0,1)
    Npdf = quadgk(u0, xmin, xmax)[1]
    pdf = x-> u0(x)/Npdf

    locs = zeros(FT, nlocs)

    samples = zeros(FT, nsample)

    # generate the random samples
    max_iter = 100
    for i=1:nsample
        count = 0
        while count < max_iter
            isample = rand(sampler,1)
            p_accept = rand(accepter, 1)
            if (p_accept[1] > pdf(isample[1]))
                samples[i] = isample[1]
                break
            end
            count += 1
        end
        if (count == max_iter)
            error("Could not randomly sample from PDF")
        end
    end

    # cluster the points
    nclusters = Int64(round((maximum(samples))/xmax*nlocs - 1))
    R = kmeans(vcat(samples'), nclusters)
    locs[1:nclusters] = sort(R.centers[1,:])

    # fill in the remaining
    nother = nlocs - nclusters
    if (nother == 1)
        locs[end] = FT(xmax)
    else
        other_pts = collect(range(locs[nclusters], stop=FT(xmax), length=nother))
        locs[nclusters+1:end] = other_pts
    end

    return locs
end

function select_rbf_shapes(rbf_locs::Array{FT}, smoothing_factor::FT = 1.6) where {FT <: Real}
    nlocs = length(rbf_locs)
    rbf_shapes = zeros(FT, nlocs)
    rbf_locs_tmp = vcat(0.0, rbf_locs, rbf_locs[end]+eps())

    for i=1:nlocs
        s1 = (rbf_locs_tmp[i+1] - rbf_locs_tmp[i])/smoothing_factor
        s2 = (rbf_locs_tmp[i+2] - rbf_locs_tmp[i+1])/smoothing_factor
        rbf_shapes[i] = max(s1, s2)
    end

    return rbf_shapes
end

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


"""
mass conserving form
"""

function get_IC_vec(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, A::Array{FT}, J::Array{FT,1}; xstart::FT = eps(), xstop::FT = 1000.0) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = <u0, basis[i]>
    # must enforce a positivty constraint
    # calculate the b_i vector
    Nb = length(basis)
    b = Array{FT}(undef, Nb)
    for i=1:Nb
        integrand = x-> u0.(x)*basis_func(basis[i])(x)
        b[i] = quadgk(integrand, xstart, xstop)[1]
    end
    mass = quadgk(x->u0.(x)*x, xstart, xstop)[1]
    
    c0 = get_constants_vec(b, A, J, mass)
    return (c0, mass)
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
    # first calculate c(t); 
    c = get_constants_vec(nj, A, J, mass)

    # time rate of change: dn/dt|_xj, t
    dndt = zeros(FT, Nb)
    for i=1:Nb
        dndt[i] = (1/2*c'*M[i,:,:]*c - c'*N[i,:,:]*c)
    end

    return dndt
end

function get_constants_vec(nj::Array{FT, 1}, A::Array{FT}, J::Array{FT,1}, mass::FT; xstart::FT = eps(), xstop::FT = 1000.0) where {FT<:Real}
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

end