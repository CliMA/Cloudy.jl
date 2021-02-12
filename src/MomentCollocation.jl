module MomentCollocation

using Cloudy.BasisFunctions
using QuadGK
using LinearAlgebra

export select_rbf_locs
export select_rbf_shapes
export get_rbf_inner_products
export get_IC_vecs
export get_kernel_rbf_sink
export get_kernel_rbf_source
export get_kernel_rbf_sink_precip
export get_injection_source
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
function select_rbf_shapes(rbf_locs::Array{FT}; smoothing_factor::FT = 1.5) where {FT <: Real}
    rbf_pts = append!([0.0], rbf_locs)
    rbf_sigma = zeros(length(rbf_locs))
    rbf_sigma[1] = rbf_locs[1]/smoothing_factor
    rbf_sigma[2:end] = (rbf_locs[2:end] - rbf_pts[1:end-2])/smoothing_factor
    return rbf_sigma
end

""" 
For conversion between collocation point and constants vector
"""
function get_rbf_inner_products(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT,1}, moment_list::Array{FT,1}) where {FT <: Real}
    # Φ_ij = Φ_i(x_j)
    Nb = length(basis)
    Nmom = length(moment_list)
    Φ = zeros(FT, Nb*Nmom, Nb)
    for j=1:Nb
        for i=1:Nb
            for (k, moment_order) in enumerate(moment_list)
                Φ[(k-1)*Nb+j,i] = rbf_locs[j]^moment_order * evaluate_rbf(basis[i], rbf_locs[j])
            end
        end
    end

    return Φ
end

"""
Calculating the initial condition vectors: non-mass conserving form
"""
function get_IC_vecs(u0::Function, basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT,1}, A::Array{FT}, moment_list::Array{FT, 1}) where {FT<:Real}
    # c0 is given by A*c0 = b, with b_i = u0(xi)
    Nb = length(basis)
    Nmom = length(moment_list)
    nj = Array{FT}(undef, Nb*Nmom)
    for i=1:Nb
        for (k, moment_order) in enumerate(moment_list)
            nj[(k-1)*Nb+i] = u0(rbf_locs[i])*rbf_locs[i]^moment_order
        end
    end
    c0 = get_constants_vec(nj, A)
    return (c0, nj)
end

function get_kernel_rbf_sink(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real}
    # N_ijk = <basis[k](x), basis[j](x'), K(x, x'), basis[i](x) dx' dx 
    Nb = length(basis)
    Nmom = length(moment_list)
    N = zeros(FT, Nb*Nmom, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = y -> basis_func(basis[k])(y)*kernel([rbf_locs[i], y])
                N[i,j,k] = basis_func(basis[j])(rbf_locs[i]) * quadgk(integrand, xstart, max(xstop - rbf_locs[i], xstart))[1]
            end
        end
    end
    for i=1:Nb
        for (l, moment_order) in enumerate(moment_list)
            N[(l-1)*Nb+i,:,:] = N[i,:,:] .* rbf_locs[i]^moment_order
        end
    end

    return N
end

"""
Collision kernel sink, with loss of particles that exceed size x_max from the system
"""
function get_kernel_rbf_sink_precip(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real}
    # N_ijk = <basis[k](x), basis[j](x'), K(x, x'), basis[i](x) dx' dx 
    Nb = length(basis)
    Nmom = length(moment_list)
    N = zeros(FT, Nb*Nmom, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = y -> basis_func(basis[k])(y)*kernel([rbf_locs[i], y])
                N[i,j,k] = basis_func(basis[j])(rbf_locs[i]) * quadgk(integrand, xstart, xstop)[1]
            end
        end
    end
    for i=1:Nb
        for (l, moment_order) in enumerate(moment_list)
            N[(l-1)*Nb+i,:,:] = N[i,:,:] .* rbf_locs[i]^moment_order
        end
    end

    return N
end

function get_kernel_rbf_source(basis::Array{PrimitiveUnivariateBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = eps()) where {FT <: Real}
    # M_ijk = 1/2 <basis[k](x-x'), basis[j](x'), K(x-x', x'), basis[i](x) dx' dx 
    Nb = length(basis)
    Nmom = length(moment_list)
    M = zeros(FT, Nb*Nmom, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                integrand = y-> basis_func(basis[j])(rbf_locs[i] - y)*basis_func(basis[k])(y)* kernel([rbf_locs[i] - y, y])
                M[i, j, k] = quadgk(integrand, xstart, rbf_locs[i])[1]
            end
        end
    end

    for i=1:Nb
        for (l, moment_order) in enumerate(moment_list)
            M[(l-1)*Nb+i,:,:] = M[i,:,:] .* rbf_locs[i]^moment_order
        end
    end

    return M
end

function get_injection_source(rbf_locs::Array{FT}, moment_list::Array{FT, 1}, injection_rate::Function) where {FT <: Real}
    Nb = length(rbf_locs)
    Nmom = length(moment_list)
    I = zeros(FT, Nb*Nmom)
    I[1:Nb] = injection_rate.(rbf_locs)
    for (l,moment_order) in enumerate(moment_list)
        I[(l-1)*Nb+1:l*Nb] = I[(l-1)*Nb+1:l*Nb] .* rbf_locs.^moment_order
    end

    return I
end

"""
Collision coalescence rate of change: non-mass conserving form
"""
function collision_coalescence(nj::Array{FT,1}, A::Array{FT,2}, M::Array{FT,3}, N::Array{FT,3}) where {FT <: Real}
    Nb_times_Nmom = length(nj)
    # first calculate c(t)
    c = get_constants_vec(nj, A)

    # time rate of change: dn/dt|_xj, t
    dndt = zeros(FT, Nb_times_Nmom)
    for i=1:Nb_times_Nmom
        dndt[i] = (1/2*c'*M[i,:,:]*c - c'*N[i,:,:]*c)
    end

    return dndt
end

"""
Collision coalescence rate of change with particle injection
"""
function collision_coalescence(nj::Array{FT,1}, A::Array{FT,2}, M::Array{FT,3}, N::Array{FT,3}, I::Array{FT,1}) where {FT <: Real}
    Nb_times_Nmom = length(nj)
    # first calculate c(t)
    c = get_constants_vec(nj, A)

    # time rate of change: dn/dt|_xj, t
    dndt = zeros(FT, Nb_times_Nmom)
    for i=1:Nb_times_Nmom
        dndt[i] = (1/2*c'*M[i,:,:]*c - c'*N[i,:,:]*c + I[i])
    end

    return dndt
end

""" 
Retrieves constants vector from a set of values at the collocation points
"""
function get_constants_vec(nj::Array{FT, 1}, A::Array{FT}) where {FT<:Real}
    return A\nj
end

end