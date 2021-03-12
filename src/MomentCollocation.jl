module MomentCollocation

using Cloudy.BasisFunctions
using QuadGK
using LinearAlgebra
using Convex
using SCS

export select_rbf_locs
export select_rbf_shapes
export get_rbf_inner_products
export get_IC_vecs
export get_IC_vecs_massbasis
export get_kernel_rbf_sink
export get_kernel_rbf_source
export get_kernel_rbf_sink_precip
export get_injection_source
export get_basis_projection
export collision_coalescence
export get_constants_vec
export get_mass_cons_term

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
function get_rbf_inner_products(basis::Array{RBF, 1}, rbf_locs::Array{FT,1}, moment_list::Array{FT,1}) where {FT <: Real, RBF <: PrimitiveUnivariateBasisFunc}
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

function get_kernel_rbf_sink(basis::Array{GlobalBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real}
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
function get_kernel_rbf_sink_precip(basis::Array{GlobalBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = eps(), xstop::FT = 1e6) where {FT <: Real}
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

function get_kernel_rbf_sink_precip(basis::Array{CompactBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = 0.0, xstop::FT=1e6) where {FT <: Real}
    # N_ijk = <basis[k](x), basis[j](x'), K(x, x'), basis[i](x) dx' dx 
    Nb = length(basis)
    Nmom = length(moment_list)
    N = zeros(FT, Nb*Nmom, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            for k=1:Nb
                supp = get_support(basis[k])
                xstartk = max(xstart, supp[1])
                xstopk = min(supp[2], xstop)
                integrand = y -> basis_func(basis[k])(y)*kernel([rbf_locs[i], y])
                N[i,j,k] = basis_func(basis[j])(rbf_locs[i]) * quadgk(integrand, xstartk, xstopk)[1]
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

"""function get_kernel_rbf_source(basis::Array{GlobalBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = eps()) where {FT <: Real}
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
end"""

function get_kernel_rbf_source(basis::Array{CompactBasisFunc, 1}, rbf_locs::Array{FT}, moment_list::Array{FT, 1}, kernel::Function; xstart::FT = 0.0) where {FT <: Real}
    # M_ijk = 1/2 <basis[k](x-x'), basis[j](x'), K(x-x', x'), basis[i](x) dx' dx 
    Nb = length(basis)
    Nmom = length(moment_list)
    M = zeros(FT, Nb*Nmom, Nb, Nb)
    for i=1:Nb
        for j=1:Nb
            suppj = get_support(basis[j])
            supp_ji = (max(0.0, rbf_locs[i] - suppj[2]), max(0.0, rbf_locs[i] - suppj[1]))
            for k=1:Nb
                suppk = get_support(basis[k])
                xstartk = max(max(xstart, suppk[1]), supp_ji[1])
                xstopk = min(min(rbf_locs[i], suppk[2]), supp_ji[2])
                if xstartk >= xstopk
                    M[i,j,k] = 0
                else
                    integrand = y-> basis_func(basis[j])(rbf_locs[i] - y)*basis_func(basis[k])(y)* kernel([rbf_locs[i] - y, y])
                    M[i, j, k] = quadgk(integrand, xstart, rbf_locs[i])[1]
                end
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

function get_basis_projection(basis::Array{RBF,1}, rbf_locs::Array{FT}, A::Array{FT, 2}, moment_list::Array{FT, 1}, unprojected_fn::Function, xmax::FT) where {FT <: Real, RBF <: PrimitiveUnivariateBasisFunc}
    Nb = length(basis)
    Nmom = length(moment_list)
    f0 = zeros(FT, Nb*Nmom)
    f0[1:Nb] = unprojected_fn.(rbf_locs)
    for (l,moment_order) in enumerate(moment_list)
        f0[(l-1)*Nb+1:l*Nb] = f0[(l-1)*Nb+1:l*Nb] .* rbf_locs.^moment_order
    end
    # project the function onto the basis space, with mass conservation imposed
    J = get_mass_cons_term(basis, xstart = 0.0, xstop = xmax)
    m_fn = quadgk(x->unprojected_fn.(x)*x, 0.0, xmax)[1]
    c_fn = get_constants_vec(f0, A, J, m_fn)
    println("mass computed is ", m_fn)

    # evaluate the projected function at collocation points and return
    function f_projected(x)
        evaluate_rbf(basis, c_fn, x)
    end
    f0_proj = zeros(FT, Nb*Nmom)
    f0_proj[1:Nb] = f_projected.(rbf_locs)
    for (l,moment_order) in enumerate(moment_list)
        f0_proj[(l-1)*Nb+1:l*Nb] = f0_proj[(l-1)*Nb+1:l*Nb] .* rbf_locs.^moment_order
    end

    return (c_fn, f0_proj)
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

function get_mass_cons_term(basis::Array{GlobalBasisFunc, 1}; xstart::FT = eps(), xstop::FT=1e6) where {FT <: Real}
    Nb = length(basis)
    J = zeros(FT, Nb)
    for i=1:Nb
        integrand = x-> basis_func(basis[i])(x) * x
        J[i] = quadgk(integrand, xstart, xstop)[1]
    end
    
    return J
end

function get_mass_cons_term(basis::Array{CompactBasisFunc, 1}; xstart::FT = eps(), xstop::FT=1e6) where {FT <: Real}
    Nb = length(basis)
    J = zeros(FT, Nb)
    for i=1:Nb
        suppi = get_support(basis[i])
        integrand = x-> basis_func(basis[i])(x) * x
        J[i] = quadgk(integrand, max(0.0, suppi[1]), min(xstop, suppi[2]))[1]
    end
    
    return J
end

end