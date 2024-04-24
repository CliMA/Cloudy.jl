"""
  multi-moment bulk microphysics implementation of coalescence

  Includes 3 variations on how to solve the SCE:
    - Numerical approach (all numerical integration)
    - Analytical approach (all power-law derived)
    - Hybrid approach (analytical except for the autoconversion integrals)
"""
module Coalescence

using ..ParticleDistributions
using ..KernelTensors
using ..KernelFunctions
using ..EquationTypes
import ..get_dist_moment_ind

using QuadGK
using LinearAlgebra
using StaticArrays

# methods that compute source terms from microphysical parameterizations
export update_coal_ints!
export initialize_coalescence_data

"""
  CoalescenceData{N, FT}
   
Represents data needed for coalesce computations. N is the number of distributions.

# Constructor
  CoalescenceData(kernel, NProgMoms, dist_thresholds, norms)
  - `kernel`: Array of kernel tensors that determine rate of coalescence based on pair of distributions and size of particles x, y
  - `NProgMoms`: Array containing number of prognostic moments associated with each distribution
  - `dist_thresholds`: PSD upper thresholds for computing S terms
  - `norms`: a two-element tuple containing normalizing factors for number and mass
"""
struct CoalescenceData{N, FT}
    "maximum number of moments that need to be precomputed before computing Q, R, S"
    N_mom_max::Int
    "number of 2d integrals that need to be precomputed before computing S"
    N_2d_ints::NTuple{N, Int}
    "mass thresholds of distributions for the computation of S term"
    dist_thresholds::NTuple{N, FT}
    "matrix containing coalescence tensors for pairs of distributions"
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}}
    
    function CoalescenceData(
        kernel::Union{CoalescenceTensor{R, FT}, SMatrix{N, N, CoalescenceTensor{R, FT}}},
        NProgMoms::NTuple{N, Int},
        dist_thresholds::NTuple{N, FT} = nothing,
        norms::Tuple{FT, FT} = (FT(1), FT(1))
        ) where {N, FT <: Real}

        _kernels = ntuple(N * N) do i
            if kernel isa CoalescenceTensor
                get_normalized_kernel_tensor(kernel, norms)
            else
                get_normalized_kernel_tensor(kernel[i], norms)
            end
        end
        matrix_of_kernels = SMatrix{N, N, CoalescenceTensor{R, FT}}(_kernels)
        
        N_mom_max = maximum(NProgMoms) + R
        N_2d_ints = diag(kernels_order) .+ [i < Ndist ? max(NProgMoms[i], NProgMoms[i + 1]) : NProgMoms[i] for i in 1:Ndist]
        
        dist_thresholds = dist_thresholds === nothing ? ntuple(N) do i Inf end : ntuple(N) do i dist_thresholds[i] / norms[2] end

        new{N, FT}(
            N_mom_max,
            N_2d_ints,
            matrix_of_kernels,
            dist_thresholds,
        )
    end
end

"""
update_coal_ints!(::AnalyticalCoalStyle, pdists::Array{ParticleDistribution{FT}}, coal_data::NamedTuple)

  - `pdists`: array of PSD subdistributions
  - `coal_data`: Dictionary carried by ODE solver that contains all dynamical parameters, including the coalescence integrals
Updates the collision-coalescence integrals.
"""
function get_coal_ints(
    cs::AnalyticalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    coal_data::CoalescenceData{N, FT},
) where {N, FT <: Real}

    NProgMoms = [nparams(dist) for dist in pdists]
    moments = get_moments(pdists, coal_data.N_mom_max)
    finite_2d_ints = get_finite_2d_integrals(pdists, coal_data.dist_thresholds, moments, coal_data.N_2d_ints)
    (; Q, R, S) = get_coalescence_integral_moment_qrs(cs, moments, NProgMoms, coal_data.matrix_of_kernels, finite_2d_ints)

    coal_ints = ntuple(N_dist) do k
        ntuple(NProgMoms[k]) do m
            if k == 1
                sum(@views Q[m][:, k]) - sum(@views R[m][:, k]) + S[m][1, k]
            else
                sum(@views Q[m][:, k]) - sum(@views R[m][:, k]) + S[m][1, k] + S[m][2, k - 1]
            end
        end
    end
    return rflatten(coal_ints)
end

function get_moments(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, M::Int) where {N, FT <: Real}
    moments = ntuple(M) do j
        ntuple(N) do i
            moment(pdists[i], FT(j - 1))
        end
    end
    return SMatrix{N, M, FT}(rflatten(moments))
end

function get_finite_2d_integrals(
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    thresholds::NTuple{N, FT},
    moments::SMatrix{N, M, FT, L},
    N_2d_ints::NTuple{N, Int},
) where {N, M, FT, L <: Real}
    # for each distribution we compute a matrix of finite_2d_ints
    return ntuple(N) do i
        # finite_2d_ints is a symmetric matrix; 
        # first the upper diagonal (j <= k) is computed
        finite_2d_ints_upper_diag =  ntuple(N_2d_ints) do j
            ntuple(N_2d_ints) do k
                mom_times_mom = moments[i, j] * moments[i, k]
                if mom_times_mom < eps(FT) || k < j
                    FT(0)
                elseif i == N
                    mom_times_mom
                else
                    min(mom_times_mom, moment_source_helper(pdists[i], FT(j - 1), FT(k - 1), thresholds[i]))
                end
            end
        end

        # Then the full matrix is computed
        finite_2d_ints = ntuple(N_2d_ints) do j
            ntuple(N_2d_ints) do k
                if k < j
                    finite_2d_ints_upper_diag[k][j]
                else
                    finite_2d_ints_upper_diag[j][k]
                end
            end
        end
        
        # Conversion from tuple of tuples to SMatrix
        SMatrix{N_2d_ints[i], N_2d_ints[i], FT}(rflatten(finite_2d_ints))
    end
end

function get_coalescence_integral_moment_qrs!(
    cs::AnalyticalCoalStyle,
    moments::SMatrix{N, M, FT},
    NProgMoms::NTuple{N, Int},
    finite_2d_ints::NTuple{N, Int},
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}},
) where {N, M , FT <: Real}
    return (; 
    Q = get_Q_coalescence_matrix!(
        cs,
        moments,
        NProgMoms,
        matrix_of_kernels,
    ),
    R = get_R_coalescence_matrix!(
        cs,
        moments,
        NProgMoms,
        matrix_of_kernels,
    ),
    S = get_S_coalescence_matrix!(
        cs,
        moments,
        NProgMoms,
        finite_2d_ints,
        matrix_of_kernels,
    ))
end

function get_Q_coalescence_matrix(
    ::AnalyticalCoalStyle, 
    moments::SMatrix{N, M, FT}, 
    NProgMoms::NTuple{N, Int}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}}
) where {N, M, R, FT <: Real}

    return ntuple(M) do i
        moment_order = i - 1
        SMatrix{N, N, FT}(rflatten(ntuple(N) do k
            ntuple(N) do j
                if k <= j || NProgMoms[k] <= moment_order
                    FT(0)
                else
                    Q_jk(moment_order, j, k, moments, matrix_of_kernels)
                end
            end
        end))
    end
end

function Q_jk(
    ::AnalyticalCoalStyle, 
    moment_order::Int,
    j::Int,
    k::Int,
    moments::SMatrix{N, M, FT}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}}
) where {N, M, R, FT <: Real}
    return sum(ntuple(R + 1) do a1
        a = a1 - 1
        sum(ntuple(R + 1) do b1
            b = b1 - 1
            sum(ntuple(moment_order + 1) do c1
                c = c1 - 1
                matrix_of_kernels[j, k].c[a + 1, b + 1] *
                binomial(moment_order, c) *
                moments[j, a + c + 1] *
                moments[k, b + moment_order - c + 1]
            end)
        end)
    end)
end

function get_R_coalescence_matrix(
    ::AnalyticalCoalStyle, 
    moments::SMatrix{N, M, FT}, 
    NProgMoms::NTuple{N, Int}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}}
) where {N, M, R, FT <: Real}
    
    return ntuple(M) do i
        moment_order = i - 1
        SMatrix{N, N, FT}(rflatten(ntuple(N) do k
            ntuple(N) do j
                if NProgMoms[k] <= moment_order
                    FT(0)
                else
                    R_jk(moment_order, j, k, moments, matrix_of_kernels)
                end
            end
        end))
    end
end

function R_jk(
    ::AnalyticalCoalStyle, 
    moment_order::Int,
    j::Int,
    k::Int,
    moments::SMatrix{N, M, FT}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}}
) where {N, M, R, FT <: Real}
    return sum(ntuple(R + 1) do a1
        a = a1 - 1
        sum(ntuple(R + 1) do b1
            b = b1 - 1
            matrix_of_kernels[j, k].c[a + 1, b + 1] * moments[j, a + 1] * moments[k, b + moment_order + 1]
        end)
    end)
end

function get_S_coalescence_matrix(
    ::AnalyticalCoalStyle, 
    moments::SMatrix{N, M, FT}, 
    NProgMoms::NTuple{N, Int}, 
    finite_2d_ints::NTuple{N, Int},
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}},
) where {N, M, R, FT <: Real}

    return ntuple(M) do i
        moment_order = i - 1
        SMatrix{2, N, FT}(rflatten(ntuple(N) do k
            if k < N && NProgMoms[k] <= moment_order && NProgMoms[k + 1] <= moment_order
                (FT(0), FT(0))
            elseif k == N && NProgMoms[k] <= moment_order
                (FT(0), FT(0))
            else
                (S_1k(moment_order, k, moments, matrix_of_kernels),
                S_2k(moment_order, k, moments, matrix_of_kernels))
            end
        end))
    end
end

function S_1k(
    ::AnalyticalCoalStyle, 
    moment_order::Int,
    k::Int,
    moments::SMatrix{N, M, FT}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}}
) where {N, M, R, FT <: Real}
    return sum(ntuple(R + 1) do a1
        a = a1 - 1
        sum(ntuple(R + 1) do b1
            b = b1 - 1
            sum(ntuple(moment_order + 1) do c1
                c = c1 - 1
                0.5 * matrix_of_kernels[k, k].c[a + 1, b + 1] *
                binomial(moment_order, c) *
                finite_2d_ints[k][a + c + 1, b + moment_order - c + 1]
            end)
        end)
    end)
end

function S_2k(
    ::AnalyticalCoalStyle, 
    moment_order::Int,
    k::Int,
    moments::SMatrix{N, M, FT}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{R, FT}}
) where {N, M, R, FT <: Real}
    return sum(ntuple(R + 1) do a1
        a = a1 - 1
        sum(ntuple(R + 1) do b1
            b = b1 - 1
            sum(ntuple(moment_order + 1) do c1
                c = c1 - 1
                0.5 * matrix_of_kernels[k, k].c[a + 1, b + 1] *
                binomial(moment_order, c) *
                (moments[k, a + c + 1] *
                moments[k, b + moment_order - c + 1] - 
                finite_2d_ints[k][a + c + 1, b + moment_order - c + 1])
            end)
        end)
    end)
end


"""
update_coal_ints!(Nmom::FT, kernel_func::KernelFunction{FT}, pdists::Array{ParticleDistribution{FT}},
    coal_data::Dict)

Updates the collision-coalescence integrals.
Nmom: number of prognostic moments per particle distribution
kernel_func: K(x,y) function that determines rate of coalescence based on size of particles x, y
pdists: array of PSD subdistributions
coal_data: Dictionary carried by ODE solver that contains all dynamical parameters, including the 
    coalescence integrals
"""
function update_coal_ints!(
    cs::NumericalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    coal_data::NamedTuple,
) where {N, FT <: Real}

    NProgMoms = [nparams(dist) for dist in pdists]
    coal_data.coal_ints .= 0
    for m in 1:maximum(NProgMoms)
        get_coalescence_integral_moment_qrs!(cs, FT(m - 1), pdists, coal_data)
        for (k, pdist) in enumerate(pdists)
            if m > nparams(pdist)
                continue
            end
            ind = get_dist_moment_ind(NProgMoms, k, m)
            coal_data.coal_ints[ind] += sum(@views coal_data.Q[:, k])
            coal_data.coal_ints[ind] -= sum(@views coal_data.R[:, k])
            coal_data.coal_ints[ind] += coal_data.S[k, 1]
            if k > 1
                coal_data.coal_ints[ind] += coal_data.S[k - 1, 2]
            end
        end
    end
end


"""
initialize_coalescence_data(Ndist::FT, dist_moments_init::Array{FT})

Initializes the collision-coalescence integral matrices as zeros.
coal_ints contains all three matrices (Q, R, S) and the overall coal_int summation term
"""
function initialize_coalescence_data(
    ::NumericalCoalStyle,
    kernel_func::CoalescenceKernelFunction{FT},
    NProgMoms::Array{Int};
    norms = (FT(1), FT(1)),
) where {FT <: Real}
    Ndist = length(NProgMoms)
    Q = zeros(FT, Ndist, Ndist)
    R = zeros(FT, Ndist, Ndist)
    S = zeros(FT, Ndist, 2)
    coal_ints = zeros(FT, sum(NProgMoms))
    kernel = get_normalized_kernel_func(kernel_func, norms)
    return (Q = Q, R = R, S = S, coal_ints = coal_ints, kernel_func = kernel)
end

function get_coalescence_integral_moment_qrs!(
    cs::NumericalCoalStyle,
    moment_order::FT,
    pdists,
    coal_data,
) where {FT <: Real}
    update_Q_coalescence_matrix!(cs, moment_order, pdists, coal_data.kernel_func, coal_data.Q)
    update_R_coalescence_matrix!(cs, moment_order, pdists, coal_data.kernel_func, coal_data.R)
    update_S_coalescence_matrix!(cs, moment_order, pdists, coal_data.kernel_func, coal_data.S)
end

function update_Q_coalescence_matrix!(::NumericalCoalStyle, moment_order, pdists, kernel, Q)
    Ndist = length(pdists)
    for j in 1:Ndist
        for k in (j + 1):Ndist
            Q[j, k] = 0.0
            if nparams(pdists[k]) <= moment_order
                continue
            end
            Q[j, k] = quadgk(
                x -> q_integrand_outer(x, j, k, kernel, pdists, moment_order),
                0.0,
                Inf;
                rtol = 1e-8,
                maxevals = 1000,
            )[1]
        end
    end
end

function update_R_coalescence_matrix!(::NumericalCoalStyle, moment_order, pdists, kernel, R)
    Ndist = length(pdists)
    for j in 1:Ndist
        for k in 1:Ndist
            R[j, k] = 0.0
            if nparams(pdists[k]) <= moment_order
                continue
            end
            R[j, k] = quadgk(
                x -> r_integrand_outer(x, j, k, kernel, pdists, moment_order),
                0.0,
                Inf;
                rtol = 1e-8,
                maxevals = 1000,
            )[1]
        end
    end
end

function update_S_coalescence_matrix!(::NumericalCoalStyle, moment_order, pdists, kernel, S)
    Ndist = length(pdists)
    for j in 1:Ndist
        S[j, 1] = 0.0
        S[j, 2] = 0.0
        if j < Ndist
            if (nparams(pdists[j]) <= moment_order) & (nparams(pdists[j + 1]) <= moment_order)
                continue
            end
        else
            if nparams(pdists[j]) <= moment_order
                continue
            end
        end
        S[j, 1] =
            quadgk(x -> s_integrand1(x, j, kernel, pdists, moment_order), 0.0, Inf; rtol = 1e-8, maxevals = 1000)[1]
        S[j, 2] =
            quadgk(x -> s_integrand2(x, j, kernel, pdists, moment_order), 0.0, Inf; rtol = 1e-8, maxevals = 1000)[1]
    end
end

function weighting_fn(x::FT, k::Int64, pdists) where {FT <: Real}
    denom = 0.0
    num = 0.0
    Ndist = length(pdists)
    if k > Ndist
        throw(AssertionError("k out of range"))
    end
    for j in 1:Ndist
        denom += normed_density(pdists[j], x) #pdists[j](x) / pdists[j].n
        if j <= k
            num += normed_density(pdists[j], x) #pdists[j](x) / pdists[j].n
        end
    end
    if denom == 0.0
        return 0.0
    else
        return num / denom
    end
end

function q_integrand_inner(x, y, j, k, kernel, pdists)
    if j == k
        throw(AssertionError("q_integrand called on j==k, should call s instead"))
    elseif y > x
        throw(AssertionError("x <= y required in Q integrals"))
    end
    integrand = 0.5 * kernel(x - y, y) * (pdists[j](x - y) * pdists[k](y) + pdists[k](x - y) * pdists[j](y))
    return integrand
end

function q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    if j == k
        throw(AssertionError("q_integrand called on j==k, should call s instead"))
    end
    outer =
        x^moment_order *
        quadgk(yy -> q_integrand_inner(x, yy, j, k, kernel, pdists), 0.0, x; rtol = 1e-8, maxevals = 1000)[1]
    return outer
end

function r_integrand_inner(x, y, j, k, kernel, pdists)
    integrand = kernel(x, y) * pdists[k](x) * pdists[j](y)
    return integrand
end

function r_integrand_outer(x, j, k, kernel, pdists, moment_order)
    outer =
        x^moment_order *
        quadgk(yy -> r_integrand_inner(x, yy, j, k, kernel, pdists), 0.0, Inf; rtol = 1e-8, maxevals = 1000)[1]
    return outer
end

function s_integrand_inner(x, k, kernel, pdists, moment_order)
    integrand_inner = y -> 0.5 * kernel(x - y, y) * pdists[k](x - y) * pdists[k](y)
    integrand_outer = x .^ moment_order * quadgk(yy -> integrand_inner(yy), 0.0, x; rtol = 1e-8, maxevals = 1000)[1]
    return integrand_outer
end

function s_integrand1(x, k, kernel, pdists, moment_order)
    integrandj = weighting_fn(x, k, pdists) * s_integrand_inner(x, k, kernel, pdists, moment_order)
    return integrandj
end

function s_integrand2(x, k, kernel, pdists, moment_order)
    integrandk = (1 - weighting_fn(x, k, pdists)) * s_integrand_inner(x, k, kernel, pdists, moment_order)
    return integrandk
end

end # module
