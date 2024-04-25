"""
  multi-moment bulk microphysics implementation of coalescence

  Includes 3 variations on how to solve the SCE:
    - Numerical approach (all numerical integration)
    - Analytical approach (all power-law derived except for the autoconversion integrals)
"""
module Coalescence

using ..ParticleDistributions
using ..KernelTensors
using ..KernelFunctions
using ..EquationTypes
import ..rflatten
import ..get_dist_moment_ind

using QuadGK
using LinearAlgebra
using StaticArrays

# method that computes source terms from microphysical parameterizations
export get_coal_ints

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
struct CoalescenceData{N, P, FT}
    "maximum number of moments that need to be precomputed before computing Q, R, S"
    N_mom_max::Int
    "number of 2d integrals that need to be precomputed before computing S"
    N_2d_ints::NTuple{N, Int}
    "mass thresholds of distributions for the computation of S term"
    dist_thresholds::NTuple{N, FT}
    "matrix containing coalescence tensors for pairs of distributions"
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}}
    
    function CoalescenceData(
        kernel::Union{CoalescenceTensor{P, FT}, SMatrix{N, N, CoalescenceTensor{P, FT}}},
        NProgMoms::NTuple{N, Int},
        dist_thresholds::Union{Nothing, NTuple{N, FT}} = nothing,
        norms::Tuple{FT, FT} = (FT(1), FT(1))
        ) where {N, P, FT <: Real}

        _kernels = ntuple(N * N) do i
            if kernel isa CoalescenceTensor
                get_normalized_kernel_tensor(kernel, norms)
            else
                get_normalized_kernel_tensor(kernel[i], norms)
            end
        end
        matrix_of_kernels = SMatrix{N, N, CoalescenceTensor{P, FT}}(_kernels)
        
        N_mom_max = maximum(NProgMoms) + (P - 1)
        N_2d_ints = ntuple(N) do i 
            if i < N
                (P - 1) + max(NProgMoms[i], NProgMoms[i + 1])
            else
                (P - 1) + NProgMoms[i]
            end
        end
        
        dist_thresholds = dist_thresholds === nothing ? ntuple(N) do i Inf end : ntuple(N) do i dist_thresholds[i] / norms[2] end

        new{N, P, FT}(
            N_mom_max,
            N_2d_ints,
            dist_thresholds,
            matrix_of_kernels,
        )
    end
end

"""
get_coal_ints(::AnalyticalCoalStyle, pdists::Array{ParticleDistribution{FT}}, coal_data::NamedTuple)

  - `pdists`: array of PSD subdistributions
  - `coal_data`: coalescence data struct
Updates the collision-coalescence integrals.
"""
function get_coal_ints(
    cs::AnalyticalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    coal_data::CoalescenceData{N, P, FT},
) where {N, P, FT <: Real}

    NProgMoms = map(pdists) do dist nparams(dist) end
    moments = get_moments(pdists, coal_data.N_mom_max)
    finite_2d_ints = get_finite_2d_integrals(pdists, coal_data.dist_thresholds, moments, coal_data.N_2d_ints)
    (; Q, R, S) = get_coalescence_integral_moment_qrs(cs, moments, NProgMoms, finite_2d_ints, coal_data.matrix_of_kernels)

    coal_ints = ntuple(N) do k
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
    moments::SMatrix{N, M, FT},
    N_2d_ints::NTuple{N, Int},
) where {N, M, FT <: Real}
    # for each distribution we compute a matrix of finite_2d_ints
    return ntuple(N) do i
        # finite_2d_ints is a symmetric matrix; 
        # first the upper diagonal (j <= k) is computed
        finite_2d_ints_upper_diag =  ntuple(N_2d_ints[i]) do j
            ntuple(N_2d_ints[i]) do k
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
        finite_2d_ints = ntuple(N_2d_ints[i]) do j
            ntuple(N_2d_ints[i]) do k
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

function get_coalescence_integral_moment_qrs(
    cs::AnalyticalCoalStyle,
    moments::SMatrix{N, M, FT},
    NProgMoms::NTuple{N, Int},
    finite_2d_ints::NTuple{N, SMatrix},
    # finite_2d_ints::NTuple{N, SMatrix{L, L, FT}},
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}},
) where {N, M, P, L, FT <: Real}
    return (; 
    Q = get_Q_coalescence_matrix(
        cs,
        moments,
        NProgMoms,
        matrix_of_kernels,
    ),
    R = get_R_coalescence_matrix(
        cs,
        moments,
        NProgMoms,
        matrix_of_kernels,
    ),
    S = get_S_coalescence_matrix(
        cs,
        moments,
        NProgMoms,
        finite_2d_ints,
        matrix_of_kernels,
    ))
end

function get_Q_coalescence_matrix(
    cs::AnalyticalCoalStyle, 
    moments::SMatrix{N, M, FT}, 
    NProgMoms::NTuple{N, Int}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}}
) where {N, M, P, FT <: Real}

    return ntuple(maximum(NProgMoms)) do i
        moment_order = i - 1
        SMatrix{N, N, FT}(rflatten(ntuple(N) do k
            ntuple(N) do j
                if k <= j || NProgMoms[k] <= moment_order
                    FT(0)
                else
                    Q_jk(cs, moment_order, j, k, moments, matrix_of_kernels)
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
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}}
) where {N, M, P, FT <: Real}
    return sum(ntuple(P) do a1
        a = a1 - 1
        sum(ntuple(P) do b1
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
    cs::AnalyticalCoalStyle, 
    moments::SMatrix{N, M, FT}, 
    NProgMoms::NTuple{N, Int}, 
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}}
) where {N, M, P, FT <: Real}
    
    return ntuple(maximum(NProgMoms)) do i
        moment_order = i - 1
        SMatrix{N, N, FT}(rflatten(ntuple(N) do k
            ntuple(N) do j
                if NProgMoms[k] <= moment_order
                    FT(0)
                else
                    R_jk(cs, moment_order, j, k, moments, matrix_of_kernels)
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
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}}
) where {N, M, P, FT <: Real}
    return sum(ntuple(P) do a1
        a = a1 - 1
        sum(ntuple(P) do b1
            b = b1 - 1
            matrix_of_kernels[j, k].c[a + 1, b + 1] * moments[j, a + 1] * moments[k, b + moment_order + 1]
        end)
    end)
end

function get_S_coalescence_matrix(
    cs::AnalyticalCoalStyle, 
    moments::SMatrix{N, M, FT}, 
    NProgMoms::NTuple{N, Int}, 
    # finite_2d_ints::NTuple{N, SMatrix{L, L, FT}},
    finite_2d_ints::NTuple{N, SMatrix},
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}},
) where {N, M, P, L, FT <: Real}

    return ntuple(maximum(NProgMoms)) do i
        moment_order = i - 1
        SMatrix{2, N, FT}(rflatten(ntuple(N) do k
            if k < N && NProgMoms[k] <= moment_order && NProgMoms[k + 1] <= moment_order
                (FT(0), FT(0))
            elseif k == N && NProgMoms[k] <= moment_order
                (FT(0), FT(0))
            else
                (S_1k(cs, moment_order, k, moments, finite_2d_ints, matrix_of_kernels),
                S_2k(cs, moment_order, k, moments, finite_2d_ints, matrix_of_kernels))
            end
        end))
    end
end

function S_1k(
    ::AnalyticalCoalStyle, 
    moment_order::Int,
    k::Int,
    moments::SMatrix{N, M, FT},
    # finite_2d_ints::NTuple{N, SMatrix{L, L, FT}},
    finite_2d_ints::NTuple{N, SMatrix},
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}}
) where {N, M, P, L, FT <: Real}
    return sum(ntuple(P) do a1
        a = a1 - 1
        sum(ntuple(P) do b1
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
    # finite_2d_ints::NTuple{N, SMatrix{L, L, FT}},
    finite_2d_ints::NTuple{N, SMatrix},
    matrix_of_kernels::SMatrix{N, N, CoalescenceTensor{P, FT}}
) where {N, M, P, L, FT <: Real}
    return sum(ntuple(P) do a1
        a = a1 - 1
        sum(ntuple(P) do b1
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
get_coal_ints(
    cs::NumericalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    kernel_func::CoalescenceKernelFunction{FT},
    )

    - pdists: array of PSD subdistributions 
    - kernel_func: K(x,y) function that determines rate of coalescence based on size of particles x, y
    
Updates the collision-coalescence integrals.
"""
function get_coal_ints(
    cs::NumericalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    kernel_func::CoalescenceKernelFunction{FT},
) where {N, FT <: Real}

    NProgMoms = [nparams(dist) for dist in pdists]
    (; Q, R, S) = get_coalescence_integral_moment_qrs(cs, pdists, kernel_func)
    
    coal_ints = ntuple(N) do k
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

function get_coalescence_integral_moment_qrs(
    cs::NumericalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    kernel_func::CoalescenceKernelFunction{FT},
) where {N, FT <: Real}
    return (;
    Q = get_Q_coalescence_matrix(cs, pdists, kernel_func),
    R = get_R_coalescence_matrix(cs, pdists, kernel_func),
    S = get_S_coalescence_matrix(cs, pdists, kernel_func),
    )
end

function get_Q_coalescence_matrix(
    ::NumericalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    kernel::CoalescenceKernelFunction{FT},
) where {N, FT <: Real}
    NProgMoms = map(pdists) do dist nparams(dist) end
    return ntuple(maximum(NProgMoms)) do i
        moment_order = i - 1
        SMatrix{N, N, FT}(rflatten(ntuple(N) do k
            ntuple(N) do j
                if k <= j || NProgMoms[k] <= moment_order
                    FT(0)
                else
                    quadgk(x -> q_integrand_outer(x, j, k, kernel, pdists, moment_order), 0.0, Inf; rtol = 1e-8, maxevals = 1000,)[1]
                end
            end
        end))
    end
end

function get_R_coalescence_matrix(
    ::NumericalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    kernel::CoalescenceKernelFunction{FT},
) where {N, FT <: Real}
    NProgMoms = map(pdists) do dist nparams(dist) end
    return ntuple(maximum(NProgMoms)) do i
        moment_order = i - 1
        SMatrix{N, N, FT}(rflatten(ntuple(N) do k
            ntuple(N) do j
                if NProgMoms[k] <= moment_order
                    FT(0)
                else
                    quadgk(x -> r_integrand_outer(x, j, k, kernel, pdists, moment_order), 0.0, Inf; rtol = 1e-8, maxevals = 1000, )[1]
                end
            end
        end))
    end
end

function get_S_coalescence_matrix(
    ::NumericalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    kernel::CoalescenceKernelFunction{FT},
) where {N, FT <: Real}
    NProgMoms = map(pdists) do dist nparams(dist) end
    return ntuple(maximum(NProgMoms)) do i
        moment_order = i - 1
        SMatrix{2, N, FT}(rflatten(ntuple(N) do k
            if k < N && NProgMoms[k] <= moment_order && NProgMoms[k + 1] <= moment_order
                (FT(0), FT(0))
            elseif k == N && NProgMoms[k] <= moment_order
                (FT(0), FT(0))
            else
                (quadgk(x -> s_integrand1(x, k, kernel, pdists, moment_order), 0.0, Inf; rtol = 1e-8, maxevals = 1000)[1],
                quadgk(x -> s_integrand2(x, k, kernel, pdists, moment_order), 0.0, Inf; rtol = 1e-8, maxevals = 1000)[1])
            end
        end))
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
