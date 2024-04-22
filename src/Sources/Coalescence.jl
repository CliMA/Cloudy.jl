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


# methods that compute source terms from microphysical parameterizations
export update_coal_ints!
export initialize_coalescence_data

"""
update_coal_ints!(::AnalyticalCoalStyle, pdists::Array{ParticleDistribution{FT}}, coal_data::NamedTuple)

  - `pdists`: array of PSD subdistributions
  - `coal_data`: Dictionary carried by ODE solver that contains all dynamical parameters, including the coalescence integrals
Updates the collision-coalescence integrals.
"""
function update_coal_ints!(
    cs::AnalyticalCoalStyle,
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    coal_data::NamedTuple,
) where {N, FT <: Real}

    update_moments!(pdists, coal_data.moments)
    update_finite_2d_integrals!(pdists, coal_data.dist_thresholds, coal_data.moments, coal_data.finite_2d_ints)

    NProgMoms = [nparams(dist) for dist in pdists]
    coal_data.coal_ints .= 0
    for m in 1:maximum(NProgMoms)
        get_coalescence_integral_moment_qrs!(cs, m - 1, NProgMoms, coal_data)
        for (k, pdist) in enumerate(pdists)
            if m > NProgMoms[k]
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
initialize_coalescence_data(
    ::AnalyticalCoalStyle, 
    kernel::Union{CoalescenceTensor{FT}, Matrix{CoalescenceTensor{FT}}},
    NProgMoms;
    dist_thresholds = nothing,)

  - `kernel`: Array of kernel tensors that determine rate of coalescence based on pair of distributions and size of particles x, y
  - `NProgMoms`: Array containing number of prognostic moments associated with each distribution
  - `dist_thresholds`: PSD upper thresholds for computing S terms
Initializes the coalescence data.
"""
function initialize_coalescence_data(
    ::AnalyticalCoalStyle,
    kernel::Union{CoalescenceTensor{N, FT, M}, Matrix{CoalescenceTensor{N, FT, M}}},
    NProgMoms::Array{Int};
    dist_thresholds = nothing,
    norms = (1.0, 1.0),
) where {N, M, FT <: Real}
    Ndist = length(NProgMoms)
    matrix_of_kernels = Array{CoalescenceTensor{N, FT, M}}(undef, Ndist, Ndist)
    if kernel isa CoalescenceTensor
        matrix_of_kernels .= get_normalized_kernel_tensor(kernel, norms)
    else
        @assert size(kernel) == (Ndist, Ndist)
        for (k, ker) in enumerate(kernel)
            matrix_of_kernels[k] = get_normalized_kernel_tensor(ker, norms)
        end
    end

    Q = zeros(FT, Ndist, Ndist)
    R = zeros(FT, Ndist, Ndist)
    S = zeros(FT, Ndist, 2)

    kernels_order = [N for ker in matrix_of_kernels]
    Nmom = maximum(NProgMoms) + maximum(kernels_order)
    moments = zeros(FT, Ndist, Nmom)

    N_2d_ints = diag(kernels_order) .+ [i < Ndist ? max(NProgMoms[i], NProgMoms[i + 1]) : NProgMoms[i] for i in 1:Ndist]
    finite_2d_ints = [zeros(FT, N_2d_ints[i], N_2d_ints[i]) for i in 1:Ndist]

    coal_ints = zeros(FT, sum(NProgMoms))

    dist_thresholds = dist_thresholds == nothing ? ones(FT, Ndist) * Inf : dist_thresholds ./ norms[2]
    @assert length(dist_thresholds) == Ndist

    return (
        Q = Q,
        R = R,
        S = S,
        moments = moments,
        finite_2d_ints = finite_2d_ints,
        coal_ints = coal_ints,
        matrix_of_kernels = matrix_of_kernels,
        dist_thresholds = dist_thresholds,
    )
end

function update_moments!(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, moments) where {N, FT <: Real}
    Ndist, Nmom = size(moments)
    for i in 1:Ndist
        for j in 1:Nmom
            moments[i, j] = moment(pdists[i], FT(j - 1))
        end
    end
end

function update_finite_2d_integrals!(
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    thresholds,
    moments,
    finite_2d_ints,
) where {N, FT <: Real}
    Ndist = size(moments)[1]
    for i in 1:Ndist
        N_2d_ints = size(finite_2d_ints[i])[1]
        for j in 1:N_2d_ints
            for k in j:N_2d_ints
                mom_times_mom = moments[i, j] * moments[i, k]
                if mom_times_mom < eps(FT)
                    finite_2d_ints[i][j, k] = FT(0)
                else
                    if i < Ndist
                        finite_2d_ints[i][j, k] =
                            min(mom_times_mom, moment_source_helper(pdists[i], FT(j - 1), FT(k - 1), thresholds[i]))
                    else
                        finite_2d_ints[i][j, k] = mom_times_mom
                    end
                end
                finite_2d_ints[i][k, j] = finite_2d_ints[i][j, k]
            end
        end
    end
end

function get_coalescence_integral_moment_qrs!(
    cs::AnalyticalCoalStyle,
    moment_order::Int,
    NProgMoms::Vector{Int},
    coal_data,
)
    order = size(coal_data.matrix_of_kernels[1].c)[1] - 1
    update_Q_coalescence_matrix!(
        cs,
        moment_order,
        coal_data.moments,
        order,
        coal_data.matrix_of_kernels,
        NProgMoms,
        coal_data.Q,
    )
    update_R_coalescence_matrix!(
        cs,
        moment_order,
        coal_data.moments,
        order,
        coal_data.matrix_of_kernels,
        NProgMoms,
        coal_data.R,
    )
    update_S_coalescence_matrix!(
        cs,
        moment_order,
        coal_data.moments,
        order,
        coal_data.finite_2d_ints,
        coal_data.matrix_of_kernels,
        NProgMoms,
        coal_data.S,
    )
end

function update_Q_coalescence_matrix!(
    ::AnalyticalCoalStyle,
    moment_order,
    moments,
    order,
    matrix_of_kernels,
    NProgMoms,
    Q,
)
    Ndist = size(moments)[1]

    for j in 1:Ndist
        for k in (j + 1):Ndist
            Q[j, k] = 0.0
            if NProgMoms[k] <= moment_order
                continue
            end
            r = order
            for a in 0:r
                for b in 0:r
                    for c in 0:moment_order
                        Q[j, k] +=
                            matrix_of_kernels[j, k].c[a + 1, b + 1] *
                            binomial(moment_order, c) *
                            moments[j, a + c + 1] *
                            moments[k, b + moment_order - c + 1]
                    end
                end
            end
        end
    end
end

function update_R_coalescence_matrix!(
    ::AnalyticalCoalStyle,
    moment_order,
    moments,
    order,
    matrix_of_kernels,
    NProgMoms,
    R,
)
    Ndist = size(moments)[1]

    for j in 1:Ndist
        for k in 1:Ndist
            R[j, k] = 0.0
            if NProgMoms[k] <= moment_order
                continue
            end
            r = order
            for a in 0:r
                for b in 0:r
                    R[j, k] +=
                        matrix_of_kernels[j, k].c[a + 1, b + 1] * moments[j, a + 1] * moments[k, b + moment_order + 1]
                end
            end
        end
    end
end

function update_S_coalescence_matrix!(
    ::AnalyticalCoalStyle,
    moment_order,
    moments,
    order,
    finite_2d_ints,
    matrix_of_kernels,
    NProgMoms,
    S,
)
    Ndist = size(moments)[1]

    for k in 1:Ndist
        S[k, 1] = 0.0
        S[k, 2] = 0.0
        if k < Ndist
            if (NProgMoms[k] <= moment_order) & (NProgMoms[k + 1] <= moment_order)
                continue
            end
        else
            if NProgMoms[k] <= moment_order
                continue
            end
        end
        r = order
        for a in 0:r
            for b in 0:r
                for c in 0:moment_order
                    _s1 =
                        0.5 *
                        matrix_of_kernels[k, k].c[a + 1, b + 1] *
                        binomial(moment_order, c) *
                        finite_2d_ints[k][a + c + 1, b + moment_order - c + 1]
                    S[k, 1] += _s1
                    S[k, 2] +=
                        0.5 *
                        matrix_of_kernels[k, k].c[a + 1, b + 1] *
                        binomial(moment_order, c) *
                        moments[k, a + c + 1] *
                        moments[k, b + moment_order - c + 1] - _s1
                end
            end
        end
    end
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
    norms = (1.0, 1.0),
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
