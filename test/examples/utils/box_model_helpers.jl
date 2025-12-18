using LinearAlgebra
using SpecialFunctions

using Cloudy
using Cloudy.KernelFunctions
using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Coalescence
using Cloudy.Condensation
using Cloudy.EquationTypes

const CPD = Cloudy.ParticleDistributions

"""
  make_box_model_rhs(coal_type::CoalescenceStyle, threshold_style::ThresholdStyle)

  `coal_type` type of coal source term function: AnalyticalCoalStyle, NumericalCoalStyle
  `threshold_style` type of integration thresholds for analytical integration: Fixed or Moving
Returns a function representing the right hand side of the ODE equation containing divergence 
of coalescence source term.
"""
function make_box_model_rhs(coal_type::CoalescenceStyle, threshold_style::ThresholdStyle = FixedThreshold())
    rhs!(dm, m, par, t) = rhs_coal!(coal_type, dm, m, par, threshold_style)
end

function rhs_coal!(coal_type::CoalescenceStyle, dmom, mom, p, threshold_style)
    mom_norms = get_moments_normalizing_factors(p.NProgMoms, p.norms)
    mom_normalized = tuple(mom ./ mom_norms...)
    p = merge(p, (; pdists = ntuple(length(p.pdists)) do i
        ind_rng = get_dist_moments_ind_range(p.NProgMoms, i)
        update_dist_from_moments(p.pdists[i], mom_normalized[ind_rng])
    end))

    if coal_type isa AnalyticalCoalStyle
        if threshold_style isa FixedThreshold
            coal_ints = get_coal_ints(coal_type, p.pdists, p.coal_data)
        elseif threshold_style isa MovingThreshold
            coal_ints = get_coal_ints(coal_type, p.pdists, p.coal_data, threshold_style)
        end
    elseif coal_type isa NumericalCoalStyle
        coal_ints = get_coal_ints(coal_type, p.pdists, p.kernel_func)
    else
        error("Invalid coal style!")
    end

    dmom .= coal_ints .* mom_norms
end

function rhs_condensation!(dmom, mom, p, s)
    mom_norms = get_moments_normalizing_factors(p.NProgMoms, p.norms)
    mom_normalized = tuple(mom ./ mom_norms...)
    p = merge(p, (; pdists = ntuple(length(p.pdists)) do i
        ind_rng = get_dist_moments_ind_range(p.NProgMoms, i)
        update_dist_from_moments(p.pdists[i], mom_normalized[ind_rng])
    end))
    ξ_normalized = p.ξ / p.norms[2]^(2 / 3)
    dmom .= get_cond_evap(p.pdists, s, ξ_normalized) .* mom_norms
end

"""
  golovin_analytical_solution(x, x0, t; b = 1.5e-3, n = 1)

  `x` particle mass
  `x0` initial exponential distribution parameter (mass reference)
  `t` time
  `b` golovin kernel parameter
  `n` initial exponential distribution parameter (initial number density)
Returns analytical solution for Golovin kernel K(x, x') = b(x+x').
"""
function golovin_analytical_solution(x, x0, t; b = 1.5e-3, n = 1)
    FT = eltype(x)
    if t < eps(FT)
        return n / x0 * exp(-x / x0)
    end
    τ = 1 - exp(-n * b * x0 * t)
    sqrt_τ = sqrt(τ)
    return n * (1 - τ) / (x * sqrt_τ) * besselix(1, 2 * x / x0 * sqrt_τ) * exp(-(1 + τ - 2 * sqrt_τ) * x / x0)
end
