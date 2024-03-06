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
  make_box_model_rhs(coal_type::CoalescenceStyle)

  `coal_type` type of coal source term function: AnalyticalCoalStyle, NumericalCoalStyle
Returns a function representing the right hand side of the ODE equation containing divergence 
of coalescence source term.
"""
function make_box_model_rhs(coal_type::CoalescenceStyle)
    rhs!(dm, m, par, t) = rhs_coal!(coal_type, dm, m, par)
end

function rhs_coal!(coal_type::CoalescenceStyle, dmom, mom, p)
    mom_norms = get_moments_normalizing_factors(p.NProgMoms, p.norms)
    mom_normalized = mom ./ mom_norms
    for (i, dist) in enumerate(p.pdists)
        ind_rng = get_dist_moments_ind_range(p.NProgMoms, i)
        update_dist_from_moments!(dist, mom_normalized[ind_rng])
    end
    update_coal_ints!(coal_type, p.pdists, p.coal_data)
    dmom .= p.coal_data.coal_ints .* mom_norms
end

function rhs_condensation!(dmom, mom, p, s)
    mom_norms = get_moments_normalizing_factors(p.NProgMoms, p.norms)
    mom_normalized = mom ./ mom_norms
    for (i, dist) in enumerate(p.pdists)
        ind_rng = get_dist_moments_ind_range(p.NProgMoms, i)
        update_dist_from_moments!(dist, mom_normalized[ind_rng])
    end
    dmom .= get_cond_evap(s, p) .* mom_norms
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
