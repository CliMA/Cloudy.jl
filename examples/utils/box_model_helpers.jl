using Interpolations
using LinearAlgebra
using SpecialFunctions

using Cloudy
using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources
using Cloudy.EquationTypes

const CPD = Cloudy.ParticleDistributions

"""
  make_box_model_rhs(coal_type::CoalescenceStyle)

  `coal_type` type of coal source term function: OneModeCoalStyle or TwoModesCoalStyle
Returns a function representing the right hand side of the ODE equation containing divergence 
of sedimentation flux and coalescence source term.
"""
function make_box_model_rhs(coal_type::CoalescenceStyle)

    rhs(m, par, t) = get_int_coalescence(coal_type, m, par, par[:kernel])
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
      return n/x0 * exp(-x/x0)
  end
  τ = 1 - exp(-n * b * x0 * t)
  sqrt_τ = sqrt(τ)
  return n * (1 - τ) / (x * sqrt_τ) * besselix(1, 2 * x / x0 * sqrt_τ) * exp(-(1 + τ - 2 * sqrt_τ) * x / x0)
end