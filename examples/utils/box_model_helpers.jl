using Interpolations
using LinearAlgebra

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