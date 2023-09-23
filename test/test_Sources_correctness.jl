"Testing correctness of Sources module."

using Cloudy.ParticleDistributions
using Cloudy.Sources
using Cloudy.EquationTypes


rtol = 1e-3

# Types
CoalescenceStyle <: AbstractStyle
OneModeCoalStyle <: CoalescenceStyle
TwoModesCoalStyle <: CoalescenceStyle
OneModeCoalStyle() isa OneModeCoalStyle
TwoModesCoalStyle() isa TwoModesCoalStyle

# Constant kernel test (e.g, Smoluchowski 1916)
function sm1916(n_steps, δt; is_kernel_function = true, is_one_mode = true)
  # Parameters & initial condition
  kernel_func = x -> 1.0
  ker = (is_kernel_function == true) ? CoalescenceTensor(kernel_func, 0, 100.0) : CoalescenceTensor([1.0])

  # Initial condition
  mom = (is_one_mode) ? [1.0, 2.0] : [0.0, 0.0, 1.0, 2.0]
  dist_one_mode = Dict(:dist => [ExponentialPrimitiveParticleDistribution(1.0, 1.0)])
  dist_two_modes = Dict(:dist => [
    ExponentialPrimitiveParticleDistribution(0.0, 1.0), 
    ExponentialPrimitiveParticleDistribution(1.0, 1.0)])

  # Euler steps
  for i in 1:n_steps
    dmom = (is_one_mode) ? 
      get_int_coalescence(OneModeCoalStyle(), mom, dist_one_mode, ker) : 
      get_int_coalescence(TwoModesCoalStyle(), mom, dist_two_modes, ker)
    mom += δt * dmom
  end

  return mom
end

# Smoluchowski 1916 analytical result for 0th moment
function sm1916_ana(t, a, b)
  1 / (1/a + b/2*t)
end

n_steps = 5
δt = 1e-4
rtol = 1e-3
# Run tests
for i in 0:n_steps
  t = δt * i
  @test sm1916(n_steps, δt) ≈ Array{FT}([sm1916_ana(t, 1, 1), 2.0]) rtol=rtol
  @test sm1916(n_steps, δt, is_kernel_function = false) ≈ Array{FT}([sm1916_ana(t, 1, 1), 2.0]) rtol=rtol
  @test sm1916(n_steps, δt, is_one_mode = false) ≈ Array{FT}([0.0, 0.0, sm1916_ana(t, 1, 1), 2.0]) rtol=rtol
  @test sm1916(n_steps, δt, is_kernel_function = false, is_one_mode = false) ≈ Array{FT}([0.0, 0.0, sm1916_ana(t, 1, 1), 2.0]) rtol=rtol
end

# get_int_coalescence
# setup
ker = CoalescenceTensor(x -> 5e-3 * (x[1] + x[2]), 0, 100.0)
mom = [1, 0.1, 0.02, 1, 1, 2]
par1 = Dict(:dist => [GammaPrimitiveParticleDistribution(FT(1), FT(0.1), FT(1))])
par2 = Dict(
    :dist => [
      GammaPrimitiveParticleDistribution(FT(1), FT(0.1), FT(1)), 
      GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))],
    :x_th => 0.5)

# test
@test get_int_coalescence(OneModeCoalStyle(), mom[1:3], par1, ker) ≈ [-0.25, 0.0, 0.005] rtol = rtol
@test get_int_coalescence(TwoModesCoalStyle(), mom, par2, ker) ≈ [
  -0.75, 
  -0.05389, 
  -0.008031, 
  -0.25, 
  0.05389, 
  0.6130] rtol = rtol

# Sedimentation moment flux tests
c = [1.0, -1.0]
dist = Dict(:dist => ExponentialPrimitiveParticleDistribution(1.0, 1.0))
mom = [1.0, 1.0]
@test get_flux_sedimentation(mom, dist, c) ≈ [0.0, 1.0] rtol=rtol
