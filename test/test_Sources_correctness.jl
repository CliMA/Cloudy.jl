"Testing correctness of Sources module."

using Cloudy.ParticleDistributions
using Cloudy.Sources


rtol = 1e-3

# Constant kernel test (e.g, Smoluchowski 1916)
function sm1916_array(n_steps, δt)
  # Parameters & initial condition
  ker = CoalescenceTensor([1.0])

  # Initial condition
  mom = [1.0, 2.0]
  dist = Dict(:dist => Exponential(1.0, 1.0))

  # Euler steps
  for i in 1:n_steps
    dmom = get_int_coalescence(mom, dist, ker)
    mom += δt * dmom
  end

  return mom
end

function sm1916_func(n_steps, δt)
  # Parameters & initial condition
  kernel_func = x -> 1.0
  ker = CoalescenceTensor(kernel_func, 0, 100.0)

  # Initial condition
  mom = [1.0, 2.0]
  dist = Dict(:dist => Exponential(1.0, 1.0))

  # Euler steps
  for i in 1:n_steps
    dmom = get_int_coalescence(mom, dist, ker)
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
  @test sm1916_array(n_steps, δt) ≈ Array{FT}([sm1916_ana(t, 1, 1), 2.0]) rtol=rtol
  @test sm1916_func(n_steps, δt) ≈ Array{FT}([sm1916_ana(t, 1, 1), 2.0]) rtol=rtol
end

# Sedimentation moment flux tests
c = [1.0, -1.0]
dist = Dict(:dist => Exponential(1.0, 1.0))
mom = [1.0, 1.0]
@test get_flux_sedimentation(mom, dist, c) ≈ [0.0, 1.0] rtol=rtol
