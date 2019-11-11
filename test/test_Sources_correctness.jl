"Testing correctness of Sources module."

using Cloudy.MassDistributions
using Cloudy.Sources

# Constant kernel test (e.g, Smoluchowski 1916)
function sm1916(n_steps, δt)
  # Parameters & initial condition
  ker = ConstantCoalescenceTensor(1.0)

  # Initial condition
  mom = Array{FT}([1.0, 2.0])
  dist = Exponential(1.0, 1.0)
  update_params!(dist, mom)

  # Euler steps
  for i in 1:n_steps
    dmom = get_src_coalescence(mom, dist, ker)
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
# Run tests
for i in 0:n_steps
  t = δt * i
  @test sm1916(n_steps, δt) ≈ Array{FT}([sm1916_ana(t, 1, 1), 2.0]) atol=1e-3
end
