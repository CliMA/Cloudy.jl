using Cloudy.ParticleDistributions
using Cloudy.ParticleDistributions: integrate_SimpsonEvenFast
using JET: @test_opt

rtol = 1e-3

# Initialization
@test_opt GammaPrimitiveParticleDistribution(1.0, 2.0, 3.0)
@test_opt ExponentialPrimitiveParticleDistribution(1.0, 2.0)
@test_opt LognormalPrimitiveParticleDistribution(1.0, 1.0, 1.0)
# Evaluations
dist1 = GammaPrimitiveParticleDistribution(1.0, 2.0, 3.0)
@test_opt dist1(1.0)
dist2 = ExponentialPrimitiveParticleDistribution(1.0, 2.0)
@test_opt dist2(1.0)
dist3 = LognormalPrimitiveParticleDistribution(1.0, 2.0, 3.0)
@test_opt dist3(1.0)

# moments <-> params
@test_opt get_moments(dist1)
moments1 = [10.0, 50.0, 300.0]
@test_opt update_dist_from_moments!(dist1, moments1)
@test_opt get_moments(dist2)
moments2 = [10.0, 50.0]
@test_opt update_dist_from_moments!(dist2, moments2)
@test_opt get_moments(dist3)
moments3 = [10.0, 50.0, 300.0]
@test_opt update_dist_from_moments!(dist3, moments3)

# moment source helper
@test_opt moment_source_helper(dist1, 1.0, 0.0, 1.2)
@test_opt moment_source_helper(dist2, 1.0, 0.0, 1.2)

# compute Nq 
pdists = (dist1, dist2, dist3)
@test_opt get_standard_N_q(pdists)

# integrate_SimpsonEvenFast
x = collect(range(1.0, 10.0, 100))
y = x .^ 2
@test_opt integrate_SimpsonEvenFast(x, y)
