
using Cloudy.SuperParticleDistributions

rtol = 1e-3

# Gamma distribution
# Initialization
dist1 = GammaParticleDistribution(1.0, 2.0, 3.0)
@test (dist1.n, dist1.θ, dist1.k) == (FT(1.0), FT(3.0), FT(2.0))
@test_throws Exception GammaParticleDistribution(-1.0, 2.0, 3.0)
@test_throws Exception GammaParticleDistribution(1.0, -2.0, 3.0)
@test_throws Exception GammaParticleDistribution(1.0, 2.0, -3.0)
# Evaluate
@test dist1(1.0) == 0.07961459006375433

# ExponentialParticleDistribution
# Initialization
dist2 = ExponentialParticleDistribution(1.0, 2.0)
@test (dist2.n, dist2.θ) == (FT(1.0), FT(2.0))
@test_throws Exception ExponentialParticleDistribution(-1.0, 2.0)
@test_throws Exception ExponentialParticleDistribution(1.0, -2.0)
# Evaluate
@test dist2(1.0) == 0.3032653298563167

# moments <-> params
@test get_moments(dist1) == [1.0, 6.0, 54.0]
@test get_moments(dist2) == [1.0, 2.0]

moments1 = [10.0, 50.0, 300.0]
update_dist_from_moments!(dist1, moments1)
@test (dist1.n, dist1.k, dist1.θ) == (10.0, 5.0, 1.0)
@test_throws Exception update_dist_from_moments!(dist1, [10.0, 50.0])

moments2 = [10.0, 50.0]
update_dist_from_moments!(dist2, moments2)
@test (dist2.n, dist2.θ) == (10.0, 5.0)
@test_throws Exception update_dist_from_moments!(dist2, [10.0, 50.0, 300.0])
