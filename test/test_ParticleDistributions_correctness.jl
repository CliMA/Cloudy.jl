"Testing correctness of ParticleDistributions module."

using SpecialFunctions: gamma, gamma_inc
using Cloudy.ParticleDistributions

import Cloudy.ParticleDistributions: nparams, get_params, update_params,
                                     check_moment_consistency, moment_func, 
                                     density_func, density
rtol = 1e-3

# Exponential distribution
# Initialization
dist = ExponentialPrimitiveParticleDistribution(1.0, 1.0)
@test (dist.n, dist.θ) == (FT(1.0), FT(1.0))
@test_throws Exception ExponentialPrimitiveParticleDistribution(-1.0, 2.)
@test_throws Exception ExponentialPrimitiveParticleDistribution(1.0, -2.)

# Getters and setters
@test nparams(dist) == 2
@test get_params(dist) == ([:n, :θ], [1.0, 1.0])
dist = update_params(dist, [1.0, 2.0])
@test get_params(dist) == ([:n, :θ], [1.0, 2.0])
@test_throws Exception update_params(dist, [-0.2, 1.1])
@test_throws Exception update_params(dist, [0.2, -1.1])

# Moments, moments, density
dist = ExponentialPrimitiveParticleDistribution(1.0, 2.0)
@test moment_func(dist)(1.0, 2.0, 0.0) == 1.0
@test moment(dist, 1.0) == 2.0
@test moment(dist, 0.0) == 1.0
@test get_moments(dist) == [1.0, 2.0]
@test moment(dist, 10.0) == 2.0^10.0 * gamma(11.0)
@test density_func(dist)(3.1) == 0.5 * exp(-3.1 / 2.0)
@test density_func(dist)(0.0) == 0.5
@test density(dist, 0.0) == 0.5
@test density(dist, 3.1) == 0.5 * exp(-3.1 / 2.0)
@test dist(0.0) == 0.5
@test dist(3.1) == 0.5 * exp(-3.1 / 2.0)
@test_throws Exception density(dist, -3.1)

## Update params or dist from moments
dist_dict = Dict(:dist => dist)
dist = update_params_from_moments(dist_dict, [1.1, 2.0])
@test moment(dist, 0.0) ≈ 1.1 rtol=rtol
@test moment(dist, 1.0) ≈ 2.0 rtol=rtol
moments = [10.0, 50.0]
update_dist_from_moments!(dist, moments)
@test (dist.n, dist.θ) == (10.0, 5.0)
@test_throws Exception update_dist_from_moments!(dist, [10.0, 50.0, 300.0])


# Gamma distribution
# Initialization
dist = GammaPrimitiveParticleDistribution(1.0, 1.0, 2.0)
@test (dist.n, dist.θ, dist.k) == (FT(1.0), FT(1.0), FT(2.0))
@test_throws Exception GammaPrimitiveParticleDistribution(-1.0, 2.0, 3.0)
@test_throws Exception GammaPrimitiveParticleDistribution(1.0, -2.0, 3.0)
@test_throws Exception GammaPrimitiveParticleDistribution(1.0, 2.0, -3.0)

# Getters and settes
@test nparams(dist) == 3
@test get_params(dist) == ([:n, :θ, :k], [1.0, 1.0, 2.0])
dist = update_params(dist, [1.0, 2.0, 1.0])
@test get_params(dist) == ([:n, :θ, :k], [1.0, 2.0, 1.0])
@test_throws Exception update_params(dist, [-0.2, 1.1, 3.4])
@test_throws Exception update_params(dist, [0.2, -1.1, 3.4])
@test_throws Exception update_params(dist, [0.2, 1.1, -3.4])

# Moments, moments, density
dist = GammaPrimitiveParticleDistribution(1.0, 1.0, 2.0)
@test moment_func(dist)(1.0, 1.0, 2.0, 0.0) == 1.0
@test moment(dist, 0.0) == 1.0
@test moment(dist, 1.0) == 2.0
@test moment(dist, 2.0) == 6.0
@test get_moments(dist) == [1.0, 2.0, 6.0]
@test moment_func(dist)(1.0, 1.0, 2.0, [0.0, 1.0, 2.0]) == [1.0, 2.0, 6.0]
@test moment(dist, 2/3) ≈ gamma(2+2/3)/gamma(2)
@test density_func(dist)(0.0) == 0.0
@test density_func(dist)(3.0) == 3/gamma(2)*exp(-3)
@test density(dist, 0.0) == 0.0
@test density(dist, 3.0) == 3/gamma(2)*exp(-3)
@test dist(0.0) == 0.0
@test dist(3.0) == 3/gamma(2)*exp(-3)
@test_throws Exception density(dist, -3.1)

# Update params or dist from moments
dist_dict = Dict(:dist => dist)
dist = update_params_from_moments(dist_dict, [1.1, 2.0, 4.1], Dict("θ" => (1e-5, 1e5), "k" => (eps(Float64), 5.0)))
@test moment(dist, 0.0) ≈ 1.726 rtol=rtol
@test moment(dist, 1.0) ≈ 2.0 rtol=rtol
@test moment(dist, 2.0) ≈ 2.782 rtol=rtol
dist = update_params_from_moments(dist_dict, [1.1, 2.423, 8.112])
@test moment(dist, 0.0) ≈ 1.1 rtol=rtol
@test moment(dist, 1.0) ≈ 2.423 rtol=rtol
@test moment(dist, 2.0) ≈ 8.112 rtol=rtol
moments = [10.0, 50.0, 300.0]
update_dist_from_moments!(dist, moments)
@test (dist.n, dist.k, dist.θ) == (10.0, 5.0, 1.0)
@test_throws Exception update_dist_from_moments!(dist, [10.0, 50.0])


# Moment consitency checks
m = [1.1, 2.1]
@test check_moment_consistency(m) == nothing
m = [0.0, 0.0]
@test check_moment_consistency(m) == nothing
m = [0.0, 1.0, 2.0]
@test check_moment_consistency(m) == nothing
m = [1.0, 1.0, 2.0]
@test check_moment_consistency(m) == nothing
m = [-0.1, 1.0]
@test_throws Exception check_moment_consistency(m)
m = [0.1, -1.0]
@test_throws Exception check_moment_consistency(m)
m = [1.0, 3.0, 2.0]
@test_throws Exception check_moment_consistency(m)
