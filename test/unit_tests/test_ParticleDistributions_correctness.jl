"Testing correctness of ParticleDistributions module."

using SpecialFunctions: gamma, gamma_inc
using Cloudy.ParticleDistributions

import Cloudy.ParticleDistributions:
    integrate_SimpsonEvenFast, check_moment_consistency, moment_func, density_func, 
    density, get_standard_N_q, compute_thresholds, compute_threshold
rtol = 1e-3

# Monodisperse distribution
# Initialization
dist = MonodispersePrimitiveParticleDistribution(1.0, 1.0)
@test (dist.n, dist.θ) == (FT(1.0), FT(1.0))
@test_throws Exception MonodispersePrimitiveParticleDistribution(-1.0, 2.0)
@test_throws Exception MonodispersePrimitiveParticleDistribution(1.0, -2.0)

# Getters and setters
@test nparams(dist) == 2
dist = MonodispersePrimitiveParticleDistribution(1.0, 2.0)
@test_throws Exception update_params(dist, [-0.2, 1.1])
@test_throws Exception update_params(dist, [0.2, -1.1])

# Moments, moments, density
dist = MonodispersePrimitiveParticleDistribution(1.0, 2.0)
@test moment_func(dist)(0.0) == 1.0
@test moment(dist, 1.0) == 2.0
@test moment(dist, 0.0) == 1.0
@test moment(dist, 10.0) == 2.0^10.0
@test density_func(dist)(0.0) == 0.0
@test density_func(dist)(2.0) == 2.5
@test density_func(dist)(3.1) == 0.0
@test density(dist, 0.0) == 0.0
@test density(dist, 2.0) == 2.5
@test density(dist, 3.1) == 0.0

## Update params from moments
dist = update_dist_from_moments(dist, (1.0, 1.0))
@test moment(dist, 0.0) ≈ 1.0 rtol = rtol
@test moment(dist, 1.0) ≈ 1.0 rtol = rtol
dist = update_dist_from_moments(dist, (1.1, 2.0))
@test moment(dist, 0.0) ≈ 1.1 rtol = rtol
@test moment(dist, 1.0) ≈ 2.0 rtol = rtol
dist = update_dist_from_moments(dist, (1.1, 0.0))
@test moment(dist, 0.0) ≈ 0.0 rtol = rtol
@test moment(dist, 1.0) ≈ 0.0 rtol = rtol


# Exponential distribution
# Initialization
dist = ExponentialPrimitiveParticleDistribution(1.0, 1.0)
@test (dist.n, dist.θ) == (FT(1.0), FT(1.0))
@test_throws Exception ExponentialPrimitiveParticleDistribution(-1.0, 2.0)
@test_throws Exception ExponentialPrimitiveParticleDistribution(1.0, -2.0)

# Getters and setters
@test nparams(dist) == 2
dist = ExponentialPrimitiveParticleDistribution(1.0, 2.0)
@test_throws Exception update_params(dist, [-0.2, 1.1])
@test_throws Exception update_params(dist, [0.2, -1.1])

# Moments, moments, density
dist = ExponentialPrimitiveParticleDistribution(1.0, 2.0)
@test moment_func(dist)(0.0) == 1.0
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
dist = update_dist_from_moments(dist, (1.1, 2.0))
@test normed_density(dist, 0.0) == 0.55
@test moment(dist, 0.0) ≈ 1.1 rtol = rtol
@test moment(dist, 1.0) ≈ 2.0 rtol = rtol
moments = (10.0, 50.0)
dist = update_dist_from_moments(dist, moments)
@test (dist.n, dist.θ) == (10.0, 5.0)
@test_throws Exception update_dist_from_moments(dist, (10.0, 50.0, 300.0))
dist = update_dist_from_moments(dist, (1.1, 0.0))
@test moment(dist, 0.0) ≈ 0.0 rtol = rtol
@test moment(dist, 1.0) ≈ 0.0 rtol = rtol


# Gamma distribution
# Initialization
dist = GammaPrimitiveParticleDistribution(1.0, 1.0, 2.0)
@test (dist.n, dist.θ, dist.k) == (FT(1.0), FT(1.0), FT(2.0))
@test_throws Exception GammaPrimitiveParticleDistribution(-1.0, 2.0, 3.0)
@test_throws Exception GammaPrimitiveParticleDistribution(1.0, -2.0, 3.0)
@test_throws Exception GammaPrimitiveParticleDistribution(1.0, 2.0, -3.0)

# Getters and settes
@test nparams(dist) == 3
dist = GammaPrimitiveParticleDistribution(1.0, 2.0, 1.0)

# Moments, moments, density
dist = GammaPrimitiveParticleDistribution(1.0, 1.0, 2.0)
@test moment_func(dist)(0.0) == 1.0
@test moment(dist, 0.0) == 1.0
@test moment(dist, 1.0) == 2.0
@test moment(dist, 2.0) == 6.0
@test get_moments(dist) == [1.0, 2.0, 6.0]
@test moment_func(dist).([0.0, 1.0, 2.0]) == [1.0, 2.0, 6.0]
@test moment(dist, 2 / 3) ≈ gamma(2 + 2 / 3) / gamma(2)
@test density_func(dist)(0.0) == 0.0
@test density_func(dist)(3.0) == 3 / gamma(2) * exp(-3)
@test density(dist, 0.0) == 0.0
@test density(dist, 3.0) == 3 / gamma(2) * exp(-3)
@test dist(0.0) == 0.0
@test dist(3.0) == 3 / gamma(2) * exp(-3)
@test_throws Exception density(dist, -3.1)

# Update params or dist from moments
dist = update_dist_from_moments(dist, (1.1, 2.0, 4.1); param_range = (; :k => (eps(Float64), 5.0)))
@test normed_density(dist, 1.0) ≈ 0.419 rtol = rtol
@test moment(dist, 0.0) ≈ 1.1 rtol = rtol
@test moment(dist, 1.0) ≈ 2.0 rtol = rtol
@test moment(dist, 2.0) ≈ 4.364 rtol = rtol
dist = update_dist_from_moments(dist, (1.1, 2.423, 8.112))
@test moment(dist, 0.0) ≈ 1.1 rtol = rtol
@test moment(dist, 1.0) ≈ 2.423 rtol = rtol
@test moment(dist, 2.0) ≈ 8.112 rtol = rtol
moments = (10.0, 50.0, 300.0)
dist = update_dist_from_moments(dist, moments)
@test (dist.n, dist.k, dist.θ) == (10.0, 5.0, 1.0)
@test_throws Exception update_dist_from_moments(dist, (10.0, 50.0))

# Lognormal distribution
# Initialization
dist = LognormalPrimitiveParticleDistribution(1.0, 1.0, 2.0)
@test (dist.n, dist.μ, dist.σ) == (FT(1.0), FT(1.0), FT(2.0))
@test_throws Exception LognormalPrimitiveParticleDistribution(-1.0, 2.0, 3.0)
@test_throws Exception LognormalPrimitiveParticleDistribution(1.0, 2.0, -3.0)

# Getters and settes
@test nparams(dist) == 3
dist = LognormalPrimitiveParticleDistribution(1.0, 2.0, 1.0)

# Moments, moments, density
dist = LognormalPrimitiveParticleDistribution(1.0, 1.0, 2.0)
@test moment_func(dist)(0.0) == 1.0
@test moment(dist, 0.0) == 1.0
@test moment(dist, 1.0) == exp(3.0)
@test moment(dist, 2.0) == exp(10.0)
@test get_moments(dist) == [1.0, exp(3.0), exp(10.0)]
@test moment(dist, 0.5) ≈ exp(1.0)
@test density_func(dist)(exp(1.0)) == 1 / 2.0 / sqrt(2 * π) / exp(1.0)
@test isnan(density_func(dist)(0.0))
@test isnan(density(dist, 0.0))
@test density(dist, exp(1.0)) == 1 / 2.0 / sqrt(2 * π) / exp(1.0)
@test isnan(dist(0.0))
@test dist(exp(1.0)) == 1 / 2.0 / sqrt(2 * π) / exp(1.0)
@test_throws Exception density(dist, -0.1)

# Update params or dist from moments
dist = update_dist_from_moments(dist, (1.1, 2.0, 4.1); param_range = (; :μ => (-1e5, 1e5), :σ => (eps(Float64), 5.0)))
@test normed_density(dist, 1.0) ≈ 0.3450 rtol = rtol
@test moment(dist, 0.0) ≈ 1.1 rtol = rtol
@test moment(dist, 1.0) ≈ 2.0 rtol = rtol
@test moment(dist, 2.0) ≈ 4.1 rtol = rtol
dist = update_dist_from_moments(dist, (1.1, 2.423, 8.112))
@test moment(dist, 0.0) ≈ 1.1 rtol = rtol
@test moment(dist, 1.0) ≈ 2.423 rtol = rtol
@test moment(dist, 2.0) ≈ 8.112 rtol = rtol
moments = (10.0, 50.0, 300.0)
dist = update_dist_from_moments(dist, moments)
@test dist.n ≈ 10.0 rtol = rtol
@test dist.μ ≈ 1.518 rtol = rtol
@test dist.σ ≈ 0.427 rtol = rtol
@test_throws Exception update_dist_from_moments(dist, (10.0, 50.0))


# Moment consistency checks
dist = update_dist_from_moments(dist, (1.1, 0.0, 8.112))
@test moment(dist, 0.0) ≈ 0.0 rtol = rtol
@test moment(dist, 1.0) ≈ 0.0 rtol = rtol
@test moment(dist, 2.0) ≈ 0.0 rtol = rtol

# Moment source helper
dist = MonodispersePrimitiveParticleDistribution(1.0, 0.5)
@test moment_source_helper(dist, 0.0, 0.0, 0.5) ≈ 0.0 rtol = rtol
@test moment_source_helper(dist, 0.0, 0.0, 1.2) ≈ 1.0 rtol = rtol
@test moment_source_helper(dist, 1.0, 0.0, 0.5) ≈ 0.0 rtol = rtol
@test moment_source_helper(dist, 0.0, 1.0, 1.2) ≈ 0.5 rtol = rtol
dist = ExponentialPrimitiveParticleDistribution(1.0, 0.5)
@test moment_source_helper(dist, 0.0, 0.0, 0.5, 20) ≈ 2.642e-1 rtol = rtol
@test moment_source_helper(dist, 1.0, 0.0, 0.5, 20) ≈ 4.015e-2 rtol = rtol
@test moment_source_helper(dist, 1.0, 1.0, 0.5, 20) ≈ 4.748e-3 rtol = rtol
dist = GammaPrimitiveParticleDistribution(1.0, 0.5, 2.0)
@test moment_source_helper(dist, 0.0, 0.0, 0.5, 20) ≈ 1.899e-2 rtol = rtol
@test moment_source_helper(dist, 1.0, 0.0, 0.5, 20) ≈ 3.662e-3 rtol = rtol
@test moment_source_helper(dist, 1.0, 1.0, 0.5, 20) ≈ 5.940e-4 rtol = rtol
dist = LognormalPrimitiveParticleDistribution(1.0, 0.5, 2.0)
@test moment_source_helper(dist, 0.0, 0.0, 2.5) ≈ 2.831e-1 rtol = rtol
@test moment_source_helper(dist, 1.0, 0.0, 2.5) ≈ 1.725e-1 rtol = rtol
@test moment_source_helper(dist, 1.0, 1.0, 2.5) ≈ 8.115e-2 rtol = rtol

# Moment consistency checks
m = (1.1, 2.1)
@test isnothing(check_moment_consistency(m))
m = (0.0, 0.0)
@test isnothing(check_moment_consistency(m))
m = (0.0, 1.0, 2.0)
@test isnothing(check_moment_consistency(m))
m = (1.0, 1.0, 2.0)
@test isnothing(check_moment_consistency(m))
m = (-0.1, 1.0)
@test_throws Exception check_moment_consistency(m)
m = (0.1, -1.0)
@test_throws Exception check_moment_consistency(m)
m = (1.0, 3.0, 2.0)
@test_throws Exception check_moment_consistency(m)

# get_standard_N_q
pdists = (ExponentialPrimitiveParticleDistribution(10.0, 1.0), GammaPrimitiveParticleDistribution(5.0, 10.0, 2.0))
Nq1 = get_standard_N_q(pdists, 1.0)
Nq2 = get_standard_N_q(pdists, 0.5)
@test Nq1.N_liq + Nq1.N_rai ≈ 15.0 rtol = rtol
@test Nq1.M_liq + Nq1.M_rai ≈ 110.0 rtol = rtol
@test Nq2.N_liq + Nq2.N_rai ≈ 15.0 rtol = rtol
@test Nq2.M_liq + Nq2.M_rai ≈ 110.0 rtol = rtol
@test Nq1.N_liq > Nq2.N_liq
@test Nq1.M_liq > Nq2.M_liq

# integrate_SimpsonEvenFast
Npt = 90
x = collect(range(1.0, 10.0, Npt + 1))
dx = x[2] - x[1]
yy(j) = x[j]^2
@test integrate_SimpsonEvenFast(Npt, dx, yy) ≈ 333.0 atol = 1e-6

# computing thresholds based on percentile
pdists = (ExponentialPrimitiveParticleDistribution(10.0, 1.0), GammaPrimitiveParticleDistribution(5.0, 10.0, 2.0))
@test compute_threshold(pdists[1], 0.75) > 1.0
@test compute_threshold(pdists[2], 0.75) > 2.0 * 10.0
@test compute_threshold(pdists[1], 0.0) ≈ 0.0 rtol = rtol
@test compute_threshold(pdists[2], 0.0) ≈ 0.0 rtol = rtol
@test compute_thresholds(pdists)[1] ≈ 3.507 rtol = rtol
@test compute_thresholds(pdists)[2] > 1e6
@test compute_thresholds(pdists, (0.5, 1.0))[1] ≈ 0.6931 rtol = rtol