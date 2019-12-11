"Testing correctness of Distributions module."

using SpecialFunctions: gamma, gamma_inc
using Cloudy.Distributions

import Cloudy.Distributions: nparams, get_params, update_params,
                             check_moment_consistency, moment_func, density_func

rtol = 1e-3

# Exponential distribution
# Initialization
dist = Exponential(1.0, 1.0)
@test (dist.n, dist.θ) == (FT(1.0), FT(1.0))
@test_throws Exception Exponential(-1.0, 2.)
@test_throws Exception Exponential(1.0, -2.)

# Getters and setters
@test nparams(dist) == 2
@test get_params(dist) == ([:n, :θ], [1.0, 1.0])
dist = update_params(dist, [1.0, 2.0])
@test get_params(dist) == ([:n, :θ], [1.0, 2.0])
@test_throws Exception update_params(dist, [-0.2, 1.1])
@test_throws Exception update_params(dist, [0.2, -1.1])

# Moments, moments, density
dist = Exponential(1.0, 2.0)
@test moment_func(dist)(1.0, 2.0, 0.0) == 1.0
@test moment(dist, 1.0) == 2.0
@test moment(dist, 0.0) == 1.0
@test moment(dist, 10.0) == 2.0^10.0 * gamma(11.0)
@test density_func(dist)(1.0, 2.0, 0.0) == 0.5
@test density_func(dist)([1.0, 1.0], [2.0, 2.0], [0.0, 3.1]) == [0.5,0.5 * exp(-3.1 / 2.0)]
@test density(dist, 0.0) == 0.5
@test density(dist, 3.1) == 0.5 * exp(-3.1 / 2.0)
@test_throws Exception density(dist, -3.1)

## Update params from moments
dist = update_params_from_moments(dist, [1.1, 2.0])
@test moment(dist, 0.0) ≈ 1.1 rtol=rtol
@test moment(dist, 1.0) ≈ 2.0 rtol=rtol


# Gamma distribution
# Initialization
dist = Gamma(1.0, 1.0, 2.0)
@test (dist.n, dist.θ, dist.k) == (FT(1.0), FT(1.0), FT(2.0))
@test_throws Exception Gamma(-1.0, 2.0, 3.0)
@test_throws Exception Gamma(1.0, -2.0, 3.0)
@test_throws Exception Gamma(1.0, 2.0, -3.0)

# Getters and settes
@test nparams(dist) == 3
@test get_params(dist) == ([:n, :θ, :k], [1.0, 1.0, 2.0])
dist = update_params(dist, [1.0, 2.0, 1.0])
@test get_params(dist) == ([:n, :θ, :k], [1.0, 2.0, 1.0])
@test_throws Exception update_params(dist, [-0.2, 1.1, 3.4])
@test_throws Exception update_params(dist, [0.2, -1.1, 3.4])
@test_throws Exception update_params(dist, [0.2, 1.1, -3.4])

# Moments, moments, density
dist = Gamma(1.0, 1.0, 2.0)
@test moment_func(dist)(1.0, 1.0, 2.0, 0.0) == 1.0
@test moment(dist, 0.0) == 1.0
@test moment(dist, 1.0) == 2.0
@test moment(dist, 2.0) == 6.0
@test moment_func(dist)(1.0, 1.0, 2.0, [0.0, 1.0, 2.0]) == [1.0, 2.0, 6.0]
@test moment(dist, 2/3) ≈ gamma(2+2/3)/gamma(2)
@test density_func(dist)(1.0, 1.0, 2.0, 0.0) == 0.0
@test density_func(dist)([1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [0.0, 3.0]) == [0.0, 3/gamma(2)*exp(-3)]
@test density(dist, 0.0) == 0.0
@test density(dist, 3.0) == 3/gamma(2)*exp(-3)
@test_throws Exception density(dist, -3.1)

# Update params from moments
dist = update_params_from_moments(dist, [1.1, 2.0, 4.1])
@test moment(dist, 0.0) ≈ 1.1 rtol=rtol
@test moment(dist, 1.0) ≈ 2.0 rtol=rtol
@test moment(dist, 2.0) ≈ 4.1 rtol=rtol
dist = update_params_from_moments(dist, [1.1, 2.423, 8.112])
@test moment(dist, 0.0) ≈ 1.1 rtol=rtol
@test moment(dist, 1.0) ≈ 2.423 rtol=rtol
@test moment(dist, 2.0) ≈ 8.112 rtol=rtol


# Mixture distributions
# Initialization
dist = Mixture(Exponential(1.0, 1.0), Exponential(2.0, 2.0))
@test typeof(dist.subdists) == Array{Distribution{FT}, 1}
@test length(dist.subdists) == 2

# Getters and setters
@test nparams(dist) == 4
@test get_params(dist) == ([[:n, :θ], [:n, :θ]], [[1.0, 1.0], [2.0, 2.0]])
dist = update_params(dist, [0.2, 0.4, 3.1, 4.1])
@test get_params(dist) == ([[:n, :θ], [:n, :θ]], [[0.2, 0.4], [3.1, 4.1]])
@test_throws Exception update_params(dist, [-0.2, 1.1, 1.1, 2.1])
@test_throws Exception update_params(dist, [0.2, -1.1, 0.1, 3.1])
@test_throws Exception update_params(dist, [0.2, 1.1, -0.1, 3.1])
@test_throws Exception update_params(dist, [0.2, 1.1, 0.1, -3.1])

# Moments, moments, density
dist = update_params(dist, [1.0, 1.0, 2.0, 2.0])
p1 = moment(Exponential(1.0, 1.0), 2.23)
p2 = moment(Exponential(2.0, 2.0), 2.23)
@test moment_func(dist)(reduce(vcat, get_params(dist)[2])..., 2.23) == p1 + p2
@test moment_func(dist)(reduce(vcat, get_params(dist)[2])..., [0.0, 1.0]) == [3.0, 5.0]
@test moment(dist, 2.23) == p1 + p2
@test moment(dist, 0.0) == 3.0
@test moment(dist, 1.0) == 5.0
@test moment(dist, 11.0) ≈ gamma(12) + 2.0 * 2.0^11 * gamma(12.0) rtol=rtol
@test density_func(dist)(reduce(vcat, get_params(dist)[2])..., 0.0) == 2.0
@test density(dist, 0.0) == 2.0
@test density(dist, 1.0) == exp(-1.0) + exp(-0.5)
@test_throws Exception density(dist, -3.1)

# Update params from moments
dist = update_params_from_moments(dist, [3.0 + 1e-6, 5.0 - 1e-6, 18.0, 102.0])
@test moment(dist, 0.0) ≈ 3.0 rtol=rtol
@test moment(dist, 1.0) ≈ 5.0 rtol=rtol
@test moment(dist, 2.0) ≈ 18.0 rtol=rtol
@test moment(dist, 3.0) ≈ 102.0 rtol=rtol
dist2 = update_params_from_moments(dist, [3.0, 4.9, 18.0, 102.0])
@test moment(dist2, 0.0) ≈ 3.0 rtol=rtol
@test moment(dist2, 1.0) ≈ 4.9 rtol=rtol
@test moment(dist2, 2.0) ≈ 18.0 rtol=rtol
@test moment(dist2, 3.0) ≈ 102.0 rtol=rtol
dist3 = update_params_from_moments(dist, [2.5, 4.9, 19.0, 104.0])
@test moment(dist3, 0.0) ≈ 2.5 rtol=1e-1
@test moment(dist3, 1.0) ≈ 4.9 rtol=1e-1
@test moment(dist3, 2.0) ≈ 19.0 rtol=1e-1
@test moment(dist3, 3.0) ≈ 104.0 rtol=1e-2
dist4 = update_params_from_moments(dist, [3.0, 4.9, 18.0, 102.0])
@test moment(dist4, 0.0) ≈ 3.0 rtol=rtol
@test moment(dist4, 1.0) ≈ 4.9 rtol=rtol
@test moment(dist4, 2.0) ≈ 18.0 rtol=rtol
@test moment(dist4, 3.0) ≈ 102.0 rtol=rtol


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
