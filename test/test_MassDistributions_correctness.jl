"Testing correctness of MassDistributions module."

using SpecialFunctions: gamma
using Cloudy.MassDistributions


# Gamma distribution
dist = Gamma(1.0, 1.0, 2.0)
@test (dist.n, dist.θ, dist.k) == (FT(1.0), FT(1.0), FT(2.0))

dist = Gamma(0.0, 1.0, 2.0)
@test (dist.n, dist.θ, dist.k) == (FT(0.0), FT(1.0), FT(2.0))

@test_throws Exception Gamma(-1.0, 2.0, 3.0)
@test_throws Exception Gamma(1.0, -2.0, 3.0)
@test_throws Exception Gamma(1.0, 2.0, -3.0)

# Exponential distribution
dist = Exponential(1.0, 1.0)
@test (dist.n, dist.θ) == (FT(1.0), FT(1.0))

dist = Exponential(0.0, 1.0)
@test (dist.n, dist.θ) == (FT(0.0), FT(1.0))

@test_throws Exception Exponential(-1.0, 2.)
@test_throws Exception Exponential(1.0, -2.)

# Calculate moments
dist = Gamma(1.0, 1.0, 2.0)
@test compute_moment(dist, 1) == FT(2.0)
@test compute_moment(dist, 2) == FT(6.0)
@test compute_moment(dist, 2/3) ≈ gamma(2+2/3)/gamma(2)

# Calculate density
dist = Gamma(1.0, 1.0, 2.0)
@test compute_density(dist, 0.0) == FT(0.0)
@test compute_density(dist, 3.0) == FT(3/gamma(2)*exp(-3))
@test_throws Exception compute_density(dist, -3.1)

dist = Exponential(1.0, 1.0)
@test compute_density(dist, 0.0) == FT(1.0)
@test compute_density(dist, 3.1) == FT(exp(-3.1))
@test_throws Exception compute_density(dist, -3.1)

# Update parameters
dist = Gamma(1.0, 1.0, 2.0)
update_params!(dist, Array{FT}([1.1, 2.0, 4.1]))
@test dist.n == 1.1
@test compute_moment(dist, 0) ≈ 1.1 atol=1e-6
@test compute_moment(dist, 1) ≈ 2.0 atol=1e-6
@test compute_moment(dist, 2) ≈ 4.1 atol=1e-6

update_params!(dist, Array{FT}([1.1, 2.423, 8.112]))
@test dist.n == 1.1
@test compute_moment(dist, 0) ≈ 1.1 atol=1e-6
@test compute_moment(dist, 1) ≈ 2.423 atol=1e-6
@test compute_moment(dist, 2) ≈ 8.112 atol=1e-6

dist = Exponential(1.0, 2.1)
update_params!(dist, Array{FT}([1.1, 2.0]))
@test dist.n == 1.1
@test compute_moment(dist, 0) ≈ 1.1 atol=1e-6
@test compute_moment(dist, 1) ≈ 2.0 atol=1e-6
