"Testing correctness of helper functions"

using Cloudy

NProgMoms = (2, 2, 3)
@test get_dist_moment_ind(NProgMoms, 1, 2) == 2
@test get_dist_moment_ind(NProgMoms, 2, 1) == 3
@test get_dist_moment_ind(NProgMoms, 3, 2) == 6
@test_throws Exception get_dist_moment_ind(NProgMoms, 4, 2)
@test_throws Exception get_dist_moment_ind(NProgMoms, 2, 0)
@test_throws Exception get_dist_moment_ind(NProgMoms, 3, 4)

@test get_dist_moments_ind_range(NProgMoms, 1) == 1:2
@test get_dist_moments_ind_range(NProgMoms, 3) == 5:7
@test_throws Exception get_dist_moments_ind_range(NProgMoms, 4)

norm_factors = get_moments_normalizing_factors(NProgMoms, (10.0, 0.1))
norm_factors_ = (10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 0.1)
for i in 1:7
    @test norm_factors[i] ≈ norm_factors_[i] atol = 1e-12
end

tmp_tuple = ((1,2), (3.2, (1.2, 1.0)), (1),)
@test Cloudy.rflatten(tmp_tuple) == (1, 2, 3.2, 1.2, 1.0, 1)
