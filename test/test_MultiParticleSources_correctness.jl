using Cloudy.SuperParticleDistributions
using Cloudy.KernelFunctions
using Cloudy.MultiParticleSources: weighting_fn, q_integrand_inner,
    q_integrand_outer, r_integrand_inner, r_integrand_outer, 
    s_integrand1, s_integrand2, s_integrand_inner,
    update_R_coalescence_matrix!, update_S_coalescence_matrix!,
    update_Q_coalescence_matrix!, get_coalescence_integral_moment_qrs!,
    initialize_coalescence_data, update_coal_ints!
rtol = 1e-4

# weighting function
dist1 = GammaParticleDistribution(10.0, 3.0, 100.0)
pdists = [dist1]
@test weighting_fn(10.0, 1, pdists) == 1.0
@test_throws AssertionError weighting_fn(10.0, 2, pdists)

dist2 = GammaParticleDistribution(20.0, 5.0, 10.0)
pdists = [dist1, dist2]
@test weighting_fn(10.0, 1, pdists) == 0.02866906313141952
@test weighting_fn(10.0, 2, pdists) == 1.0
@test abs(weighting_fn(rtol, 1, pdists) - 1.0) <= rtol

# q_integrands
kernel = LinearKernelFunction(1.0)
dist3 = GammaParticleDistribution(2.0, 6.0, 50.0)
pdists = [dist1, dist2, dist3]
x = 50.0
y = 20.0
for j in 1:3
    for k in 1:3
        if j == k
            @test_throws AssertionError q_integrand_inner(x, y, j, k, kernel, pdists)
        else
            @test q_integrand_inner(x, y, j, k, kernel, pdists) > 0.0
            @test_throws AssertionError q_integrand_inner(y, x, j, k, kernel, pdists)
            for moment_order in 0:2
                @test q_integrand_outer(x, j, k, kernel, pdists, FT(moment_order)) > 0.0
                @test q_integrand_outer(y, j, k, kernel, pdists, FT(moment_order)) > 0.0
            end
        end
    end
end

# r_integrands
for moment_order in 0:2
    for j in 1:3
        for k in 1:3
            @test r_integrand_outer(x, j, k, kernel, pdists, FT(moment_order)) > 0.0
            @test r_integrand_outer(y, j, k, kernel, pdists, FT(moment_order)) > 0.0
            @test r_integrand_inner(x, y, j, k, kernel, pdists) > 0.0
            @test r_integrand_inner(y, x, j, k, kernel, pdists) > 0.0
            @test r_integrand_inner(x, y, k, j, kernel, pdists) > 0.0
        end
    end
end

# s_integrands
for k in 1:3
    for moment_order in 0:2
        @test s_integrand_inner(x, k, kernel, pdists, FT(moment_order)) > 0.0
        @test s_integrand1(x, k, kernel, pdists, FT(moment_order)) >= 0.0
        @test s_integrand2(x, k, kernel, pdists, FT(moment_order)) >= 0.0
        @test isapprox(s_integrand1(x, k, kernel, pdists, FT(moment_order)) + s_integrand2(x, k, kernel, pdists, FT(moment_order)),
            s_integrand_inner(x, k, kernel, pdists, FT(moment_order)), rtol=1e-6)
    end
end

(Q, R, S, coal_ints) = initialize_coalescence_data(3, 3)

moment_order = 1

update_Q_coalescence_matrix!(moment_order, kernel, pdists, Q)
@test maximum(Q[end,:]) == 0.0
@test minimum(Q[1,2:end]) > 0.0
update_R_coalescence_matrix!(moment_order, kernel, pdists, R)
@test minimum(R) > 0.0

update_S_coalescence_matrix!(moment_order, kernel, pdists, S)
@test S[end,2] == 0.0
@test maximum(S) > 0.0

moment_order = 0
get_coalescence_integral_moment_qrs!(moment_order, kernel, pdists, Q, R, S)
@test maximum(Q[end,:]) == 0.0
@test minimum(Q[1,2:end]) > 0.0
@test minimum(R) > 0.0
@test S[end,2] == 0.0
@test maximum(S) > 0.0

coal_data = initialize_coalescence_data(3, 3)
update_coal_ints!(3, kernel, pdists, coal_data)
@test coal_data.coal_ints[1,1] < 0.0
@test sum(coal_data.coal_ints[:,1]) < 0.0
@test isapprox(sum(coal_data.coal_ints[:,2]), 0.0; atol=1e-2)
@test sum(coal_data.coal_ints[:,3]) > 0.0

dist1b = ExponentialParticleDistribution(10.0, 100.0)
pdists = [dist1b, dist2, dist3]
@test_throws ArgumentError update_coal_ints!(3, kernel, pdists, coal_data)