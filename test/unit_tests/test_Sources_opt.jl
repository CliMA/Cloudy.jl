using Cloudy.ParticleDistributions
using Cloudy.EquationTypes
using Cloudy.KernelFunctions
using Cloudy.KernelTensors
using Cloudy.Coalescence:
    weighting_fn,
    q_integrand_inner,
    q_integrand_outer,
    r_integrand_inner,
    r_integrand_outer,
    s_integrand1,
    s_integrand2,
    s_integrand_inner,
    update_R_coalescence_matrix!,
    update_S_coalescence_matrix!,
    update_Q_coalescence_matrix!,
    initialize_coalescence_data,
    get_coalescence_integral_moment_qrs!,
    update_coal_ints!,
    update_moments!,
    update_finite_2d_integrals!
using Cloudy.Sedimentation
using Cloudy.Condensation
using JET: @test_opt
using QuadGK

rtol = 1e-4

dist1a = GammaPrimitiveParticleDistribution(10.0, 100.0, 3.0)
dist1b = ExponentialPrimitiveParticleDistribution(10.0, 100.0)
dist2a = GammaPrimitiveParticleDistribution(1.0, 10.0, 5.0)
dist2b = ExponentialPrimitiveParticleDistribution(10.0, 1000.0)

# Analytical Coal
kernel = CoalescenceTensor((x, y) -> x + y, 1, 1e-6)
NProgMoms = [3, 3, 3]
@test_opt initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms)
@test_opt initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms, norms = [10.0, 0.1])
moment_order = 0

for pdists in ([dist1a], [dist1a, dist2a])
    NProgMoms = [nparams(dist) for dist in pdists]
    cd = initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms)

    @test_opt update_moments!(pdists, cd.moments)
    @test_opt update_finite_2d_integrals!(pdists, cd.dist_thresholds, cd.moments, cd.finite_2d_ints)
    @test_opt update_Q_coalescence_matrix!(
        AnalyticalCoalStyle(),
        moment_order,
        cd.moments,
        cd.matrix_of_kernels,
        NProgMoms,
        cd.Q,
    )
    @test_opt update_R_coalescence_matrix!(
        AnalyticalCoalStyle(),
        moment_order,
        cd.moments,
        cd.matrix_of_kernels,
        NProgMoms,
        cd.R,
    )
    @test_opt update_S_coalescence_matrix!(
        AnalyticalCoalStyle(),
        moment_order,
        cd.moments,
        cd.finite_2d_ints,
        cd.matrix_of_kernels,
        NProgMoms,
        cd.S,
    )
    @test_opt get_coalescence_integral_moment_qrs!(AnalyticalCoalStyle(), moment_order, NProgMoms, cd)
    @test_opt update_coal_ints!(AnalyticalCoalStyle(), pdists, cd)
end

for pdists in ([dist1b], [dist1b, dist2b])
    NProgMoms = [nparams(dist) for dist in pdists]
    cd = initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms)

    @test_opt update_moments!(pdists, cd.moments)
    @test_opt update_finite_2d_integrals!(pdists, cd.dist_thresholds, cd.moments, cd.finite_2d_ints)
    @test_opt update_Q_coalescence_matrix!(
        AnalyticalCoalStyle(),
        moment_order,
        cd.moments,
        cd.matrix_of_kernels,
        NProgMoms,
        cd.Q,
    )
    @test_opt update_R_coalescence_matrix!(
        AnalyticalCoalStyle(),
        moment_order,
        cd.moments,
        cd.matrix_of_kernels,
        NProgMoms,
        cd.R,
    )
    @test_opt update_S_coalescence_matrix!(
        AnalyticalCoalStyle(),
        moment_order,
        cd.moments,
        cd.finite_2d_ints,
        cd.matrix_of_kernels,
        NProgMoms,
        cd.S,
    )
    @test_opt get_coalescence_integral_moment_qrs!(AnalyticalCoalStyle(), moment_order, NProgMoms, cd)
    @test_opt update_coal_ints!(AnalyticalCoalStyle(), pdists, cd)
end

# Numerical Coal
kernel = LinearKernelFunction(1.0)
x = 50.0
y = 20.0
j = 1
k = 2
for pdists in ([dist1a, dist2a], [dist1b, dist2b])
    # weighting function
    @test_opt weighting_fn(10.0, 1, pdists)
    @test_opt weighting_fn(8.0, 2, pdists)

    # q_integrands
    @test_opt q_integrand_inner(x, y, j, k, kernel, pdists)
    @test_opt q_integrand_outer(x, j, k, kernel, pdists, 0.0)
    @test_opt q_integrand_outer(x, j, k, kernel, pdists, 1.0)
    @test_opt q_integrand_outer(x, j, k, kernel, pdists, 1.5)

    # r_integrands
    @test_opt r_integrand_inner(x, y, j, k, kernel, pdists)
    @test_opt r_integrand_outer(x, j, k, kernel, pdists, 0.0)
    @test_opt r_integrand_outer(x, j, k, kernel, pdists, 1.0)

    # s_integrands
    @test_opt s_integrand_inner(x, k, kernel, pdists, 1.0)
    @test_opt s_integrand1(x, k, kernel, pdists, 1.0)
    @test_opt s_integrand2(x, k, kernel, pdists, 1.0)
end

# overall Q R S fill matrices 
# n = 1
kernel = LinearKernelFunction(1.0)
NProgMoms = [3, 3, 3]
@test_opt initialize_coalescence_data(NumericalCoalStyle(), kernel, NProgMoms)
@test_opt initialize_coalescence_data(NumericalCoalStyle(), kernel, NProgMoms, norms = [10.0, 0.1])
moment_order = 0.0

for pdists in ([dist1a], [dist1a, dist2a], [dist1b], [dist1b, dist2b])
    NProgMoms = [nparams(dist) for dist in pdists]
    cd = initialize_coalescence_data(NumericalCoalStyle(), kernel, NProgMoms)

    @test_opt update_Q_coalescence_matrix!(NumericalCoalStyle(), moment_order, pdists, cd.kernel_func, cd.Q)
    @test_opt update_R_coalescence_matrix!(NumericalCoalStyle(), moment_order, pdists, cd.kernel_func, cd.R)
    @test_opt update_S_coalescence_matrix!(NumericalCoalStyle(), moment_order, pdists, cd.kernel_func, cd.S)
    @test_opt get_coalescence_integral_moment_qrs!(NumericalCoalStyle(), moment_order, pdists, cd)
    @test_opt update_coal_ints!(NumericalCoalStyle(), pdists, cd)
end

## Sedimentation.jl
# Sedimentation moment flux tests
par = (; pdists = [ExponentialPrimitiveParticleDistribution(1.0, 1.0)], vel = [(1.0, 0.0), (-1.0, 1.0 / 6)])
@test_opt get_sedimentation_flux(par.pdists, par.vel)

## Condensation.jl
# Condensation moment tests
par = (; pdists = [ExponentialPrimitiveParticleDistribution(1.0, 1.0)], ξ = 1e-6)
@test_opt get_cond_evap(0.01, par)
