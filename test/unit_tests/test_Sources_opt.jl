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
    get_R_coalescence_matrix,
    get_S_coalescence_matrix,
    get_Q_coalescence_matrix,
    CoalescenceData,
    get_coalescence_integral_moment_qrs,
    get_coal_ints,
    get_moments_matrix,
    get_finite_2d_integrals
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
order = 1
kernel = CoalescenceTensor((x, y) -> x + y, order, 1e-6)
NProgMoms = (3, 3, 3)
@test_opt CoalescenceData(kernel, NProgMoms, (0.1, 1.0, 10.0), (10.0, 0.1))

for pdists in ((dist1a,), (dist1a, dist2a), (dist1b,), (dist1b, dist2b))
    local NProgMoms = map(pdists) do dist
        nparams(dist)
    end
    cd = CoalescenceData(kernel, NProgMoms, map(pdists) do d
        Inf
    end)

    @test_opt get_moments_matrix(pdists, Val(cd.N_mom_max), cd.N_mom_max)
    moments = get_moments_matrix(pdists, Val(cd.N_mom_max), cd.N_mom_max)
    @test_opt get_finite_2d_integrals(pdists, cd.dist_thresholds, moments, cd.N_2d_ints)
    finite_2d_ints = get_finite_2d_integrals(pdists, cd.dist_thresholds, moments, cd.N_2d_ints)

    @test_opt Cloudy.Coalescence.Q_jk(AnalyticalCoalStyle(), 0, 1, 1, moments, cd.kernels[1][1])
    @test_opt Cloudy.Coalescence.R_jk(AnalyticalCoalStyle(), 0, 1, 1, moments, cd.kernels[1][1])
    @test_opt Cloudy.Coalescence.S_1k(AnalyticalCoalStyle(), 0, 1, moments, finite_2d_ints, cd.kernels[1][1])
    @test_opt Cloudy.Coalescence.S_2k(AnalyticalCoalStyle(), 0, 1, moments, finite_2d_ints, cd.kernels[1][1])

    @test_opt get_Q_coalescence_matrix(AnalyticalCoalStyle(), moments, NProgMoms, cd.kernels)
    @test_opt get_R_coalescence_matrix(AnalyticalCoalStyle(), moments, NProgMoms, cd.kernels)
    @test_opt get_S_coalescence_matrix(AnalyticalCoalStyle(), moments, NProgMoms, finite_2d_ints, cd.kernels)
    @test_opt get_coalescence_integral_moment_qrs(AnalyticalCoalStyle(), moments, NProgMoms, finite_2d_ints, cd.kernels)
    @test_opt get_coal_ints(AnalyticalCoalStyle(), pdists, cd)
end

# Numerical Coal
kernel = LinearKernelFunction(1.0)
x = 50.0
y = 20.0
j = 1
k = 2
for pdists in ((dist1a, dist2a), (dist1b, dist2b))
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
# kernel = LinearKernelFunction(1.0)

# for pdists in ((dist1a,), (dist1a, dist2a), (dist1b,), (dist1b, dist2b))

#     @test_opt get_Q_coalescence_matrix(NumericalCoalStyle(), pdists, kernel)
#     @test_opt get_R_coalescence_matrix(NumericalCoalStyle(), pdists, kernel)
#     @test_opt get_S_coalescence_matrix(NumericalCoalStyle(), pdists, kernel)
#     @test_opt get_coalescence_integral_moment_qrs(NumericalCoalStyle(), pdists, kernel)
#     @test_opt get_coal_ints(NumericalCoalStyle(), pdists, kernel)

# end

## Sedimentation.jl
# Sedimentation moment flux tests
pdists = (ExponentialPrimitiveParticleDistribution(1.0, 1.0),)
vel = ((1.0, 0.0), (-1.0, 1.0 / 6))
@test_opt get_sedimentation_flux(pdists, vel)
@test 64 >= @allocated get_sedimentation_flux(pdists, vel)
pdists = (ExponentialPrimitiveParticleDistribution(1.0, 1.0), GammaPrimitiveParticleDistribution(1.0, 2.0, 3.0))
@test_opt get_sedimentation_flux(pdists, vel)
get_sedimentation_flux(pdists, vel)
@test 64 >= @allocated get_sedimentation_flux(pdists, vel)

## Condensation.jl
# Condensation moment tests
pdists = (ExponentialPrimitiveParticleDistribution(1.0, 1.0),)
ξ = 1e-6
s = 0.01
@test_opt get_cond_evap(pdists, s, ξ)
@test 32 >= @allocated get_cond_evap(pdists, s, ξ)
pdists = (
    ExponentialPrimitiveParticleDistribution(1.0, 1.0),
    GammaPrimitiveParticleDistribution(1.0, 2.0, 3.0),
    GammaPrimitiveParticleDistribution(0.1, 10.0, 3.0),
)
@test_opt get_cond_evap(pdists, s, ξ)
@test 80 >= @allocated get_cond_evap(pdists, s, ξ)
