"Testing correctness of Sources modules, including Coalescence and Sedimentation"

using Cloudy
using Cloudy.ParticleDistributions
using Cloudy.EquationTypes
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
    get_coalescence_integral_moment_qrs!,
    initialize_coalescence_data,
    update_coal_ints!
using Cloudy.Sedimentation
using Cloudy.Condensation
using Cloudy.KernelTensors
using Cloudy.KernelFunctions
using SpecialFunctions: gamma, gamma_inc

rtol = 1e-3

## EquationTypes.jl
CoalescenceStyle <: AbstractStyle
AnalyticalCoalStyle <: CoalescenceStyle
NumericalCoalStyle <: CoalescenceStyle
AnalyticalCoalStyle() isa NumericalCoalStyle
NumericalCoalStyle() isa NumericalCoalStyle

## Coalescence.jl
## Analytical cases
# Constant kernel test (e.g, Smoluchowski 1916)
function sm1916(n_steps, δt; is_kernel_function = true, is_one_mode = true)
    # Parameters & initial condition
    kernel_func = (x, y) -> 1.0
    ker = (is_kernel_function == true) ? CoalescenceTensor(kernel_func, 0, 100.0) : CoalescenceTensor([1.0])

    # Initial condition
    mom = (1.0, 2.0)
    dist = [ExponentialPrimitiveParticleDistribution(1.0, 1.0)]
    coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), ker, [nparams(dist[1])])

    # Euler steps
    for i in 1:n_steps
        dist[1] = update_dist_from_moments(dist[1], mom)
        update_coal_ints!(AnalyticalCoalStyle(), dist, coal_data)
        dmom = coal_data.coal_ints
        mom = tuple(δt * dmom .+ mom...)
    end

    return mom
end

# Smoluchowski 1916 analytical result for 0th moment
function sm1916_ana(t, a, b)
    1 / (1 / a + b / 2 * t)
end

n_steps = 5
δt = 1e-4
rtol = 1e-3
# Run tests
for i in 0:n_steps
    t = δt * i
    @test all(isapprox.(sm1916(n_steps, δt), (sm1916_ana(t, 1, 1), 2.0); rtol))
    @test all(isapprox.(sm1916(n_steps, δt; is_kernel_function = false), (sm1916_ana(t, 1, 1), 2.0); rtol))
end

# Test Exponential + Gamma
# setup
mom_p = [100.0, 10.0, 2.0, 1.0, 1]
dist = [
    GammaPrimitiveParticleDistribution(FT(100), FT(0.1), FT(1)),
    ExponentialPrimitiveParticleDistribution(FT(1), FT(1)),
]
kernel = CoalescenceTensor((x, y) -> 5e-3 * (x + y), 1, FT(10))
NProgMoms = [nparams(d) for d in dist]
r = kernel.r #maximum([ker.r for ker in kernel])
thresholds = [FT(0.5), Inf]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms, dist_thresholds = thresholds)

# action
update_coal_ints!(AnalyticalCoalStyle(), dist, coal_data)

n_mom = maximum(NProgMoms) + r
mom = zeros(FT, 2, n_mom)
for i in 1:2
    for j in 1:n_mom
        mom[i, j] = moment(dist[i], FT(j - 1))
    end
end

int_w_thrsh = zeros(FT, n_mom, n_mom)
mom_times_mom = zeros(FT, n_mom, n_mom)
for i in 1:n_mom
    for j in i:n_mom
        mom_times_mom[i, j] = mom[1, i] * mom[1, j]
        tmp =
            (mom_times_mom[i, j] < eps(FT)) ? FT(0) : moment_source_helper(dist[1], FT(i - 1), FT(j - 1), thresholds[1])
        int_w_thrsh[i, j] = min(mom_times_mom[i, j], tmp)
        mom_times_mom[j, i] = mom_times_mom[i, j]
        int_w_thrsh[j, i] = int_w_thrsh[i, j]
    end
end

coal_int = similar(mom_p)
for i in 1:2
    j = (i == 1) ? 2 : 1
    for k in 0:(NProgMoms[i] - 1)
        temp = 0.0

        for a in 0:r
            for b in 0:r
                coef = kernel.c[a + 1, b + 1] #kernel[i, j].c[a + 1, b + 1]
                temp -= coef * mom[i, a + k + 1] * mom[i, b + 1]
                temp -= coef * mom[i, a + k + 1] * mom[j, b + 1]
                for c in 0:k
                    coef_binomial = coef * binomial(k, c)
                    if i == 1
                        temp += 0.5 * coef_binomial * int_w_thrsh[a + c + 1, b + k - c + 1]
                    elseif i == 2
                        tmp_s12 =
                            0.5 *
                            coef_binomial *
                            (mom_times_mom[a + c + 1, b + k - c + 1] - int_w_thrsh[a + c + 1, b + k - c + 1])
                        temp += tmp_s12
                        tmp_s21 = 0.5 * coef_binomial * mom[i, a + c + 1] * mom[i, b + k - c + 1]
                        temp += tmp_s21
                        temp += coef_binomial * mom[j, a + c + 1] * mom[i, b + k - c + 1]
                    end
                end
            end
        end

        ind = get_dist_moment_ind(NProgMoms, i, k + 1)
        coal_int[ind] = temp
    end
end

# test
@test coal_data.coal_ints[get_dist_moment_ind(NProgMoms, 1, 1)] ≈ coal_int[get_dist_moment_ind(NProgMoms, 1, 1)] rtol =
    10 * eps(FT)
@test coal_data.coal_ints[get_dist_moment_ind(NProgMoms, 1, 2)] ≈ coal_int[get_dist_moment_ind(NProgMoms, 1, 2)] rtol =
    10 * eps(FT)
@test coal_data.coal_ints[get_dist_moment_ind(NProgMoms, 1, 3)] ≈ coal_int[get_dist_moment_ind(NProgMoms, 1, 3)] rtol =
    10 * eps(FT)
@test coal_data.coal_ints[get_dist_moment_ind(NProgMoms, 2, 1)] ≈ coal_int[get_dist_moment_ind(NProgMoms, 2, 1)] rtol =
    10 * eps(FT)
@test coal_data.coal_ints[get_dist_moment_ind(NProgMoms, 2, 2)] ≈ coal_int[get_dist_moment_ind(NProgMoms, 2, 2)] rtol =
    10 * eps(FT)

# Numerical cases
# weighting function
dist1 = GammaPrimitiveParticleDistribution(10.0, 10.0, 3.0)
pdists = [dist1]
@test weighting_fn(10.0, 1, pdists) == 1.0
@test_throws AssertionError weighting_fn(10.0, 2, pdists)

dist2 = GammaPrimitiveParticleDistribution(20.0, 100.0, 5.0)
pdists = [dist1, dist2]
@test weighting_fn(100.0, 1, pdists) == 0.5969233398831713
@test weighting_fn(100.0, 2, pdists) == 1.0
@test abs(weighting_fn(rtol, 1, pdists) - 1.0) <= rtol

# q_integrands
kernel = LinearKernelFunction(1.0)
dist3 = GammaPrimitiveParticleDistribution(2.0, 500.0, 6.0)
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
        @test isapprox(
            s_integrand1(x, k, kernel, pdists, FT(moment_order)) + s_integrand2(x, k, kernel, pdists, FT(moment_order)),
            s_integrand_inner(x, k, kernel, pdists, FT(moment_order)),
            rtol = 1e-6,
        )
    end
end

NProgMoms = [3, 3, 3]
(Q, R, S, coal_ints, kernel_func) = initialize_coalescence_data(NumericalCoalStyle(), kernel, NProgMoms)

moment_order = 1

update_Q_coalescence_matrix!(NumericalCoalStyle(), moment_order, pdists, kernel_func, Q)
@test maximum(Q[end, :]) == 0.0
@test minimum(Q[1, 2:end]) > 0.0
update_R_coalescence_matrix!(NumericalCoalStyle(), moment_order, pdists, kernel_func, R)
@test minimum(R) > 0.0

update_S_coalescence_matrix!(NumericalCoalStyle(), moment_order, pdists, kernel_func, S)
@test S[end, 2] == 0.0
@test maximum(S) > 0.0

moment_order = 0
coal_data = initialize_coalescence_data(NumericalCoalStyle(), kernel, [3, 3, 3])
get_coalescence_integral_moment_qrs!(NumericalCoalStyle(), moment_order, pdists, coal_data)
@test maximum(Q[end, :]) == 0.0
@test minimum(Q[1, 2:end]) > 0.0
@test minimum(R) > 0.0
@test S[end, 2] == 0.0
@test maximum(S) > 0.0

update_coal_ints!(NumericalCoalStyle(), pdists, coal_data)
@test coal_data.coal_ints[get_dist_moment_ind(NProgMoms, 1, 1)] < 0.0
dM = zeros(Float64, 3)
for i in 1:length(NProgMoms)
    for j in 1:3
        dM[j] += coal_data.coal_ints[get_dist_moment_ind(NProgMoms, i, j)]
    end
end
@test dM[1] < 0.0
@test isapprox(dM[2], 0.0; atol = 1e-2)
@test dM[3] > 0.0

## Sedimentation.jl
# Sedimentation moment flux tests
par = (; pdists = [ExponentialPrimitiveParticleDistribution(1.0, 1.0)], vel = [(1.0, 0.0), (-1.0, 1.0 / 6)])
@test get_sedimentation_flux(par.pdists, par.vel) ≈ [-1.0 + gamma(1.0 + 1.0 / 6), -1.0 + gamma(2.0 + 1.0 / 6)] rtol =
    rtol

## Condensation.jl
# Condensation moment tests
pdists = (ExponentialPrimitiveParticleDistribution(1.0, 1.0),)
ξ = 1e-6
s = 0.01
@test all(get_cond_evap(pdists, s, ξ) .≈ (0.0, 3 * 1e-6 * 0.01 * moment(pdists[1], 1 - 2 / 3)))

pdists = (ExponentialPrimitiveParticleDistribution(1.0, 1.0), GammaPrimitiveParticleDistribution(1.0, 2.0, 3.0))
@test all(
    get_cond_evap(pdists, s, ξ) .≈ (
        0.0,
        3 * 1e-6 * 0.01 * moment(pdists[1], 1 - 2 / 3),
        0.0,
        3 * 1e-6 * 0.01 * moment(pdists[2], 1 - 2 / 3),
        3 * 2 * 1e-6 * 0.01 * moment(pdists[2], 2 - 2 / 3),
    ),
)
