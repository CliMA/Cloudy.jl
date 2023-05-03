"Linear coalescence kernel example"

using Random: seed!
using QuadGK

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.MultiParticleSources

seed!(123)

# Numerical parameters
FT = Float64
tol = 1e-4
n_samples = 300

# Physicsal parameters
# Time has been rescaled below by a factor of 1e1 so that
# 1 sec = 10 deciseconds
time_scale = 1e1
mass_scale = 0.33e-9

T_end = 120 * time_scale
coalescence_coeff = 2.0e3 / time_scale # 1.5e3 cm^3 g^-1 s-1
kernel_func = LinearKernelFunction(coalescence_coeff)

# Parameter transform used to transform native distribution
# parameters to moments and back
tracked_moments = [0, 1, 2]
moments_from_params = dist_params -> [dist_params[1], dist_params[1]*dist_params[2]*dist_params[3], dist_params[1]*dist_params[2]*(dist_params[2]+1)*dist_params[3]^2]
params_from_moments = dist_moments -> [dist_moments[1], (dist_moments[2]/dist_moments[1])/(dist_moments[3]/dist_moments[2]-dist_moments[2]/dist_moments[1]), dist_moments[3]/dist_moments[2]-dist_moments[2]/dist_moments[1]]

# Initial condition
particle_number = [1e4, 1e1]
mean_particles_mass = [mass_scale, 100 * mass_scale]
particle_mass_std = [mass_scale, 50 * mass_scale]
params_init = reduce(vcat, transpose.([particle_number, (mean_particles_mass ./ particle_mass_std).^2, particle_mass_std.^2 ./ mean_particles_mass]))
dist_moments_init = similar(params_init)
for i in 1:length(params_init[1,:])
    dist_moments_init[:,i] = moments_from_params(params_init[:,i])
end

dist_moments = dist_moments_init
kernel = kernel_func


# Transform dist_moments to native distribution parameters
dist_params = similar(dist_moments)
for i in 1:length(dist_moments[1,:])
    dist_params[:,i] = params_from_moments(dist_moments[:,i])
end

# Evaluate processes at inducing points using a closure distribution
pdists = map(1:length(particle_number)) do i
    GammaParticleDistribution(dist_params[1, i], dist_params[2,i], dist_params[3,i])
end
println("Distributions:  ", pdists, "\n")

coal_ints = similar(dist_moments)
println("coal ints initialized")


function weighting_fn(x, pdist1, pdist2)
    denom = pdist1(x) + pdist2(x)
    if denom == 0.0
        return 0.0
    else
        return pdist1(x) / (pdist1(x) + pdist2(x))
    end
    end

    function q_integrand_inner(x, y, j, k, kernel, pdists)
    integrand = 0.5 * kernel(x - y, y) * pdists[j](x-y) * pdists[k](y)
    return integrand
    end

    function q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    outer = x.^moment_order * quadgk(yy -> q_integrand_inner(x, yy, j, k, kernel, pdists), 0.0, x; rtol=1e-4)[1]
    return outer
    end

    function r_integrand(x, y, j, k, kernel, pdists, moment_order)
    integrand = x.^moment_order * kernel(x, y) * pdists[j](x) * pdists[k](y)
    return integrand
    end

    function s_integrand1(x, j, k, kernel, pdists, moment_order)
    integrandj = weighting_fn(x, pdists[j], pdists[k]) * q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    return integrandj
    end

    function s_integrand2(x, j, k, kernel, pdists, moment_order)
    integrandk = (1 - weighting_fn(x, pdists[j], pdists[k])) * q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    return integrandk
    end

j = 1
k = 1
max_mass = ParticleDistributions.moment(pdists[j+1], 1.0)
x = max_mass / 10
y = x / 2
moment_order = 1