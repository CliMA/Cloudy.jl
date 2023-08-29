"Test case with a single exponential distribution"

using DifferentialEquations

using Cloudy.KernelFunctions
using Cloudy.SuperParticleDistributions
using Cloudy.MultiParticleSources
using Cloudy.PlottingUtils

FT = Float64
tol = 1e-4

function rhs!(ddist_moments, dist_moments, p, t)
    # update the ParticleDistributions
    for i=1:p.Ndist
        update_dist_from_moments!(p.pdists[i], dist_moments[i,:])
    end
    # update the information
    update_coal_ints!(p.Nmom, p.kernel_func, p.pdists, p.coal_data)
    ddist_moments .= p.coal_data.coal_ints
end

function main()
    T_end = 1.0
    coalescence_coeff = 1e-3
    kernel = LinearKernelFunction(coalescence_coeff)

    # Initial condition 
    Ndist = 1
    particle_number = [100.0]
    mass_scale = [100.0]
    Nmom = 2

    # Initialize distributions
    pdists = map(1:Ndist) do i
        ExponentialParticleDistribution(particle_number[i], mass_scale[i])
    end

    dist_moments = zeros(FT, Ndist, Nmom)
    for i in 1:Ndist
        dist_moments[i,:] = get_moments(pdists[i])
    end

    # Set up ODE information
    coal_data = initialize_coalescence_data(Ndist, Nmom)
    p = (Ndist=Ndist, Nmom=Nmom, pdists=pdists, kernel_func=kernel, coal_data=coal_data)

    tspan = (0.0, T_end)
    prob = ODEProblem(rhs!, dist_moments, tspan, p; progress=true)
    sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)
    @show sol.u
    plot_moments!(sol, p; plt_title="single_particle_exp_moments")
    plot_spectra!(sol, p; plt_title="single_particle_exp_spectra")
end

@time main()