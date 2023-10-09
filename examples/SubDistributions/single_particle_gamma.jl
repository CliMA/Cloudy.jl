"Test case with a single gamma distribution"

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using DifferentialEquations

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.MultiParticleSources

include("../utils/plotting_helpers.jl")

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
    T_end = 0.2
    coalescence_coeff = 1e-3
    kernel = LinearKernelFunction(coalescence_coeff)

    # Initial condition 
    Ndist = 1
    particle_number = [100.0]
    mass_scale = [30.0]
    gamma_shape = [3.0]
    Nmom = 3

    # Initialize distributions
    pdists = map(1:Ndist) do i
        GammaPrimitiveParticleDistribution(particle_number[i], mass_scale[i], gamma_shape[i])
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
    plot_moments!(sol, p; file_name="examples/single_particle_gam_moments.png")
    plot_spectra!(sol, p; file_name="examples/single_particle_gam_spectra.png")
end

@time main()