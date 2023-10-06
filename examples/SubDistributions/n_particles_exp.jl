"Test case with N exponential distributions"

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using DifferentialEquations
using Plots

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
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
    #@show (t, dist_moments, ddist_moments)
end

function main()
    T_end = 0.3
    #kernel = LinearKernelFunction(coalescence_coeff)
    coalescence_eff = 1e-12
    kernel = HydrodynamicKernelFunction(coalescence_eff)

    # Initial condition 
    Ndist = 2
    N0 = 100.0
    m0 = 100.0
    Nmom = 2

    particle_number = [0.0 for k in 1:Ndist] #N0 * [100.0^(-k) for k in 1:Ndist] / sum(100.0^(-k) for k in 1:Ndist)
    particle_number[1] = N0
    mass_scale = m0 * [1000.0^(k-1) for k in 1:Ndist]

    # Initialize distributions
    pdists = map(1:Ndist) do i
        ExponentialPrimitiveParticleDistribution(particle_number[i], mass_scale[i])
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
    plot_moments!(sol, p; plt_title="n_particle_exp_moments")
    plot_spectra!(sol, p; plt_title="n_particle_exp_spectra")
end

@time main()