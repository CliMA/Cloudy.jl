"Linear coalescence kernel example"

using DifferentialEquations
using LinearAlgebra
using Plots
using Random: seed!

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.MultiParticleSources

seed!(123)

function main()

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

    # Set up the right hand side of ODE
    function rhs!(ddist_moments, dist_moments, p, t)
        # Transform dist_moments to native distribution parameters
        dist_params = similar(dist_moments)
        for i in 1:length(dist_moments[1,:])
            dist_params[:,i] = params_from_moments(dist_moments[:,i])
        end

        # Evaluate processes at inducing points using a closure distribution
        pdists = Array{GammaParticleDistribution{FT}}(undef, length(particle_number))
        for i in 1:length(particle_number)
            pdists[i] = GammaParticleDistribution(dist_params[1, i], dist_params[2,i], dist_params[3,i])
        end
        println("Distributions:  ", pdists, "\n")
        
        coal_ints = similar(dist_moments)
        println("coal ints initialized")
        # Compute the sources and sinks
        for (m, moment_order) in enumerate(tracked_moments)
            println("let's see if it works")
            (Q, R, S) = get_coalescence_integral_moment_qrs(moment_order, kernel_func, pdists)
            println("integrals for ", moment_order)
            for j in 1:length(pdists)-1
                coal_ints[m,j] += S[j,1] - R[j,j] - R[j,j+1]
                coal_ints[m,j+1] += S[j,2] + Q[j,j+1] + Q[j+1,j] + Q[j+1,j+1] - R[j+1,j] - R[j+1,j+1]
            end
            ddist_moments[m,:] = coal_ints[m,:]
            println("integrals assigned")
        end
        println(ddist_moments)
        flush(stdout)
    end

    ddist_moments = similar(dist_moments_init)
    rhs!(ddist_moments, dist_moments_init, 0.0, 0.0)
    # # Step 3) Solve the ODE
    # tspan = (0.0, T_end)
    # prob = ODEProblem(rhs!, dist_moments_init, tspan)
    # sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

    # # Step 4) Plot the results
    # time = sol.t / time_scale
    # # Get the native distribution parameters
    # n = params_from_moments.(vcat(sol.u'...)[:, 1])
    # k = params_from_moments.(vcat(sol.u'...)[:, 2])
    # θ = params_from_moments.(vcat(sol.u'...)[:, 3])
    # # Calculate moments for plotting
    # moment_0 = n
    # moment_1 = n.*k.*θ
    # moment_2 = n.*k.*(k.+1.0).*θ.^2

    # p1 = plot(time,
    #     moment_0,
    #     linewidth=3,
    #     xaxis="time [s]",
    #     yaxis="M0 [1/cm^3]",
    #     xlims=(0, maximum(time)),
    #     ylims=(0, 1.5*maximum(moment_0)),
    #     label="M0 CLIMA"
    # )
    # plot!(p1, time,
    #     t-> (moment_0[1] * exp(-moment_1[1] * coalescence_coeff * t * time_scale)),
    #     lw=3,
    #     ls=:dash,
    #     label="M0 Exact"
    # )

    # p2 = plot(time,
    #     moment_1,
    #     linewidth=3,
    #     xaxis="time [s]",
    #     yaxis="M1 [milligrams/cm^3]",
    #     ylims=(0, 1.5*maximum(moment_1)),
    #     label="M1 CLIMA"
    # )
    # plot!(p2, time,
    #     t-> moment_1[1],
    #     lw=3,
    #     ls=:dash,
    #     label="M1 Exact"
    # )
    # p3 = plot(time,
    #     moment_2,
    #     linewidth=3,
    #     xaxis="time [s]",
    #     yaxis="M2 [milligrams^2/cm^3]",
    #     ylims=(0, 1.5*maximum(moment_2)),
    #     label="M2 CLIMA"
    # )
    # plot!(p3, time,
    # t-> (moment_2[1] * exp(2 * moment_1[1] * coalescence_coeff * t * time_scale)),
    # lw=3,
    #     ls=:dash,
    #     label="M2 Exact"
    # )
    # plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
    # savefig("golovin_kernel_test.png")
end

main()