"Linear coalescence kernel example"

using DifferentialEquations
using LinearAlgebra
using Plots
using Random: seed!

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.MultiParticleSources

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

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
    mass_scale = 1.0 #0.33e-9

    T_end = 1 * time_scale
    #coalescence_coeff = 2.0e3 / time_scale # 1.5e3 cm^3 g^-1 s-1
    #kernel_func = LinearKernelFunction(coalescence_coeff)
    kernel_func = ConstantKernelFunction(1e-4 / time_scale)

    # Parameter transform used to transform native distribution
    # parameters to moments and back
    tracked_moments = [0, 1, 2]
    moments_from_params = dist_params -> [dist_params[1], dist_params[1]*dist_params[2]*dist_params[3], dist_params[1]*dist_params[2]*(dist_params[2]+1)*dist_params[3]^2]
    params_from_moments = dist_moments -> [dist_moments[1], (dist_moments[2]/dist_moments[1])/(dist_moments[3]/dist_moments[2]-dist_moments[2]/dist_moments[1]), dist_moments[3]/dist_moments[2]-dist_moments[2]/dist_moments[1]]

    # Initial condition
    particle_number = [1e4, 1e2]
    mean_particles_mass = [mass_scale, 100 * mass_scale]
    particle_mass_std = [mass_scale/4, 50 * mass_scale]
    params_init = reduce(vcat, transpose.([particle_number, (mean_particles_mass ./ particle_mass_std).^2, particle_mass_std.^2 ./ mean_particles_mass]))
    dist_moments_init = similar(params_init)
    for i in 1:length(params_init[1,:])
        dist_moments_init[:,i] = moments_from_params(params_init[:,i])
    end
    @show params_init
    @show dist_moments_init
    params = Vector{FT}()

    # Set up the right hand side of ODE
    function rhs!(ddist_moments, dist_moments, p, t)
        @show t
        @show dist_moments
        # Transform dist_moments to native distribution parameters
        dist_params = similar(dist_moments)
        for i in 1:length(dist_moments[1,:])
            dist_params[:,i] = params_from_moments(dist_moments[:,i])
        end
        append!(params, dist_params)
        @show dist_params

        # Evaluate processes at inducing points using a closure distribution
        pdists = map(1:length(particle_number)) do i
            GammaParticleDistribution(dist_params[1, i], dist_params[2,i], dist_params[3,i])
        end
        
        coal_ints = fill!(similar(dist_moments), 0)
        # Compute the sources and sinks
        for (m, moment_order) in enumerate(tracked_moments)
            (Q, R, S) = try get_coalescence_integral_moment_qrs(moment_order, kernel_func, pdists)
            catch
                error("failure in integrals")
            end
            for j in 1:length(pdists)-1
                coal_ints[m,j] += S[j,1] - R[j,j] - R[j,j+1]
                coal_ints[m,j+1] += S[j,2] + Q[j,j+1] + Q[j+1,j] + Q[j+1,j+1] - R[j+1,j] - R[j+1,j+1]
            end
            ddist_moments[m,:] = coal_ints[m,:]

            if any(isnan.(ddist_moments[m,:]))
                println("nan found in derivative")
                @show moment_order
                @show Q
                @show R
                @show S
                @show ddist_moments
                error("nan ddt")
            end
        end
        if ddist_moments[1, 1] > 0.0
            @show (Q, R, S)
            error("positive dn1")
        end
        @show ddist_moments
    end

    # TODO: callbacks

    ddist_moments = similar(dist_moments_init)
    rhs!(ddist_moments, dist_moments_init, params, 0.0)
    
    # Step 3) Solve the ODE
    tspan = (0.0, T_end)
    prob = ODEProblem(rhs!, dist_moments_init, tspan; progress=true)
    sol = solve(prob, reltol=tol, abstol=tol)
    @show sol.u

    # Step 4) Plot the results
    time = sol.t / time_scale
    @show time
    # Get the native distribution parameters
    # n1 = params_from_moments.(vcat(sol.u'...)[:, :, 1])
    # k1 = params_from_moments.(vcat(sol.u'...)[:, 2])
    # θ1 = params_from_moments.(vcat(sol.u'...)[:, 3])
    # # Calculate moments for plotting
    # moment_0 = n
    # moment_1 = n.*k.*θ
    # moment_2 = n.*k.*(k.+1.0).*θ.^2
    moment_0 = reshape(vcat(sol.u'...)[:, 1], 2, length(time))
    moment_1 = reshape(vcat(sol.u'...)[:, 2], 2, length(time))
    moment_2 = reshape(vcat(sol.u'...)[:, 3], 2, length(time))
    @show moment_0
    @show moment_1
    @show moment_2

    p1 = plot(time,
        moment_0[1,:],
        linewidth=2,
        xaxis="time [s]",
        yaxis="M0 [1/cm^3]",
        xlims=(0, maximum(time)),
        ylims=(0, 1.5*maximum(moment_0)),
        label="M_{0,1}"
    )
    plot!(p1, time,
        moment_0[2,:],
        linewidth=2,
        label="M_{0,2}")
    plot!(p1, time,
        sum(moment_0, dims=1)[:],
        linewidth=3,
        label="M_{0,1+2}")

    p2 = plot(time,
        moment_1[1,:],
        linewidth=2,
        xaxis="time [s]",
        yaxis="M1",
        xlims=(0, maximum(time)),
        ylims=(0, 1.5*maximum(moment_1)),
        label="M_{1,1}"
    )
    plot!(p2, time,
        moment_1[2,:],
        linewidth=2,
        label="M_{1,2}")
    plot!(p2, time,
        sum(moment_1, dims=1)[:],
        linewidth=3,
        label="M_{1,1+2}")

    p3 = plot(time,
        moment_2[1,:],
        linewidth=2,
        xaxis="time [s]",
        yaxis="M2 [1/cm^3]",
        xlims=(0, maximum(time)),
        ylims=(0, 1.5*maximum(moment_2)),
        label="M_{2,1}"
    )
    plot!(p3, time,
        moment_2[2,:],
        linewidth=2,
        label="M_{2,2}")
    plot!(p3, time,
        sum(moment_2, dims=1)[:],
        linewidth=3,
        label="M_{2,1+2}")

    plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
    savefig("test.png")
end

main()