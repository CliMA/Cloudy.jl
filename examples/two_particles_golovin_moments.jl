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

function moments_from_params(dist_params)
    return [dist_params[1], dist_params[1]*dist_params[2]*dist_params[3], dist_params[1]*dist_params[2]*(dist_params[2]+1)*dist_params[3]^2]
end

function params_from_moments(dist_moments)
    return [dist_moments[1], (dist_moments[2]/dist_moments[1])/(dist_moments[3]/dist_moments[2]-dist_moments[2]/dist_moments[1]), dist_moments[3]/dist_moments[2]-dist_moments[2]/dist_moments[1]]
end

# Set up the right hand side of ODE
function rhs!(ddist_moments, dist_moments, p, t)
    # @show dist_moments
     # Transform dist_moments to native distribution parameters
     dist_params = similar(dist_moments)
     for i in 1:length(dist_moments[1,:])
         dist_params[:,i] = params_from_moments(dist_moments[:,i])
     end
     @show dist_params
     # Evaluate processes at inducing points using a closure distribution
     pdists = map(1:p.Ndist) do i
         GammaParticleDistribution(dist_params[1, i], dist_params[2,i], dist_params[3,i])
     end
     # Compute the sources and sinks
     for (m, moment_order) in enumerate(p.tracked_moments)
        p.coal_ints[m,:] .= 0.0
         try get_coalescence_integral_moment_qrs!(moment_order, p.kernel_func, pdists, p.Q, p.R, p.S)
         catch
             @show dist_moments
             @show dist_params
             error("failure in integrals")
         end
         for k in 1:length(pdists)
            for j in 1:k-1
                p.coal_ints[m,k] += p.Q[j,k]
            end
            for j in 1:length(pdists)
                p.coal_ints[m,k] -= p.R[j,k]
            end
            p.coal_ints[m,k] += p.S[k,1]
            if k > 1
                p.coal_ints[m,k] += p.S[k-1, 2]
            end
        end
         ddist_moments[m,:] = p.coal_ints[m,:]

         if any(isnan.(ddist_moments[m,:]))
             println("nan found in derivative")
             @show moment_order
             @show p.Q
             @show p.R
             @show p.S
             @show ddist_moments
             error("nan ddt")
         end
     end
     #@show ddist_moments
     if ddist_moments[1, 1] > 0.0
         @show (p.Q, p.R, p.S)
         error("positive dn1")
     end
end

function main()

    # Numerical parameters
    FT = Float64
    tol = 1e-4

    # Physical parameters
    # Time has been rescaled below by a factor of 1e1 so that
    # 1 sec = 10 deciseconds
    time_scale = 1.0
    mass_scale = 0.5

    T_end = 1.0 * time_scale
    coalescence_coeff = 1.0
    #kernel_func = LinearKernelFunction(coalescence_coeff)
    kernel_func = ConstantKernelFunction(coalescence_coeff)

    # Parameter transform used to transform native distribution
    # parameters to moments and back
    tracked_moments = [0, 1, 2]

    # Initial condition
    Ndist = 2
    particle_number = [1, 1e-6]
    particle_k = [2, 2]
    particle_θ = [mass_scale, 10*mass_scale]
    params_init = reduce(vcat, transpose.([particle_number, particle_k, particle_θ]))
    dist_moments_init = similar(params_init)
    for i in 1:length(params_init[1,:])
        dist_moments_init[:,i] = moments_from_params(params_init[:,i])
    end
    @show params_init
    @show dist_moments_init

    Q = zeros(FT, (Ndist, Ndist))
    R = zeros(FT, (Ndist, Ndist))
    S = zeros(FT, (Ndist, 2))
    coal_ints = fill!(similar(dist_moments_init), 0)
    p = (Ndist=Ndist, tracked_moments=tracked_moments, kernel_func=kernel_func, Q=Q, R=R, S=S, coal_ints=coal_ints)

    # TODO: callbacks

    ddist_moments = similar(dist_moments_init)
    rhs!(ddist_moments, dist_moments_init, p, 0.0)
    
    # Step 3) Solve the ODE
    tspan = (0.0, T_end)
    prob = ODEProblem(rhs!, dist_moments_init, tspan, p; progress=true)
    sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)
    # Step 4) Plot the results
    time = sol.t / time_scale
    @show time

    moment_0 = reshape(vcat(sol.u'...)[:, 1], 2, length(time))
    moment_1 = reshape(vcat(sol.u'...)[:, 2], 2, length(time))
    moment_2 = reshape(vcat(sol.u'...)[:, 3], 2, length(time))
    @show moment_0
    @show moment_1
    @show moment_2

    params = zeros((length(time), Ndist, 3))
    for i in 1:length(time)
        for j in 1:Ndist
            params[i, j, :] = params_from_moments([moment_0[j, i], moment_1[j, i], moment_2[j, i]])
        end
    end
    @show params

    # p1 = plot(time,
    #     moment_0[1,:],
    #     linewidth=2,
    #     xaxis="time [s]",
    #     yaxis="M0 [1/cm^3]",
    #     xlims=(0, maximum(time)),
    #     ylims=(0, 1.5*maximum(moment_0)),
    #     label="M_{0,1}"
    # )
    # plot!(p1, time,
    #     moment_0[2,:],
    #     linewidth=2,
    #     label="M_{0,2}")
    # plot!(p1, time,
    #     sum(moment_0, dims=1)[:],
    #     linewidth=3,
    #     label="M_{0,1+2}")

    p2 = plot(time,
        moment_1[1,:],
        linewidth=2,
        xaxis="dimensionless time",
        yaxis="M1 (mass)",
        xlims=(0, maximum(time)),
        ylims=(0, 1.5*maximum(moment_1)),
        lc=:blue,
        label="Subdistribution 1"
    )
    plot!(p2, time,
        moment_1[2,:],
        linewidth=2,
        lc=:orange,
        label="Subdistribution 2")
    plot!(p2, time,
        sum(moment_1, dims=1)[:],
        linewidth=3,
        lc=:green,
        label="Sum")

    # p3 = plot(time,
    #     moment_2[1,:],
    #     linewidth=2,
    #     xaxis="time [s]",
    #     yaxis="M2 [1/cm^3]",
    #     xlims=(0, maximum(time)),
    #     ylims=(0, 1.5*maximum(moment_2)),
    #     label="M_{2,1}"
    # )
    # plot!(p3, time,
    #     moment_2[2,:],
    #     linewidth=2,
    #     label="M_{2,2}")
    # plot!(p3, time,
    #     sum(moment_2, dims=1)[:],
    #     linewidth=3,
    #     label="M_{2,1+2}")

    # plot spectra
    r = 10 .^ (collect(range(-0.5, 0.5, 100)))
    x = 4/3 * r.^ 3
    pdists_init = map(1:p.Ndist) do i
        GammaParticleDistribution(params[1, i, 1], params[1, i, 2], params[1, i, 3])
    end
    pdists_fin = map(1:p.Ndist) do i
        GammaParticleDistribution(params[end, i, 1], params[end, i, 2], params[end, i, 3])
    end
    pnew = plot(r,
        3 * x .* pdists_init[1].(x),
        linewidth=2,
        xscale=:log10,
        xaxis="dimensionless particle size",
        yaxis="marginal mass distribution",
        xlims=(minimum(r), maximum(r)),
        ls=:dash,
        lc=:blue,
        label="")
    plot!(pnew,
        r,
        3 * x .* pdists_init[2].(x),
        linewidth=2,
        ls=:dash,
        lc=:orange,
        label="")
    plot!(pnew,
        r,
        3 * x .* pdists_init[2].(x) .+ 3 * x .* pdists_init[1].(x),
        linewidth=3,
        ls=:dash,
        lc=:green,
        label="Initial")
    plot!(pnew,
        r,
        3 * x .* pdists_fin[1].(x),
        linewidth=2,
        lc=:blue,
        ls=:solid,
        label="")
    plot!(pnew,
        r,
        3 * x .* pdists_fin[2].(x),
        linewidth=2,
        lc=:orange,
        ls=:solid,
        label="")
    plot!(pnew,
        r,
        3 * x .* pdists_fin[2].(x) .+ 3 * x .* pdists_fin[1].(x),
        linewidth=3,
        ls=:solid,
        lc=:green,
        label="Final")
    # p4 = plot(x,
    #     pdists_init[1].(x),
    #     linewidth=2,
    #     xaxis=:log,
    #     yaxis="N [1/cm^3]",
    #     xlims=(minimum(x), maximum(x)),
    #     label="N_{init,1}")
    # plot!(p4, x,
    #     pdists_init[2].(x),
    #     linewidth=2,
    #     label="N_{init,2}")
    # plot!(p4, x,
    #     pdists_init[1].(x) + pdists_init[2].(x),
    #     linewidth=3,
    #     label="N_{init}")

    # p5 = plot(x,
    #     pdists_fin[1].(x).*x,
    #     linewidth=2,
    #     xaxis=:log,
    #     yaxis="N [1/cm^3]",
    #     xlims=(minimum(x), maximum(x)),
    #     label="N_{fin,1}")
    # plot!(p5, x,
    #     pdists_fin[2].(x).*x,
    #     linewidth=2,
    #     label="N_{fin,2}")
    # plot!(p5, x,
    #     pdists_fin[1].(x).*x + pdists_fin[2].(x).*x,
    #     linewidth=3,
    #     label="N_{fin}")

    #plot(p1, p2, p3, p4, p5, layout=(2, 3), size=(1200, 700), margin=5Plots.mm)
    plot(p2, pnew, layout=(1,2), size=(700, 300), margin=5Plots.mm)
    savefig("test2.pdf")
end

#@code_warntype main()
@time main()