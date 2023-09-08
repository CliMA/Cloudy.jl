module PlottingUtils

using Plots
using Cloudy.ParticleDistributions

export plot_moments!, plot_spectra!

function plot_moments!(sol, p; plt_title="test_moments")
    Plots.gr()
    time = sol.t
    moment_plots = []
    if p.Ndist == 1
        for i in 1:p.Nmom
            ploti = plot(time,
                    vcat(sol.u...)[:,i],
                    linewidth=2,
                    xaxis="time [s]",
                    yaxis="M"*string(i-1),
                    label="M_{"*string(i-1)*",1}",
                    xlims=(0, maximum(time)),
                    ylims=(0, 1.2*maximum(vcat(sol.u...)[:,i]))
                )
            push!(moment_plots, 
                ploti
            )
        end
    else
        plotu = reshape(vcat(sol.u...), (p.Ndist,length(time),p.Nmom))
        for i in 1:p.Nmom
            ploti = plot(time,
                    plotu[1,:,i],
                    linewidth=2,
                    xaxis="time [s]",
                    yaxis="M"*string(i-1),
                    label="M_{"*string(i-1)*",1}",
                    xlims=(0, maximum(time)),
                    ylims=(0, 1.2*maximum(plotu[1,:,i]))
                )
            for j in 2:p.Ndist
                plot!(ploti,
                    time,
                    plotu[j,:,i],
                    linewidth=2,
                    label="M_{"*string(i-1)*","*string(j)*"}",)
            end
            plot!(ploti,
                time,
                vcat(sum(plotu[:,:,i], dims=1)...),
                linestyle=:dash,
                linecolor=:black,
                label="M_"*string(i-1),
                linewidth=2
            )
            push!(moment_plots, 
            ploti
        )
        end
    end
    plot(moment_plots...)
    savefig("examples/"*plt_title*".png")
end

function plot_spectra!(sol, p; plt_title="test_spectra", logxrange=(0, 8))
    Plots.gr()
    x = 10 .^ (collect(range(logxrange[1], logxrange[2], 100)))
    r = (x * 3 / 4 / π) .^ (1/3)

    # initial distribution
    if p.Ndist > 1
        for i in 1:p.Ndist
            update_dist_from_moments!(p.pdists[i], reshape(vcat(sol.u...), (p.Ndist,length(sol.t),p.Nmom))[i,1,:])
        end
    else
        update_dist_from_moments!(p.pdists[1], vcat(sol.u...)[1,:])
    end
    pinit = plot(r,
        3*x.*p.pdists[1].(x),
        linewidth=2,
        xaxis=:log,
        yaxis="dV / d(ln r)",
        xlabel="r",
        xlim=(minimum(r), maximum(r)),
        label="Initial, pdist 1"
    )
    if p.Ndist > 1
        prinit = 3*x.*p.pdists[1].(x)
        for i=2:p.Ndist
            plot!(pinit, r,
                3*x.*p.pdists[i].(x),
                linewidth=2,
                label="Initial, pdist "*string(i)
            )
            prinit = prinit .+ 3*x.*p.pdists[i].(x)
        end
    end
    plot!(pinit, r,
        prinit,
        linewidth=2,
        linestyle=:dash,
        linecolor=:black,
        label="Initial"
    )

    # final distribution
    if p.Ndist > 1
        for i in 1:p.Ndist
            update_dist_from_moments!(p.pdists[i], reshape(vcat(sol.u...), (p.Ndist,length(sol.t),p.Nmom))[i,end,:])
        end
    else
        update_dist_from_moments!(p.pdists[1], vcat(sol.u...)[end,:])
    end

    pfin = plot(r,
        3*x.*p.pdists[1].(x),
        linewidth=2,
        xaxis=:log,
        yaxis="dV / d(ln r)",
        xlabel="r",
        xlim=(minimum(r), maximum(r)),
        label="Final, pdist 1"
    )
    if p.Ndist > 1
        prfini = 3*x.*p.pdists[1].(x)
        for i=2:p.Ndist
            plot!(pfin, r,
                3*x.*p.pdists[i].(x),
                linewidth=2,
                label="Final, pdist "*string(i)
            )
            prfini = prfini .+ 3*x.*p.pdists[i].(x)
        end
    end
    plot!(pfin, r,
        prfini,
        linewidth=2,
        linestyle=:dash,
        linecolor=:black,
        label="Final"
    )

    plot(pinit, pfin)
    savefig("examples/"*plt_title*".png")
end

end # module 