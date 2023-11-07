using Plots

include("./box_model_helpers.jl")
include("./rainshaft_helpers.jl")

"""
  plot_moments(sol, p; file_name = "test_moments.png")

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the moment time series results
"""
function plot_moments!(sol, p; file_name="examples/test_moments.png")
    time = sol.t
    moments = vcat(reshape.(sol.u', 1, size(sol.u[1]')[1] * size(sol.u[1]')[2])...)

    Ndist = length(p.pdists)
    n_params = [nparams(p.pdists[i]) for i in 1:Ndist]
    Nmom_min = minimum(n_params)
    Nmom_max = maximum(n_params)
    moments_sum = zeros(length(time), Nmom_min)

    plt = Array{Plots.Plot}(undef, Nmom_max)
    for i in 1:Nmom_max
        plt[i] = plot()
    end
    
    ind = 1
    for i in 1:Ndist
        for j in 1:n_params[i]
            plt[j] = plot(plt[j], 
                time,
                moments[:,j+ind-1],
                linewidth=2,
                xaxis="time [s]",
                yaxis="M"*string(j-1),
                label="M_{"*string(j-1)*","*string(i)*"}",
                ylims=(-0.1*maximum(moments[:,j+ind-1]), 1.1*maximum(moments[:,j+ind-1]))
            )
            if j <= Nmom_min
                moments_sum[:, j] += moments[:, j+ind-1]
            end
        end
        ind += n_params[i]
    end
    for i in 1:Nmom_min
        plt[i] = plot(plt[i],
            time,
            moments_sum[:,i],
            linestyle=:dash,
            linecolor=:black,
            label="M_"*string(i-1),
            linewidth=2,
            ylims=(-0.1*maximum(moments_sum[:,i]), 1.1*maximum(moments_sum[:,i]))
        )
    end
    Nrow = floor(Int, sqrt(Nmom_max))
    Ncol = ceil(Int, sqrt(Nmom_max))
    if Nrow * Ncol < Nmom_max
        Nrow += 1
    end
    plot(plt..., layout = grid(Nrow, Ncol), 
        size = (Ncol*500, Nrow*350), 
        foreground_color_legend = nothing,
        left_margin = 5Plots.mm, 
        bottom_margin = 5Plots.mm)
    savefig(file_name)
end

"""
  plot_spectra(sol, p; file_name = "test_spectra.png", logxrange=(0, 8))

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the spectra
"""
function plot_spectra!(sol, p; file_name="examples/test_spectra.png", logxrange=(0, 8))
    x = 10 .^ (collect(range(logxrange[1], logxrange[2], 100)))
    r = (x * 3 / 4 / π) .^ (1/3)

    moments = vcat(reshape.(sol.u', 1, size(sol.u[1]')[1] * size(sol.u[1]')[2])...)
    Ndist = length(p.pdists)
    n_params = [nparams(p.pdists[i]) for i in 1:Ndist]

    plt = Array{Plots.Plot}(undef, 3)
    t_ind = [1, floor(Int, length(sol.t)/2), length(sol.t)]
    sp_sum = zeros(length(r), 3)
    
    for i in 1:3
        ind = 1
        plt[i] = plot()
        for j in 1:Ndist
            update_dist_from_moments!(p.pdists[j], moments[t_ind[i], ind:ind+n_params[j]-1])
            plot!(r,
                3*x.^2 .*p.pdists[j].(x),
                linewidth=2,
                xaxis=:log,
                yaxis="dV / d(ln r)",
                xlabel="r",
                label="Pdist "*string(j),
                title = "time = "*string(round(sol.t[t_ind[i]], sigdigits = 4))
            )
            sp_sum[:, i] += 3*x.^2 .*p.pdists[j].(x)
            ind += n_params[j]
        end
        plot!(r,
            sp_sum[:, i],
            linewidth=2,
            linestyle=:dash,
            linecolor=:black,
            label="Sum"
        )
    end
    

    plot(plt..., layout = grid(1,3), 
        size = (1500, 350), 
        foreground_color_legend = nothing,
        left_margin = 7Plots.mm, 
        bottom_margin = 8Plots.mm)
    savefig(file_name)
end

"""
  plot_params!(sol, p; file_name = "examples/box_model.pdf")

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the evolution of particle distribution parameters in time.
"""
function plot_params!(sol, p; yscale = :log10, file_name = "examples/box_model.pdf")
    time = sol.t
    moments = vcat(reshape.(sol.u', 1, size(sol.u[1]')[1] * size(sol.u[1]')[2])...)
    params = similar(moments)

    n_dist = length(p.pdists)
    plt = Array{Plots.Plot}(undef, n_dist)
    n_params = [nparams(p.pdists[i]) for i in 1:n_dist]
    ind = 1
    for i in 1:n_dist
        rng = ind:ind+n_params[i]-1
        for j in 1:size(params)[1]
            CPD.update_dist_from_moments!(p.pdists[i], moments[j, rng])
            params[j, rng] = vcat(CPD.get_params(p.pdists[i])[2]...)
        end
        
        plot()
        for j in rng
            plot!(time, params[:, j], linewidth=2, label="p_"*string(j-ind+1), yscale = yscale)
        end
        plt[i] = plot!(xaxis="time", yaxis="parameters (mode "*string(i)*")")

        ind += n_params[i]
    end
    nrow = floor(Int, sqrt(n_dist))
    ncol = ceil(Int, sqrt(n_dist))
    if nrow * ncol < n_dist
        nrow += 1
    end
    plot(plt..., layout = grid(nrow, ncol), 
        size = (ncol*400, nrow*270), 
        foreground_color_legend = nothing,
        left_margin = 5Plots.mm, 
        bottom_margin = 5Plots.mm
        )
    savefig(file_name)
end

"""
  plot_rainshaft_results(z, res, par; outfile = "rainshaft.pdf")

  `z` - array of discrete hieghts
  `res` - results of ODE; an array containing matrices of prognostic moments of arbitrary number of modes at discrete times.
  `ODE_parameters` - a dict containing array of distributions and terminal celocity coefficients
Plots rainshaft simulation results for arbitrary number and any combination of modes
"""
function plot_rainshaft_results(z, res, par; file_name = "examples/rainshaft.pdf", plot_analytical_sedimentation = false)
    ic = res[1]
    n_dist = length(par[:dist])
    nm = [nparams(dist) for dist in par[:dist]]
    nm_max = maximum(nm)
    n_plots = nm_max * n_dist
    p = Array{Plots.Plot}(undef, n_plots)
    nt = length(res)
    plot_time_inds = [1, floor(Int, nt/5), floor(Int, 2*nt/5), floor(Int, 3*nt/5), floor(Int, 4*nt/5), nt]
    for i in 1:n_dist
        for j in 1:nm_max
            xlabel_ext = " (mode "*string(i)*")"
            p[(i-1)*nm_max+j] = plot(xaxis = "M_" * string(j-1) * xlabel_ext, yaxis = "z(km)")
            if j > nm[i] continue end
            for (k, t_ind) in enumerate(plot_time_inds)
                plot!(res[t_ind][:,(i-1)*nm_max+j], z/1000, lw = 3, c = k, label = false)
            end
        end
    end

    if plot_analytical_sedimentation
        for (k, t_ind) in enumerate(plot_time_inds)
            t = t_ind * par[:dt]
            ind = 1
            for i in 1:n_dist
                sdm_anl = analytical_sol(par[:dist][i], ic[:, ind:ind-1+nm[i]], par[:vel], z, t)
                for j in 1:nm[i]
                    plot!(p[(i-1)*nm_max+j], sdm_anl[:, j], z/1000, lw = 1, ls = :dash, c = k, label = false)
                end
                ind += nm[i]
            end
        end
        plot!(p[1], NaN.*z, z/1000, lw = 3, ls = :solid, c = :black, label = "numerical solution")
        plot!(p[1], NaN.*z, z/1000, lw = 1, ls = :dash, c = :black, label = "analytical sedimentation")
    end
    plot(
        p..., 
        layout = grid(n_dist, nm_max), 
        foreground_color_legend = nothing, 
        size = (400 * nm_max, 270 * n_dist),
        left_margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,
        )
    savefig(file_name)
end