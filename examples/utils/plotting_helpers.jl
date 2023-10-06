using Plots

include("./box_model_helpers.jl")
include("./rainshaft_helpers.jl")

"""
  plot_box_model_results(ode_sol, dists; outfile = "box_model.pdf")

  `ode_sol` - ODE solution
  `dists` - array of particle distributions
Plots box model simulation results for arbitrary number and any combination of modes
"""
function plot_box_model_results(ode_sol, dists; outfile = "box_model.pdf")
    time = ode_sol.t
    moments = vcat(ode_sol.u'...)

    n_dist = length(dists)
    p = Array{Plots.Plot}(undef, n_dist * 2)
    n_params = [nparams(dists[i]) for i in 1:n_dist]
    ind = 1
    for i in 1:n_dist
        m_ = moments[:, ind:ind+n_params[i]-1]
        ind += n_params[i]
        params = similar(m_)
        for j in 1:size(params)[1]
            dists[i] = CPD.moments_to_params(dists[i], m_[j, :])
            params[j, :] = vcat(CPD.get_params(dists[i])[2]...)
        end

        plot()
        for j in 1:size(m_)[2]
            plot!(time, m_[:, j], linewidth=3, label="M_"*string(j), yscale = :log10)
        end
        p[2*i-1] = plot!(xaxis="time", yaxis="moments (mode "*string(i)*")")
        
        plot()
        for j in 1:size(params)[2]
            plot!(time, params[:, j], linewidth=3, label="p_"*string(j), yscale = :log10)
        end
        p[2*i] = plot!(xaxis="time", yaxis="parameters (mode "*string(i)*")")

    end
    plot(p..., 
        layout = grid(n_dist, 2),
        size = (800, 270 * n_dist),
        foreground_color_legend = nothing, 
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
    )
    savefig(outfile)
end

"""
  plot_rainshaft_results(z, res, par; outfile = "rainshaft.pdf")

  `z` - array of discrete hieghts
  `res` - results of ODE; an array containing matrices of prognostic moments of arbitrary number of modes at discrete times.
  `ODE_parameters` - a dict containing array of distributions and terminal celocity coefficients
Plots rainshaft simulation results for arbitrary number and any combination of modes
"""
function plot_rainshaft_results(z, res, par; outfile = "rainshaft.pdf")
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
    plot(
        p..., 
        layout = grid(n_dist, nm_max), 
        foreground_color_legend = nothing, 
        size = (400 * nm_max, 270 * n_dist),
        left_margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,
        )
    savefig(outfile)
end