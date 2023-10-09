using Plots

include("./box_model_helpers.jl")
include("./rainshaft_helpers.jl")

"""
  plot_box_model_results(ode_sol, dists; outfile = "box_model.pdf")

  `ode_sol` - ODE solution
  `dists` - array of particle distributions
Plots box model simulation results for arbitrary number and any combination of modes
"""
function plot_box_model_results(ode_sol, dists;
    x_lim = [1e-3, 1e4], 
    y_scale = :log10, 
    outfile = "box_model.pdf",
    plot_spectrum = true,
    golovin_analytical_sol = Dict("plot" => false)
    )
    time = ode_sol.t
    moments = vcat(ode_sol.u'...)
    params = similar(moments)

    n_dist = length(dists)
    p = Array{Plots.Plot}(undef, n_dist * 2 + 2)
    p[end-1] = plot()
    p[end] = plot()
    x = 10 .^ (log10(x_lim[1]):0.01:log10(x_lim[2]))
    n_params = [nparams(dists[i]) for i in 1:n_dist]
    ind = 1
    for i in 1:n_dist
        rng = ind:ind+n_params[i]-1
        for j in 1:size(params)[1]
            dists[i] = CPD.moments_to_params(dists[i], moments[j, rng])
            params[j, rng] = vcat(CPD.get_params(dists[i])[2]...)
        end

        plot()
        for j in rng
            plot!(time, moments[:, j], linewidth=3, label="M_"*string(j-1), yscale = y_scale)
        end
        p[2*i-1] = plot!(xaxis="time", yaxis="moments (mode "*string(i)*")")
        
        plot()
        for j in rng
            plot!(time, params[:, j], linewidth=3, label="p_"*string(j-ind+1), yscale = y_scale)
        end
        p[2*i] = plot!(xaxis="time", yaxis="parameters (mode "*string(i)*")")

        f = CPD.density_func(dists[i])
        p[end-1] = plot(p[end-1], x, x.^1 .* f.(params[1, rng]..., x), linewidth=1, c=1, ls = :dashdotdot)
        plot!(x, x.^1 .* f.(params[floor(Int, end/2), rng]..., x), linewidth=1, c=2, ls = :dashdotdot)
        plot!(x, x.^1 .* f.(params[end, rng]..., x), linewidth=1, c=3, xscale = :log10, ls = :dashdotdot)

        p[end] = plot(p[end], x, x.^2 .* f.(params[1, rng]..., x), linewidth=1, c=1, ls = :dashdotdot)
        plot!(x, x.^2 .* f.(params[floor(Int, end/2), rng]..., x), linewidth=1, c=2, ls = :dashdotdot)
        plot!(x, x.^2 .* f.(params[end, rng]..., x), linewidth=1, c=3, xscale = :log10, ls = :dashdotdot)

        ind += n_params[i]
    end
    # plot sum of number density and mass distributions
    num = [p[end-1][1][1][:y], p[end-1][1][2][:y], p[end-1][1][3][:y]]
    mass = [p[end][1][1][:y], p[end][1][2][:y], p[end][1][3][:y]]
    for i in 2:n_dist
        for j in 1:3
            num[j] += p[end-1][1][3*(i-1)+j][:y]
            mass[j] += p[end][1][3*(i-1)+j][:y]
        end
    end
    p[end-1] = plot(p[end-1], x, num[1], linewidth=2, c=1, ls = :solid)
    p[end-1] = plot!(x, num[2], linewidth=2, c=2, ls = :solid)
    p[end-1] = plot!(x, num[3], linewidth=2, c=3, ls = :solid)
    p[end-1] = plot!(xaxis="x [μg]", yaxis="dn/dlogx [1/cm^3]", legend =false)
    p[end] = plot(p[end], x, mass[1], linewidth=2, c=1, ls = :solid)
    p[end] = plot!(x, mass[2], linewidth=2, c=2, ls = :solid)
    p[end] = plot!(x, mass[3], linewidth=2, c=3, ls = :solid)
    p[end] = plot!(xaxis="x [μg]", yaxis="dm/dlogx [μg/cm^3]", legend =false)

    # plot Golovin's analytical solution
    if golovin_analytical_sol["plot"] == true
        dist_num = golovin_analytical_sol["dist_num"]
        FT = eltype(moments)
        @assert typeof(dists[dist_num]) in [ExponentialPrimitiveParticleDistribution{FT}, GammaPrimitiveParticleDistribution{FT}]
        ind = sum(n_params[1:dist_num-1])+1
        n = params[1,ind]
        x0 = params[1,ind+1]
        b = golovin_analytical_sol["b"]
        nt = size(moments)[1] 
        t0, t1, t2 = (0.0, floor(Int, nt/2), nt) .* golovin_analytical_sol["dt"]
        f(x_, t0) = golovin_analytical_solution(x_, x0, t0; b = b, n = n)
        p[end-1] = plot(p[end-1], x, x.^1 .* f.(x, t0), linewidth=1, c=1, ls = :dash)
        p[end-1] = plot!(x, x.^1 .* f.(x, t1), linewidth=1, c=2, ls = :dash)
        p[end-1] = plot!(x, x.^1 .* f.(x, t2), linewidth=1, c=3, ls = :dash)
        p[end] = plot(p[end], x, x.^2 .* f.(x, t0), linewidth=1, c=1, ls = :dash)
        p[end] = plot!(x, x.^2 .* f.(x, t1), linewidth=1, c=2, ls = :dash)
        p[end] = plot!(x, x.^2 .* f.(x, t2), linewidth=1, c=3, ls = :dash)
    end

    p_ = plot(p[1:end-2]..., layout = grid(n_dist, 2), foreground_color_legend = nothing,
        size = (800, 300 * n_dist), 
        left_margin = 5Plots.mm,
        bottom_margin = 7Plots.mm
    )
    if plot_spectrum
        plot(p_, p[end-1:end]..., layout=grid(3,1, heights = [n_dist, 1, 1]./(n_dist+2)), 
            size = (800, 300 * (n_dist+2)), 
            left_margin = 5Plots.mm,
            bottom_margin = 7Plots.mm
            )
    end
    savefig(outfile)
end

"""
  plot_rainshaft_results(z, res, par; outfile = "rainshaft.pdf")

  `z` - array of discrete hieghts
  `res` - results of ODE; an array containing matrices of prognostic moments of arbitrary number of modes at discrete times.
  `ODE_parameters` - a dict containing array of distributions and terminal celocity coefficients
Plots rainshaft simulation results for arbitrary number and any combination of modes
"""
function plot_rainshaft_results(z, res, par; outfile = "rainshaft.pdf", plot_analytical_sedimentation = false)
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
    savefig(outfile)
end