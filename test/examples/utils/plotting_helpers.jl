using Plots

include("./box_model_helpers.jl")
include("./rainshaft_helpers.jl")


"""
  get_params(dist)

  - `dist` - is a particle mass distribution
Returns the names and values of settable parameters for a dist.
"""
function get_params(dist::CPD.PrimitiveParticleDistribution{FT} where {FT <: Real})
    params = Array{Symbol, 1}(collect(propertynames(dist)))
    values = Array{FT, 1}([getproperty(dist, p) for p in params])
    return params, values
end

"""
  plot_moments(sol, p; file_name = "test_moments.png")

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the moment time series results
"""
function plot_moments!(sol, p; file_name = "test_moments.png")
    time = sol.t
    moments = vcat(sol.u'...)

    Ndist = length(p.pdists)
    n_params = [nparams(p.pdists[i]) for i in 1:Ndist]
    Nmom_min = minimum(n_params)
    Nmom_max = maximum(n_params)
    moments_sum = zeros(length(time), Nmom_min)

    plt = Array{Plots.Plot}(undef, Nmom_max)
    for i in 1:Nmom_max
        plt[i] = plot()
    end

    for i in 1:Ndist
        for j in 1:n_params[i]
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            plt[j] = plot(
                plt[j],
                time,
                moments[:, ind],
                linewidth = 2,
                xaxis = "time [s]",
                yaxis = "M" * string(j - 1),
                label = "M_{" * string(j - 1) * "," * string(i) * "}",
                ylims = (-0.1 * maximum(moments[:, ind]), 1.1 * maximum(moments[:, ind])),
            )
            if j <= Nmom_min
                moments_sum[:, j] += moments[:, ind]
            end
        end
    end
    for i in 1:Nmom_min
        plt[i] = plot(
            plt[i],
            time,
            moments_sum[:, i],
            linestyle = :dash,
            linecolor = :black,
            label = "M_" * string(i - 1),
            linewidth = 2,
            ylims = (-0.1 * maximum(moments_sum[:, i]), 1.1 * maximum(moments_sum[:, i])),
        )
    end
    Nrow = floor(Int, sqrt(Nmom_max))
    Ncol = ceil(Int, sqrt(Nmom_max))
    if Nrow * Ncol < Nmom_max
        Nrow += 1
    end
    plot(
        plt...,
        layout = grid(Nrow, Ncol),
        size = (Ncol * 400, Nrow * 270),
        foreground_color_legend = nothing,
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    path = joinpath(pkgdir(Cloudy), "test/outputs/")
    mkpath(path)
    savefig(path * file_name)
end

"""
  plot_spectra(sol, p; file_name = "test_spectra.png", logxrange=(0, 8))

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the spectra
"""
function plot_spectra!(sol, p; file_name = "test_spectra.png", logxrange = (-15, -3), print = false)
    x = 10 .^ (collect(range(logxrange[1], logxrange[2], 100)))
    r = (x / 1000 * 3 / 4 / π) .^ (1 / 3) * 1e6 # plot in µm

    if print
        @show x
        @show r
    end

    moments = vcat(sol.u'...)
    Ndist = length(p.pdists)
    n_params = [nparams(p.pdists[i]) for i in 1:Ndist]

    plt = Array{Plots.Plot}(undef, 3)
    t_ind = [1, floor(Int, length(sol.t) / 2), length(sol.t)]
    sp_sum = zeros(length(r), 3)

    for i in 1:3
        plt[i] = plot()
        for j in 1:Ndist
            ind_rng = get_dist_moments_ind_range(p.NProgMoms, j)
            moms = moments[t_ind[i], ind_rng]
            pdist_tmp = update_dist_from_moments(p.pdists[j], ntuple(length(moms)) do i
                moms[i]
            end)
            plot!(
                r,
                3 * x .^ 2 .* pdist_tmp.(x),
                linewidth = 2,
                xaxis = :log,
                yaxis = "dm / d(ln r)",
                xlabel = "r (μm)",
                label = i == 1 ? "Pdist " * string(j) : "",
                title = "time = " * string(round(sol.t[t_ind[i]], sigdigits = 4)),
            )
            sp_sum[:, i] += 3 * x .^ 2 .* pdist_tmp.(x)

            if print
                @show 3 * x .^ 2 .* pdist_tmp.(x)
            end
        end
        plot!(r, sp_sum[:, i], linewidth = 2, linestyle = :dash, linecolor = :black, label = i == 1 ? "Sum " : "")
    end

    plot(
        plt...,
        layout = grid(1, 3),
        size = (1200, 270),
        foreground_color_legend = nothing,
        left_margin = 7Plots.mm,
        bottom_margin = 8Plots.mm,
    )

    path = joinpath(pkgdir(Cloudy), "test/outputs/")
    mkpath(path)
    savefig(path * file_name)
end

"""
  plot_params!(sol, p; file_name = "box_model.pdf")

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Plots the evolution of particle distribution parameters in time (for normalized moments).
"""
function plot_params!(sol, p; yscale = :log10, file_name = "box_model.pdf")
    time = sol.t
    mom_norms = get_moments_normalizing_factors(p.NProgMoms, p.norms)
    moments = vcat(sol.u'...) ./ collect(mom_norms)'
    params = similar(moments)
    n_dist = length(p.pdists)
    labels = ["N", "θ", "k"]
    plt = Array{Plots.Plot}(undef, n_dist)
    n_params = [nparams(p.pdists[i]) for i in 1:n_dist]
    for i in 1:n_dist
        ind_rng = get_dist_moments_ind_range(p.NProgMoms, i)
        for j in 1:size(params)[1]
            moms_tmp = moments[j, ind_rng]
            pdist_tmp = CPD.update_dist_from_moments(p.pdists[i], ntuple(length(moms_tmp)) do i
                moms_tmp[i]
            end)
            params[j, ind_rng] = vcat(get_params(pdist_tmp)[2]...)
        end

        plot()
        for j in ind_rng
            plot!(time, params[:, j], linewidth = 2, label = labels[j - ind_rng[1] + 1], yscale = yscale)
        end
        plt[i] = plot!(xaxis = "time [s]", yaxis = "parameters (mode " * string(i) * ")")
    end
    nrow = floor(Int, sqrt(n_dist))
    ncol = ceil(Int, sqrt(n_dist))
    if nrow * ncol < n_dist
        nrow += 1
    end
    plot(
        plt...,
        layout = grid(nrow, ncol),
        size = (ncol * 400, nrow * 270),
        foreground_color_legend = nothing,
        left_margin = 5Plots.mm,
        bottom_margin = 5Plots.mm,
    )

    path = joinpath(pkgdir(Cloudy), "test/outputs/")
    mkpath(path)
    savefig(path * file_name)
end

"""
  print_box_results!(sol, p)

  `sol` - ODE solution
  `p` - additional ODE parameters carried in the solver
Prints the evolution of moments in time, plus the distribution parameters at a few times
"""
function print_box_results!(sol, p)
    time = sol.t
    moments = vcat(sol.u'...)

    Ndist = length(p.pdists)
    Nmom_min = minimum(p.NProgMoms)
    Nmom_max = maximum(p.NProgMoms)
    moments_sum = zeros(length(time), Nmom_min)

    for i in 1:Ndist
        for j in 1:Nmom_min
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            moments_sum[:, j] += moments[:, ind]
            @show moments[:, ind]
        end
    end
    @show time
    for j in 1:Nmom_min
        @show moments_sum[:, j]
    end

    t_ind = [1, ceil(Int, length(sol.t) / 2), length(sol.t)]
    params = zeros(length(t_ind), Ndist, Nmom_max)
    for i in 1:3
        for j in 1:Ndist
            ind_rng = get_dist_moments_ind_range(p.NProgMoms, j)
            moms = moments[t_ind[i], ind_rng]
            pdist_tmp = update_dist_from_moments(p.pdists[j], ntuple(length(moms)) do i
                moms[i]
            end)
            params[i, j, 1:p.NProgMoms[j]] = vcat(get_params(pdist_tmp)[2]...)
        end
    end
    @show t_ind
    @show params[:, 1, :]
    if Ndist > 1
        @show params[:, 2, :]
    end
end

"""
  plot_rainshaft_results(z, res, p; outfile = "rainshaft.pdf")

  `z` - array of discrete hieghts
  `res` - results of ODE; an array containing matrices of prognostic moments of arbitrary number of modes at discrete times.
  `p` - additional ODE parameters carried in the solver
Plots rainshaft simulation results for arbitrary number and any combination of modes
"""
function plot_rainshaft_results(
    z,
    res,
    p;
    file_name = "rainshaft.pdf",
    plot_analytical_sedimentation = false,
    print = false,
)
    ic = res[1]
    n_dist = length(p.pdists)
    nm = [nparams(dist) for dist in p.pdists]
    nm_max = maximum(nm)
    n_plots = nm_max * n_dist
    plt = Array{Plots.Plot}(undef, n_plots)
    nt = length(res)
    plot_time_inds = [1, floor(Int, nt / 4), floor(Int, 2 * nt / 4), nt]
    for i in 1:n_dist
        for j in 1:nm_max
            xlabel_ext = " (mode " * string(i) * ")"
            plt[(i - 1) * nm_max + j] = plot(xaxis = "M_" * string(j - 1) * xlabel_ext, yaxis = "z(km)")
            if j > nm[i]
                continue
            end
            for (k, t_ind) in enumerate(plot_time_inds)
                plot!(res[t_ind][:, (i - 1) * nm_max + j], z / 1000, lw = 3, c = k, label = false)
            end
        end
    end

    if print
        for t_ind in plot_time_inds
            Nc = zeros(length(z))
            Nr = zeros(length(z))
            Mc = zeros(length(z))
            Mr = zeros(length(z))
            for iz in 1:length(z)
                pdists_tmp = ntuple(n_dist) do ip
                    ind_rng = get_dist_moments_ind_range(p.NProgMoms, ip)
                    update_dist_from_moments(p.pdists[ip], ntuple(length(ind_rng)) do im
                        res[t_ind][iz, ind_rng[im]]
                    end)
                end
                (; N_liq, M_liq, N_rai, M_rai) = get_standard_N_q(pdists_tmp, 5.236e-10)
                Nc[iz] = N_liq
                Nr[iz] = N_rai
                Mc[iz] = M_liq
                Mr[iz] = M_rai
            end
            @show t_ind
            @show Nc
            @show Nr
            @show Mc
            @show Mr
        end
    end

    if plot_analytical_sedimentation
        for (k, t_ind) in enumerate(plot_time_inds)
            t = t_ind * p.dt
            for i in 1:n_dist
                ind_rng = get_dist_moments_ind_range(p.NProgMoms, j)
                sdm_anl = analytical_sol(p.pdists[i], ic[:, ind_rng], p.vel, z, t)
                for j in 1:nm[i]
                    plot!(plt[(i - 1) * nm_max + j], sdm_anl[:, j], z / 1000, lw = 1, ls = :dash, c = k, label = false)
                end
            end
        end
        plot!(plt[1], NaN .* z, z / 1000, lw = 3, ls = :solid, c = :black, label = "numerical solution")
        plot!(plt[1], NaN .* z, z / 1000, lw = 1, ls = :dash, c = :black, label = "analytical sedimentation")
    end
    plot(
        plt...,
        layout = grid(n_dist, nm_max),
        foreground_color_legend = nothing,
        size = (400 * nm_max, 270 * n_dist),
        left_margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,
    )

    path = joinpath(pkgdir(Cloudy), "test/outputs/")
    mkpath(path)
    savefig(path * file_name)
end

"""
  plot_rainshaft_results(z, t, res, p; outfile = "rainshaft_contour.pdf")

  `z` - array of discrete hieghts
  `res` - results of ODE; an array containing matrices of prognostic moments of arbitrary number of modes at discrete times.
  `p` - additional ODE parameters carried in the solver
Plots rainshaft simulation results for arbitrary number and any combination of modes
"""
function plot_rainshaft_contours(z, t, res, p; file_name = "rainshaft_contour.pdf")
    ic = res[1]
    n_dist = length(p.pdists)
    nm = [nparams(dist) for dist in p.pdists]
    nm_max = maximum(nm)
    n_plots = nm_max * n_dist
    plt = Array{Plots.Plot}(undef, n_plots)
    nt = length(t)
    u = hcat(res[:][:, :]...)
    @show ((u[:, 1:6:end]))
    #u[:,1:6:end]

    # res[time][z_ind, moment_ind]
    for i in 1:n_dist
        for j in 1:nm_max
            xlabel_ext = " (mode " * string(i) * ")"
            plt[(i - 1) * nm_max + j] = plot(xaxis = "t(s)", yaxis = "z(km)", title = "M" * string(j) * xlabel_ext)
            if j > nm[i]
                continue
            end
            qty = collect(u[:, ((i - 1) * nm_max + j):n_plots:end])
            heatmap!([1, 2], z / 1000, [z / 1000, 2 * z / 1000])#z/1000, t, qty)
        end
    end


    plot(
        plt...,
        layout = grid(n_dist, nm_max),
        foreground_color_legend = nothing,
        size = (400 * nm_max, 270 * n_dist),
        left_margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,
    )

    path = joinpath(pkgdir(Cloudy), "test/outputs/")
    mkpath(path)
    savefig(path * file_name)
end
