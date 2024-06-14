"""
    NetCDF output
"""

using NCDatasets
using Cloudy.ParticleDistributions

include(joinpath(pkgdir(Cloudy), "test", "examples", "utils", "plotting_helpers.jl"))

function box_output(sol, p, filename, FT)
    path = joinpath(pkgdir(Cloudy), "test/outputs/")
    ds = NCDataset(joinpath(path, filename), "c")

    # define the dimensions
    time = sol.t
    Ndist = length(p.pdists)
    defDim(ds, "t", length(time))
    defDim(ds, "dist", Ndist)
    defDim(ds, "order", maximum(p.NProgMoms))
    t = defVar(ds, "time", FT, ("t",))
    t[:] = time

    # moments
    M = defVar(ds, "moments", FT, ("t", "dist", "order"))
    Mtot = defVar(ds, "total_moments", FT, ("t", "order"))
    moments = vcat(sol.u'...)
    for i in 1:Ndist
        for j in 1:p.NProgMoms[i]
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            M[:, i, j] = moments[:, ind]
        end
    end

    Nmom_min = minimum(p.NProgMoms)
    moments_sum = zeros(length(time), Nmom_min)
    for i in 1:Ndist
        for j in 1:Nmom_min
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            moments_sum[:, j] += moments[:, ind]
        end
    end
    Mtot[:, 1:Nmom_min] = moments_sum

    # distribution parameters
    Nmom_max = maximum(p.NProgMoms)
    pp = defVar(ds, "params", FT, ("t", "dist", "order"))
    params = zeros(length(time), Ndist, Nmom_max)
    for i in 1:length(time)
        for j in 1:Ndist
            ind_rng = get_dist_moments_ind_range(p.NProgMoms, j)
            moms = moments[i, ind_rng]
            pdist_tmp = update_dist_from_moments(p.pdists[j], ntuple(length(moms)) do i
                moms[i]
            end)
            params[i, j, 1:p.NProgMoms[j]] = vcat(get_params(pdist_tmp)[2]...)
        end
    end
    pp[:, :, :] = params

    close(ds)
end

function rainshaft_output(z, sol, p, filename, FT)
    path = joinpath(pkgdir(Cloudy), "test/outputs/")
    ds = NCDataset(joinpath(path, filename), "c")

    # define the dimensions
    time = sol.t
    Ndist = length(p.pdists)
    defDim(ds, "t", length(time))
    defDim(ds, "z", length(z))
    defDim(ds, "dist", Ndist)
    defDim(ds, "order", maximum(p.NProgMoms))
    t = defVar(ds, "time", FT, ("t",))
    zz = defVar(ds, "altitude", FT, ("z",))
    t[:] = time
    zz[:] = z

    # moments
    M = defVar(ds, "moments", FT, ("t", "z", "dist", "order"))
    Mtot = defVar(ds, "total_moments", FT, ("t", "z", "order"))
    moments = sol.u
    for i in 1:Ndist
        for j in 1:p.NProgMoms[i]
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            for it in 1:length(time)
                M[it, :, i, j] = moments[it][:, ind]
            end
        end
    end

    Nmom_min = minimum(p.NProgMoms)
    moments_sum = zeros(length(time), length(z), Nmom_min)
    for i in 1:Ndist
        for j in 1:Nmom_min
            ind = get_dist_moment_ind(p.NProgMoms, i, j)
            for it in 1:length(time)
                moments_sum[it, :, j] += moments[it][:, ind]
            end
        end
    end
    Mtot[:, :, 1:Nmom_min] = moments_sum

    Nc = defVar(ds, "Nc", FT, ("t", "z"))
    Nr = defVar(ds, "Nr", FT, ("t", "z"))
    Mc = defVar(ds, "Mc", FT, ("t", "z"))
    Mr = defVar(ds, "Mr", FT, ("t", "z"))
    for it in 1:length(time)
        for iz in 1:length(z)
            pdists_tmp = ntuple(Ndist) do ip
                ind_rng = get_dist_moments_ind_range(p.NProgMoms, ip)
                update_dist_from_moments(p.pdists[ip], ntuple(length(ind_rng)) do im
                    moments[it][iz, ind_rng[im]]
                end)
            end
            (; N_liq, M_liq, N_rai, M_rai) = get_standard_N_q(pdists_tmp, 5.236e-10)
            Nc[it, iz] = N_liq
            Nr[it, iz] = N_rai
            Mc[it, iz] = M_liq
            Mr[it, iz] = M_rai
        end
    end

    close(ds)
end
