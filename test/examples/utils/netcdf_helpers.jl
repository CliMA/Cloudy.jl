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
    Mtot[:,1:Nmom_min] = moments_sum

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
