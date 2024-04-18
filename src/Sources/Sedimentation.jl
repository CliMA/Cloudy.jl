"""
  multi-moment bulk microphysics implementation of sedimentation

  Includes only a single variations that involves analytical integration
"""
module Sedimentation

using Cloudy.ParticleDistributions

export get_sedimentation_flux


"""
  get_sedimentation_flux(pdists, vel)

  `pdists` - list of ParticleDistributions
  `vel` - terminal velocity coefficients
Returns sedimentation flux of all prognostic moments, which is the integral of terminal velocity times prognostic moments. The
terminal velocity of particles is assumed to be expressed as: ∑ vel[i][1] * x^(vel[i][2]) where vel is a vector of tuples.
"""
function get_sedimentation_flux(pdists, vel)

    n_dist = length(pdists)
    n_params = [nparams(dist) for dist in pdists]
    n_vel = length(vel)
    FT = typeof(pdists[1].n)

    # Need to build diagnostic moments
    mom = [zeros(n, n_vel) for n in n_params]
    for i in 1:n_dist
        for j in 1:n_params[i]
            for k in 1:n_vel
                mom[i][j, k] = moment(pdists[i], FT(j - 1 + vel[k][2]))
            end
        end
    end

    # only calculate sedimentation flux for prognostic moments
    sedi_int = [zeros(ns) for ns in n_params]
    for i in 1:n_dist
        for j in 1:n_params[i]
            tmp = 0.0
            for k in 1:n_vel
                tmp -= vel[k][1] * mom[i][j, k]
            end
            sedi_int[i][j] = tmp
        end
    end

    return vcat(sedi_int...)
end

end #module Sedimentation.jl
