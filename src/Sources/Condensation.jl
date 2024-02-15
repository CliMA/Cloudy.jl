"""
  multi-moment bulk microphysics implementation of condensation/evaporation

  Includes only a single variations that involves analytical integration
"""
module Condensation

using Cloudy
using Cloudy.ParticleDistributions

export get_cond_evap


"""
    get_cond_evap(s::FT, par::NamedTuple)

    's' - supersaturation
    `par` - ODE parameters, a NamedTuple containing a list of ParticleDistributions and the condensation coefficient ξ
Returns the rate of change of all prognostic moments due to condensation and evaporation (without ventilation effects)
based on the equation dg(x) / dt = -3ξs d/dx(x^{1/3} g(x))
"""
function get_cond_evap(s::FT, par::NamedTuple) where {FT <: Real}
    ξ = par.ξ
    n_dist = length(par.pdists)
    NProgMoms = [nparams(dist) for dist in par.pdists]

    # build diagnostic moments
    mom = zeros(FT, sum(NProgMoms))
    for i in 1:n_dist
        for j in 2:NProgMoms[i]
            ind = get_dist_moment_ind(NProgMoms, i, j)
            mom[ind] = moment(par.pdists[i], FT(j - 1 - 2 / 3))
        end
    end

    # calculate condensation/evaporation flux for prognostic moments
    cond_evap_int = zeros(FT, sum(NProgMoms))
    for i in 1:n_dist
        for j in 2:NProgMoms[i]
            ind = get_dist_moment_ind(NProgMoms, i, j)
            cond_evap_int[ind] = 3 * FT(ξ) * s * (j - 1) * mom[ind]
        end
    end

    return cond_evap_int
end

end #module Condensation.jl
