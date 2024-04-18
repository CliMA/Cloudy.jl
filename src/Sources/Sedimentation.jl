"""
  multi-moment bulk microphysics implementation of sedimentation

  Includes only a single variations that involves analytical integration
"""
module Sedimentation

using Cloudy
using Cloudy.ParticleDistributions

export get_sedimentation_flux


"""
  get_sedimentation_flux(pdists, vel)

  `pdists` - list of ParticleDistributions
  `vel` - terminal velocity coefficients
Returns sedimentation flux of all prognostic moments, which is the integral of terminal velocity times prognostic moments. The
terminal velocity of particles is assumed to be expressed as: ∑ vel[i][1] * x^(vel[i][2]) where vel is a vector of tuples.
"""
function get_sedimentation_flux(
    pdists::NTuple{N, PrimitiveParticleDistribution{FT}},
    vel::NTuple{M, Tuple{FT, FT}},
) where {N, M, FT <: Real}

    # build diagnostic moments and compute sedimentation flux for prognostic moments
    sedi_int = map(pdists) do pdist
        ntuple(nparams(pdist)) do j
            sum(ntuple(length(vel)) do k
                -vel[k][1] * moment(pdist, FT(j - 1 + vel[k][2]))
            end)
        end
    end

    return rflatten(sedi_int)
end

end #module Sedimentation.jl
