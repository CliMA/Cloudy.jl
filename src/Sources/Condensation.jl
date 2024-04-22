"""
  multi-moment bulk microphysics implementation of condensation/evaporation

  Includes only a single variations that involves analytical integration
"""
module Condensation

using ..ParticleDistributions
import ..rflatten

export get_cond_evap

"""
    get_cond_evap(pdists, s::FT, ξ::FT)

    'pdists` - local particle size distributions
    's' - spatially-varying supersaturation
    'ξ' - spatially-varying condensation coefficient (T, P dependent)
Returns the rate of change of all prognostic moments due to condensation and evaporation (without ventilation effects)
based on the equation dg(x) / dt = -3ξs d/dx(x^{1/3} g(x))
"""
function get_cond_evap(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, s::FT, ξ::FT) where {N, FT <: Real}
    # build diagnostic moments & compute rate of change
    cond_evap_int = map(pdists) do pdist
        ntuple(nparams(pdist)) do j
            j < 2 ? FT(0) : 3 * ξ * s * (j - 1) * moment(pdist, FT(j - 1 - 2 / 3))
        end
    end

    return rflatten(cond_evap_int)
end

end #module Condensation.jl
