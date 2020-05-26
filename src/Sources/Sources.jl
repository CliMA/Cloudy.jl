"""
  multi-moment bulk microphysics scheme

Microphysics parameterization based on moment approximations:
  - condensation/evaporation as relaxation to equilibrium
  - collision-coalescence
  - collision-breakup
  - sedimentation
"""
module Sources

using ..ParticleDistributions
using ..KernelTensors

# methods that compute source terms from microphysical parameterizations
export get_int_coalescence
export get_flux_sedimentation


"""
get_int_coalescence(mom_p::Array{Real}, ODE_parameters::Dict, ker::KernelTensor{Real})

  - `mom_p` - prognostic moments of particle mass distribution
  - `ODE_parameters` - ODE parameters, a Dict containing a key ":dist" whose 
                       is the distribution at the previous time step. dist is a 
                       ParticleDistribution; it is used to calculate the 
                       diagnostic moments and also to dispatch to the
                       moments-to-parameters mapping (done by the function
                       moments_to_params) for the given type of distribution
  - `ker` - coalescence kernel tensor
Returns the coalescence integral for all moments in `mom_p`.
"""
function get_int_coalescence(mom_p::Array{FT}, ODE_parameters::Dict, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r
  s = length(mom_p)

  # Need to build diagnostic moments
  dist = update_params_from_moments(ODE_parameters, mom_p)
  # Update the distribution that is carried along in the ODE_parameters for use
  # in next time step
  ODE_parameters[:dist] = dist
  mom_d = Array{FT}(undef, r)
  for k in 0:r-1
    mom_d[k+1] = moment(dist, FT(s+k))
  end
  mom = vcat(mom_p, mom_d)

  # only calculate coalescence integral for prognostic moments
  coal_int = similar(mom_p)
  for k in 0:s-1
    if k == 1
      # implies conservation of 1st moment (~mass) under coalescence processes
      temp = 0.0
    end
    if k != 1
      temp = 0.0
      # coalescence integral for kth moment (k = 0 or k > 1)
      # see design document's microphysics for details of derivation
      for a in 0:r
        for b in 0:r
          coef = ker.c[a+1, b+1]
          temp -= coef * mom[a+k+1] * mom[b+1]
          for c in 0:k
            temp += 0.5 * coef * binomial(k, c) * mom[a+c+1] * mom[b+k-c+1]
          end
        end
      end
    end

  coal_int[k+1] = temp
  end

  return coal_int
end


"""
  get_flux_sedimentation(mom_p::Array{Real}, ODE_parameters::Dict, vel::Array{Real})

  - `mom_p` - prognostic moments of particle mass distribution
  - `ODE_parameters` - ODE parameters, a Dict containing a key ":dist" whose 
                       is the distribution at the previous time step. dist is a 
                       ParticleDistribution; it is used to calculate the 
                       diagnostic moments and also to dispatch to the
                       moments-to-parameters mapping (done by the function
                       moments_to_params) for the given type of distribution
  - `vel` - settling velocity coefficient tensor
Returns the sedimentation flux for all moments in `mom_p`.
"""
function get_flux_sedimentation(mom_p::Array{FT}, ODE_parameters::Dict, vel::Array{FT}) where {FT <: Real}
  r = length(vel)-1
  s = length(mom_p)

  # Need to build diagnostic moments
  dist = update_params_from_moments(ODE_parameters, mom_p)
  mom_d = Array{FT}(undef, r)
  # Update the distribution that is carried along in the ODE_parameters for use
  # in next time step
  ODE_parameters[:dist] = dist
  for k in 0:r-1
    mom_d[k+1] = moment(dist, FT(s+k))
  end
  mom = vcat(mom_p, mom_d)

  # only calculate sedimentation flux for prognostic moments
  sedi_int = similar(mom_p)
  for k in 0:s-1
    temp = 0.0
    for i in 0:r
      coef = vel[i+1]
      temp -= coef * mom[i+k+1]
    end
    sedi_int[k+1] = temp
  end

  return sedi_int
end

end #module Sources.jl
