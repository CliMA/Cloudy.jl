"""
  multi-moment bulk microphysics scheme

Microphysics parameterization based on moment approximations:
  - condensation/evaporation as relaxation to equilibrium
  - collision-coalescence
  - collision-breakup
  - sedimentation
"""
module Sources

using Cloudy.Distributions
using Cloudy.KernelTensors

# methods that compute source terms from microphysical parameterizations
export get_int_coalescence
export get_flux_sedimentation


"""
  get_int_coalescence(mom_p::Array{Real}, dist::Distribution{Real}, ker::KernelTensor{Real})

  - `mom_p` - prognostic moments of particle mass distribution
  - `dist` - particle mass distribution used to calculate diagnostic moments
  - `ker` - coalescence kernel tensor
Returns the coalescence integral for all moments in `mom_p`.
"""
function get_int_coalescence(mom_p::Array{FT}, dist::Distribution{FT}, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r
  s = length(mom_p)

  # Need to build diagnostic moments
  dist = update_params_from_moments(dist, mom_p)
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
  get_flux_sedimentation(mom_p::Array{Real}, dist::Distribution{Real}, vel::Array{Real})

  - `mom_p` - prognostic moments of particle mass distribution
  - `dist` - particle mass distribution used to calculate diagnostic moments
  _ `vel` - settling velocity coefficient tensor
Returns the sedimentation flux for all moments in `mom_p`.
"""
function get_flux_sedimentation(mom_p::Array{FT}, dist::Distribution{FT}, vel::Array{FT}) where {FT <: Real}
  r = length(vel)-1
  s = length(mom_p) 
  
  # Need to build diagnostic moments
  dist = update_params_from_moments(dist, mom_p)
  mom_d = Array{FT}(undef, r)
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
