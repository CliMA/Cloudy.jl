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
using ..EquationTypes

# methods that compute source terms from microphysical parameterizations
export get_int_coalescence
export get_flux_sedimentation


"""
get_int_coalescence(::OneModeCoalStyle, mom_p::Array{Real}, ODE_parameters::Dict, ker::KernelTensor{Real})

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
function get_int_coalescence(::OneModeCoalStyle, mom_p::Array{FT}, ODE_parameters::Dict, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r
  s = length(mom_p)

  # Need to build diagnostic moments
  dist_prev = ODE_parameters[:dist][1]
  dist = moments_to_params(dist_prev, mom_p)
  # Update the distribution that is carried along in the ODE_parameters for use
  # in next time step
  ODE_parameters[:dist][1] = dist
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
get_int_coalescence(::TwoModesCoalStyle, mom_p::Array{Real}, ODE_parameters::Dict, ker::KernelTensor{Real})

  - `mom_p` - prognostic moments of the two particle mass distributions
  - `ODE_parameters` - ODE parameters, a Dict containing a key ":dist" which 
                       contains the two distributions at the previous time step. dists is an 
                       array of ParticleDistribution; it is used to calculate the 
                       diagnostic moments and also to dispatch to the
                       moments-to-parameters mapping (done by the function
                       moments_to_params) for the given types of distribution
  - `ker` - coalescence kernel tensor
Returns the coalescence integral for all moments in `mom_p`.
"""
function get_int_coalescence(::TwoModesCoalStyle, mom_p::Array{FT}, ODE_parameters::Dict, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r

  dist_prev = ODE_parameters[:dist]
  n_params = [nparams(dist_prev[i]) for i in 1:2]
  mom_p_ = [mom_p[1:n_params[1]], mom_p[n_params[1]+1:end]]
  s = [length(mom_p_[i]) for i in 1:2]

  # Need to build diagnostic moments
  dist = [moments_to_params(dist_prev[i], mom_p_[i]) for i in 1:2]
  # Update the distributions that is carried along in the ODE_parameters for use
  # in next time step
  ODE_parameters[:dist] = dist
  n_mom = maximum(s) + r
  mom = zeros(FT, 2, n_mom)
  for i in 1:2
    for j in 1:n_mom
      if j <= s[i]
        mom[i, j] = mom_p_[i][j]
      else
        mom[i, j] = moment(dist[i], FT(j-1))
      end
    end
  end

  # Compute the integral term with threshold (mode1-mode1 coalescence contributing to both mode 1 and 2)
  int_w_thrsh = zeros(FT, n_mom, n_mom)
  mom_times_mom = zeros(FT, n_mom, n_mom)
  for i in 1:n_mom
    for j in i:n_mom
      mom_times_mom[i, j] = mom[1, i] * mom[1, j]
      tmp = (mom_times_mom[i, j] < eps(FT)) ? FT(0) : moment_source_helper(dist[1], FT(i-1), FT(j-1), ODE_parameters[:x_th])
      int_w_thrsh[i, j] = min(mom_times_mom[i, j], tmp)
      mom_times_mom[j, i] = mom_times_mom[i, j]
      int_w_thrsh[j, i] = int_w_thrsh[i, j]
    end
  end

  # only calculate coalescence integral for prognostic moments
  coal_int = [similar(mom_p_[1]), similar(mom_p_[2])]
  for i in 1:2
    j = (i==1) ? 2 : 1
    for k in 0:s[i]-1
      temp = 0.0

      for a in 0:r
        for b in 0:r
          coef = ker.c[a+1, b+1]
          temp -= coef * mom[i, a+k+1] * mom[i, b+1]
          temp -= coef * mom[i, a+k+1] * mom[j, b+1]
          for c in 0:k
            coef_binomial = coef * binomial(k, c)
            if i == 1
              temp += 0.5 * coef_binomial * int_w_thrsh[a+c+1, b+k-c+1]
            elseif i == 2
              temp += 0.5 * coef_binomial * (mom_times_mom[a+c+1, b+k-c+1] - int_w_thrsh[a+c+1, b+k-c+1])
              temp += 0.5 * coef_binomial * mom[i, a+c+1] * mom[i, b+k-c+1]
              temp += coef_binomial * mom[j, a+c+1] * mom[i, b+k-c+1]
            end
          end
        end
      end

    coal_int[i][k+1] = temp
    end
  end

  return vcat(coal_int[1], coal_int[2])
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
