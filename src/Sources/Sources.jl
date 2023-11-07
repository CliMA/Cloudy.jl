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
export get_sedimentation_flux


"""
get_int_coalescence(::OneModeCoalStyle, mom_p::Array{Real}, par::Dict, ker::KernelTensor{Real})

  - `mom_p` - prognostic moments of particle mass distribution
  - `par` - ODE parameters, a Dict containing a key ":dist" whose value is a list of ParticleDistribution;
            it is used to calculate the diagnostic moments.
  - `ker` - coalescence kernel tensor
Returns the coalescence integral for all moments in `mom_p`.
"""
function get_int_coalescence(::OneModeCoalStyle, mom_p::Array{FT}, par::Dict, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r
  s = length(mom_p)

  # Need to build diagnostic moments
  update_dist_from_moments!(par[:dist][1], mom_p)
  # Update the distribution that is carried along in par for use in next time steps
  mom_d = Array{FT}(undef, r)
  for k in 0:r-1
    mom_d[k+1] = moment(par[:dist][1], FT(s+k))
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

function get_int_coalescence(::TwoModesCoalStyle, mom_p::Array{FT}, par::Dict, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r

  n_params = [nparams(par[:dist][i]) for i in 1:2]
  mom_p_ = [mom_p[1:n_params[1]], mom_p[n_params[1]+1:end]]
  s = [length(mom_p_[i]) for i in 1:2]

  # update distributions from moments
  for i in 1:2
    update_dist_from_moments!(par[:dist][i], mom_p_[i])
  end
  
  # Need to build diagnostic moments
  n_mom = maximum(s) + r
  mom = zeros(FT, 2, n_mom)
  for i in 1:2
    for j in 1:n_mom
      if j <= s[i]
        mom[i, j] = mom_p_[i][j]
      else
        mom[i, j] = moment(par[:dist][i], FT(j-1))
      end
    end
  end

  # Compute the integral term with threshold (mode1-mode1 coalescence contributing to both mode 1 and 2)
  int_w_thrsh = zeros(FT, n_mom, n_mom)
  mom_times_mom = zeros(FT, n_mom, n_mom)
  for i in 1:n_mom
    for j in i:n_mom
      mom_times_mom[i, j] = mom[1, i] * mom[1, j]
      tmp = (mom_times_mom[i, j] < eps(FT)) ? FT(0) : moment_source_helper(par[:dist][1], FT(i-1), FT(j-1), par[:x_th])
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
  get_sedimentation_flux(mom_p::Array{Real}, par::Dict)

  `mom_p` - prognostic moments
  `par` - ODE parameters, a dict containing a list of ParticleDistributions and terminal celocity coefficients.
Returns sedimentation flux of all prognostic moments, which is the integral of terminal velocity times prognostic moments. The
Terminal velocity of particles is assumed to be expressed as: vel[1] + vel[2] * x^(1/6).
"""
function get_sedimentation_flux(mom_p::Array{FT}, par::Dict) where {FT <: Real}

    vel = par[:vel]
    n_dist = length(par[:dist])
    n_params = [nparams(dist) for dist in par[:dist]]
    mom_p_ = []
    ind = 1
    for i in 1:n_dist
        push!(mom_p_, mom_p[ind:ind-1+n_params[i]])
        ind += n_params[i]
        # update distributions from moments
        update_dist_from_moments!(par[:dist][i], mom_p_[i])
    end

    # Need to build diagnostic moments
    mom_d = [zeros(nd) for nd in n_params]
    for i in 1:n_dist
        for j in 0:n_params[i]-1
            mom_d[i][j+1] = moment(par[:dist][i], FT(j+1.0/6))
        end
    end

    # only calculate sedimentation flux for prognostic moments
    sedi_int = [zeros(ns) for ns in n_params]
    for i in 1:n_dist
        for k in 1:n_params[i]
            sedi_int[i][k] = -vel[1] * mom_p_[i][k] - vel[2] * mom_d[i][k]
        end
    end

    return vcat(sedi_int...)
end

end #module Sources.jl
