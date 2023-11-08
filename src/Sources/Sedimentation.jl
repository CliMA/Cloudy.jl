"""
  multi-moment bulk microphysics implementation of sedimentation

  Includes only a single variations that involves analytical integration
"""
module Sedimentation

using Cloudy.ParticleDistributions

export get_sedimentation_flux


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

end #module Sedimentation.jl
