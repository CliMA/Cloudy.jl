using Interpolations
using LinearAlgebra

using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources
using Cloudy.EquationTypes

import Cloudy.ParticleDistributions: density

"""
  initial_condition(z)

  `z` - array of discrete heights
Returns initial profiles of three moments for two different modes. The initial profile is 
chosen to be formed by using atan functions.
"""
function initial_condition(z, mom_amp)
    
    zmax = findmax(z)[1]
    zs1 = 2 .* (z .- 0.5 .* zmax) ./ zmax .* 500.0
    zs2 = 2 .* (z .- 0.75 .* zmax) ./ zmax .* 500.0
    at1 = 0.5 .* (1 .+ atan.(zs1) .* 2 ./ pi)
    at2 = 0.5 .* (1 .+ atan.(zs2) .* 2 ./ pi)
    at = 1e-6 .+ at1 .- at2
  
    nmom = length(mom_amp)
    ic = zeros(length(z), nmom)
    for i in 1:nmom
        ic[:, i] = mom_amp[i] * at
    end
  
    return ic 
end

"""
  get_sedimentation_flux_two_modes(mom_p, ODE_parameters)

  `mom_p` - prognostic moments
  `ODE_parameters` - a dict containing array of distributions and terminal celocity coefficients
Returns sedimentation flux of all prognostic moments, which is the integral of terminal velocity times prognostic moments. The
Terminal velocity is assumed to be a power series.
"""
function sedimentation_flux(mom_p, ODE_parameters) 

    vel = ODE_parameters[:vel]
    dist_prev = ODE_parameters[:dist]
    n_dist = length(dist_prev)
    n_params = [nparams(dist) for dist in dist_prev]
    mom_p_ = []
    ind = 1
    for i in 1:n_dist
        push!(mom_p_, mom_p[ind:ind-1+n_params[i]])
        ind += n_params[i]
    end

    # Need to build diagnostic moments
    dist = [moments_to_params(dist_prev[i], mom_p_[i]) for i in 1:n_dist]
    ODE_parameters[:dist] = dist
    mom_d = [zeros(nd) for nd in n_params]
    for i in 1:n_dist
        for j in 0:n_params[i]-1
            mom_d[i][j+1] = moment(dist[i], FT(j+1.0/6))
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

"""
  make_rainshaft_rhs(coal_type::CoalescenceStyle)

  `coal_type` type of coal source term function: OneModeCoalStyle or TwoModesCoalStyle
Returns a function representing the right hand side of the ODE equation containing divergence 
of sedimentation flux and coalescence source term.
"""
function make_rainshaft_rhs(coal_type::CoalescenceStyle)

    function rhs(m, par, t)
        nz = size(m)[1]
        nmom = size(m)[2]
        m[findall(x -> x<0, m)] .= 0

        coal_source = similar(m)
        for i in 1:nz
            if all(m[i, :] .< eps(Float64))
                coal_source[i, :] = zeros(1, nmom)
            else
                coal_source[i, :] = get_int_coalescence(coal_type, m[i, :], par, par[:kernel])
            end
        end

        u = similar(m)
        for i in 1:nz
            u[i, :] = sedimentation_flux(m[i, :], par)
        end
        u_top = zeros(1, nmom)
        u = [u; u_top]
        sedi_source = similar(m)
        for i in 1:nz
            sedi_source[i, :] = -(u[i+1, :] - u[i, :]) / par[:dz]
        end

        return coal_source .+ sedi_source
    end
end

"""
  analytical_sol(dist, ic, coeff, z, t)

  `dist` - distribution of particles
  `ic` - initial condition of moments
  `coeff` - coefficnets of power series for individual particle terminal velocity
  `z` - array of discrete heights
  `t` - time
Returns semi-analytiacal profiles of moments at time `t` for particles with the distribution `dist`
in a pure sedimentation process.
"""
function analytical_sol(dist, ic, coeff, z, t)
    nz, nmom = size(ic)
    nm = 10000
    m_ = 10 .^ range(-5, 4, nm)
    ic_f = linear_interpolation((z, 0:nmom-1), ic, extrapolation_bc = Line())

    mom = zeros(nz, nmom)
    for (i, z_) in enumerate(z)
        for j in 2:nm-1
            m = m_[j]
            dm = (m_[j+1] - m_[j-1])/2
            v = coeff[1] + coeff[2] * m^(1/6)
            z0 = z_ + v * t
            if z0 > maximum(z)
                continue
            end
            dist = update_params_from_moments(Dict(:dist => dist), ic_f.(z0, 0:nmom-1))
            for k in 1:nmom
                mom[i, k] += m^(k-1) * density(dist, m) * dm
            end
        end
    end
    return mom
end