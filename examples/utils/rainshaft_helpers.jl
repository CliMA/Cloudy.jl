using Interpolations
using LinearAlgebra

using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Coalescence
using Cloudy.Sedimentation
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
            u[i, :] = get_sedimentation_flux(m[i, :], par)
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