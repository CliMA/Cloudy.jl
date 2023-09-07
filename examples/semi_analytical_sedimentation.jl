"Sedimentation of an initial layer of liquid with a single-mode gamma dist"

using LinearAlgebra
using Plots
using Interpolations

using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources

import Cloudy.ParticleDistributions: density, nparams, density_func

function initial_condition(z)
    zmax = findmax(z)[1]
    zs1 = 2 .* (z .- 0.5 .* zmax) ./ zmax .* 500.0
    zs2 = 2 .* (z .- 0.75 .* zmax) ./ zmax .* 500.0
    at1 = 0.5 .* (1 .+ atan.(zs1) .* 2 ./ pi)
    at2 = 0.5 .* (1 .+ atan.(zs2) .* 2 ./ pi)
    at = 1e-6 .+ at1 .- at2
  
    ic = zeros(length(z), 3)
    ic[:, 1] = at
    ic[:, 2] = at
    ic[:, 3] = 2 * at
  
    return ic 
end

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

FT = Float64
# Build discrete domain
a = 0.0
b = 3000.0
dz = (b-a) / 60
z = a+dz/2:dz:b

coeff = [0.0, 2.0]
t = 1000.0
dist = GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))
ic = initial_condition(z)
res0 = analytical_sol(dist, ic, coeff, z, 0)
res1 = analytical_sol(dist, ic, coeff, z, t/5)
res2 = analytical_sol(dist, ic, coeff, z, 2*t/5)
res3 = analytical_sol(dist, ic, coeff, z, 3*t/5)
res4 = analytical_sol(dist, ic, coeff, z, 4*t/5)
res5 = analytical_sol(dist, ic, coeff, z, t)

p = Array{Plots.Plot}(undef, 3)
for i in 1:3
    plot(res0[:,i], z, lw = 3, xaxis = "M_" * string(i-1), yaxis = "Height[m]")
    plot!(res1[:,i], z, lw = 3, xaxis = "M_" * string(i-1), yaxis = "Height[m]")
    plot!(res2[:,i], z, lw = 3, xaxis = "M_" * string(i-1), yaxis = "Height[m]")
    plot!(res3[:,i], z, lw = 3, xaxis = "M_" * string(i-1), yaxis = "Height[m]")
    plot!(res4[:,i], z, lw = 3, xaxis = "M_" * string(i-1), yaxis = "Height[m]")
    p[i] = plot!(res5[:,i], z, lw = 3, xaxis = "M_" * string(i-1), yaxis = "Height[m]")
end

plot(p..., layout = grid(2,2), legend = false, size = (800, 500))

