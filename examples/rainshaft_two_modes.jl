"1D Rainshaft model with coalescence and sedimentation for two gamma modes"

using LinearAlgebra
using DifferentialEquations
using Plots

using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources

import Cloudy.ParticleDistributions: density, nparams


FT = Float64

function initial_condition(z)
    zmax = findmax(z)[1]
    zs1 = 2 .* (z .- 0.5 .* zmax) ./ zmax .* 500.0
    zs2 = 2 .* (z .- 0.75 .* zmax) ./ zmax .* 500.0
    at1 = 0.5 .* (1 .+ atan.(zs1) .* 2 ./ pi)
    at2 = 0.5 .* (1 .+ atan.(zs2) .* 2 ./ pi)
    at = 1e-6 .+ at1 .- at2
  
    ic = zeros(length(z), 6)
    ic[:, 1] = at
    ic[:, 2] = 0.1 * at
    ic[:, 3] = 0.02 * at
    ic[:, 4] = at
    ic[:, 5] = at
    ic[:, 6] = 2 * at
  
    return ic 
end

function get_sedimentation_flux_two_modes(mom_p, ODE_parameters) 

    vel = ODE_parameters[:vel]
    dist_prev = ODE_parameters[:dist]
    n_params = [nparams(dist_prev[i]) for i in 1:2]
    mom_p_ = [mom_p[1:n_params[1]], mom_p[n_params[1]+1:end]]
    s = [length(mom_p_[i]) for i in 1:2]

    # Need to build diagnostic moments
    dist = [moments_to_params(dist_prev[i], mom_p_[i]) for i in 1:2]

    mom_d = [zeros(s[1]), zeros(s[2])]
    ODE_parameters[:dist] = dist
    for i in 1:2
        for j in 0:s[i]-1
            mom_d[i][j+1] = moment(dist[i], FT(j+1.0/6))
        end
    end

    # only calculate sedimentation flux for prognostic moments
    sedi_int = [zeros(s[1]), zeros(s[2])]
    for i in 1:2
        for k in 1:s[i]
            sedi_int[i][k] = -vel[1] * mom_p_[i][k] - vel[2] * mom_d[i][k]
        end
    end

    return vcat(sedi_int[1], sedi_int[2])
end

function sedi_flux(m, par)
    mom = parent(m)
    flux = similar(mom)
    for i in 1:size(mom)[1]
        flux[i, :] = get_sedimentation_flux_two_modes(mom[i, :], par)
    end
    return flux
end

function rhs(m, par, t)
    nz = size(m)[1]
    nmom = size(m)[2]
    m[findall(x -> x<0, m)] .= 0

    kernel_func = x -> 5e-3 * (x[1] + x[2])
    kernel = CoalescenceTensor(kernel_func, 1, FT(500))
    coal_source = similar(m)
    for i in 1:nz
        if all(m[i, :] .< eps(Float64))
            coal_source[i, :] = zeros(1, nmom)
        else
            coal_source[i, :] = get_int_coalescence_two_modes(m[i, :], par, kernel)
        end
    end

    u = sedi_flux(m, par)
    u_top = zeros(1, nmom)
    u = [u; u_top]
    sedi_source = similar(m)
    for i in 1:nz
        sedi_source[i, :] = -(u[i+1, :] - u[i, :]) / par[:dz]
    end

    return coal_source .+ sedi_source
end

# Build discrete domain
a = 0.0
b = 3000.0
dz = (b-a) / 60
z = a+dz/2:dz:b

# Initial condition
m = initial_condition(z)

# Solver
ODE_parameters = Dict(
    :dist => [GammaPrimitiveParticleDistribution(FT(1), FT(0.1), FT(1)), GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))],
    :vel => [0.0, 2.0],
    :dz => dz, 
    :dt => 1.0,
    :x_th => 0.5,
    :mom_norm => [1 1 1 1 1 1])
tspan = [0, 500.0]
prob = ODEProblem(rhs, m, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])
res = sol.u

p = Array{Plots.Plot}(undef, 6)
xlabel_exts = [" (mode 1)", " (mode 2)"]
for i in 1:6
    xlabel_ext = i < 4 ? xlabel_exts[1] : xlabel_exts[2]
    plot(res[1][:,i], z, lw = 3, xaxis = "M_" * string((i-1)%3) * xlabel_ext, yaxis = "Height[m]")
    plot!(res[floor(Int, end/5)][:,i], z, lw = 3, xaxis = "M_" * string((i-1)%3) * xlabel_ext, yaxis = "Height[m]")
    plot!(res[floor(Int, 2 * end/5)][:,i], z, lw = 3, xaxis = "M_" * string((i-1)%3) * xlabel_ext, yaxis = "Height[m]")
    plot!(res[floor(Int, 3 * end/5)][:,i], z, lw = 3, xaxis = "M_" * string((i-1)%3) * xlabel_ext, yaxis = "Height[m]")
    plot!(res[floor(Int, 4 * end/5)][:,i], z, lw = 3, xaxis = "M_" * string((i-1)%3) * xlabel_ext, yaxis = "Height[m]")
    p[i] = plot!(res[end][:,i], z, lw = 3, xaxis = "M_" * string((i-1)%3) * xlabel_ext, yaxis = "Height[m]")
end

plot(p[1], p[4], p[2], p[5], p[3], p[6], layout = grid(3,2), legend = false, size = (800, 800))