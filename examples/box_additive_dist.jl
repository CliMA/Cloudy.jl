"Box model with an additive dist containing two monodisperse"

using DifferentialEquations
using Plots

using Cloudy
using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources

const CPD = Cloudy.ParticleDistributions

FT = Float64

# Physicsal parameters
kernel_func = x -> 5e-3 * (x[1] + x[2])
kernel = CoalescenceTensor(kernel_func, 1, FT(500))

# Initial condition
moments_init = [2.0, 1.1, 1.01, 1.001]
dist_init = MonodisperseAdditiveParticleDistribution(
    MonodispersePrimitiveParticleDistribution(FT(1), FT(0.1)),
    MonodispersePrimitiveParticleDistribution(FT(1), FT(1))
    )

rhs(m, par, t) = get_int_coalescence(m, par, kernel)
tspan = (FT(0), FT(500))
ODE_parameters = Dict(:dist => dist_init, :dt => FT(1))
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])

time = sol.t
moments = vcat(sol.u'...)

plot()
for j in 1:size(moments)[2]
    plot!(time, moments[:, j], linewidth=3, label="M_$(j-1)")
end
p1 = plot!(xaxis="time", yaxis="moments", yscale = :log)

ODE_parameters[:dist] = dist_init
function p2m(p)
    f = CPD.moment_func(dist_init)
    return [f(p..., i) for i in 0:nparams(dist_init)-1]
end
function m2p!(m)
    ODE_parameters[:dist] = CPD.moments_to_params(ODE_parameters[:dist], m)
    return CPD.get_params(ODE_parameters[:dist])[2]
end
params = similar(moments)
for i in 1:size(params)[1]
    par = m2p!(moments[i, :])
    params[i, :] = [par[1]..., par[2]...]
end

plot()
for j in 1:size(params)[2]
    plot!(time, params[:, j], linewidth=3, label="p_$(j)")
end
p2 = plot!(xaxis="time", yaxis="parameters", yscale = :log)


plot(p1, p2, layout = grid(1,2), size = (800, 400))