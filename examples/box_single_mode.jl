"Box model with a single gamma mode"

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
moments_init = [1.0, 1, 2]
dist_init = GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))

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

function p2m(p)
    f = CPD.moment_func(dist_init)
    return [f(p..., i) for i in 0:nparams(dist_init)-1]
end
function m2p(m)
    return CPD.get_params(CPD.moments_to_params(dist_init, m))[2]
end
params = similar(moments)
for i in 1:size(params)[1]
    params[i, :] = m2p(moments[i, :])
end

plot()
for j in 1:size(params)[2]
    plot!(time, params[:, j], linewidth=3, label="p_$(j)")
end
p2 = plot!(xaxis="time", yaxis="parameters", yscale = :log)

plot(p1, p2, layout = grid(1,2), size = (800, 400))


