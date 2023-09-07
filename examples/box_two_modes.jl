"Box model with two modes"

using DifferentialEquations
using Plots

using Cloudy
using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources

const CPD = Cloudy.ParticleDistributions

FT = Float64

function p2M(p, add_dist)
    f = CPD.moment_func(add_dist)
    return [f(p..., i) for i in 0:nparams(add_dist)-1]
end
function m2p(m, dists)
    n_params = [nparams(dists[i]) for i in 1:2]
    m_ = [m[1:n_params[1]], m[n_params[1]+1:end]]
    dists_tmp = [CPD.moments_to_params(dists[i], m_[i]) for i in 1:2]
    params = [CPD.get_params(dists_tmp[i])[2] for i in 1:2]
    return vcat(params[1], params[2])
end

# Physicsal parameters
kernel_func = x -> 5e-3 * (x[1] + x[2])
kernel = CoalescenceTensor(kernel_func, 1, FT(500))

# Initial condition
dists = [
    GammaPrimitiveParticleDistribution(FT(1), FT(0.1), FT(1)),
    GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))
    ]
add_dist = GammaAdditiveParticleDistribution(dists...)
tspan = (FT(0), FT(500))

m_init = [1.0, 0.1, 0.02, 1.0, 1.0, 2.0]
rhs(m, par, t) = get_int_coalescence_two_modes(m, par, kernel)
ODE_parameters = Dict(:dist => dists, :dt => FT(1), :x_th => FT(0.5))
prob = ODEProblem(rhs, m_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])
moments = vcat(sol.u'...)
Moments = similar(moments)
params = similar(moments)
for i in 1:size(params)[1]
    params[i, :] = m2p(moments[i, :], dists)
    Moments[i, :] = p2M(params[i, :], add_dist)
end

time = sol.t
plot()
labels=["M^0_1", "M^1_1", "M^2_1", "M^0_2", "M^1_2", "M^2_2"]
for j in 1:size(moments)[2]
    plot!(time, moments[:, j], linewidth=3, label=labels[j], yscale = :log)
end
p1 = plot!(xaxis="time", yaxis="moments")
plot()
labels=["n_1", "θ_1", "k_1", "n_2", "θ_2", "k_2"]
for j in 1:size(params)[2]
    plot!(time, params[:, j], linewidth=3, label=labels[j], yscale = :log)
end
p2 = plot!(xaxis="time", yaxis="parameters")
plot()
labels=["M^0", "M^1", "M^2", "M^3", "M^4", "M^5"]
for j in 1:size(moments)[2]
    plot!(time, Moments[:, j], linewidth=3, label=labels[j], yscale = :log)
end
p3 = plot!(xaxis="time", yaxis="moments")


plot(p1, p2, p3, layout = grid(1,3), size = (1100, 400))