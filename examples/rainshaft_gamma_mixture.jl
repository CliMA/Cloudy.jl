"1D Rainshaft model with coalescence and sedimentation for two gamma modes"

using DifferentialEquations

include("./utils/rainshaft_helpers.jl")
include("./utils/plotting_helpers.jl")


FT = Float64

# Build discrete domain
a = 0.0
b = 3000.0
dz = (b-a) / 60
z = a+dz/2:dz:b

# Initial condition
mom_max = [1, 0.1, 0.02, 1, 1, 2]
nm_tot = length(mom_max)
ic = initial_condition(z, mom_max)
m = ic

# Solver
kernel_func = x -> 5e-3 * (x[1] + x[2])
ODE_parameters = Dict(
    :dist => [GammaPrimitiveParticleDistribution(FT(1), FT(0.1), FT(1)), GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))],
    :kernel => CoalescenceTensor(kernel_func, 1, FT(500)),
    :vel => [0.0, 2.0],
    :dz => dz, 
    :dt => 1.0,
    :x_th => 0.5)
tspan = [0, 1000.0]
rhs = make_rainshaft_rhs(TwoModesCoalStyle())
prob = ODEProblem(rhs, m, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])
res = sol.u

plot_rainshaft_results(z, res, ODE_parameters, file_name = "rainshaft_gamma_mixture.pdf")
plot!()