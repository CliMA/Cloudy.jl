"Box model with two gamma modes"

using DifferentialEquations

include("./utils/box_model_helpers.jl")
include("./utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moment_init = [1.0, 0.1, 0.02, 1.0, 1.0, 2.0]
dist_init = [
    GammaPrimitiveParticleDistribution(FT(1), FT(0.1), FT(1)),
    GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))
    ]

# Solver
kernel_func = x -> 5e-3 * (x[1] + x[2])
kernel = CoalescenceTensor(kernel_func, 1, FT(500))
tspan = (FT(0), FT(1000))
rhs = make_box_model_rhs(TwoModesCoalStyle())
ODE_parameters = Dict(
    :dist => dist_init, 
    :kernel => kernel,
    :dt => FT(1), 
    :x_th => FT(0.5)
    )
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])

plot_box_model_results(sol, dist_init; outfile = "box_gamma_mixture.pdf")