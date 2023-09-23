"Box model with a single gamma mode"

using DifferentialEquations

include("./utils/box_model_helpers.jl")
include("./utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = [1.0, 1, 2]
dist_init = [GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))]

# Solver
tspan = (FT(0), FT(1000))
kernel_func = x -> 5e-3 * (x[1] + x[2])
kernel = CoalescenceTensor(kernel_func, 1, FT(500))
ODE_parameters = Dict(
    :dist => dist_init,
    :kernel => kernel,
    :dt => FT(1)
    )
rhs = make_box_model_rhs(OneModeCoalStyle())
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])

plot_box_model_results(sol, dist_init; outfile = "box_single_gamma.pdf")


