"Box model with an additive dist containing two monodisperse"

using DifferentialEquations

include("./utils/box_model_helpers.jl")
include("./utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = [2.0, 1.1, 1.01, 1.001]
dist_init = [MonodisperseAdditiveParticleDistribution(
    MonodispersePrimitiveParticleDistribution(FT(1), FT(0.1)),
    MonodispersePrimitiveParticleDistribution(FT(1), FT(1))
    )]

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

plot_box_model_results(sol, dist_init; outfile = "box_mono_mixture.pdf")