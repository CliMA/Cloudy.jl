"Box model with an additive dist containing two monodisperse"

using DifferentialEquations

include("./utils/box_model_helpers.jl")
include("./utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = [1.0, 0.1, 1.0, 1.0, 2.0]
dist_init = [
    MonodispersePrimitiveParticleDistribution(FT(1), FT(0.1)),
    GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(2))
    ]

# Solver
tspan = (FT(0), FT(1000))
kernel_func = x -> 5e-3 * (x[1] + x[2])
kernel = CoalescenceTensor(kernel_func, 1, FT(500))
ODE_parameters = Dict(
    :dist => dist_init, 
    :kernel => kernel,
    :dt => FT(1),
    :x_th => FT(0.5)
    )
rhs = make_box_model_rhs(TwoModesCoalStyle())
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])

plot_params!(sol, (;pdists = dist_init); file_name = "box_mono_gamma_mixture_params.pdf")
plot_moments!(sol, (;pdists = dist_init); file_name = "box_mono_gamma_mixture_moments.pdf")
plot_spectra!(sol, (;pdists = dist_init); file_name = "box_mono_gamma_mixture_spectra.pdf", logxrange=(-2, 8))