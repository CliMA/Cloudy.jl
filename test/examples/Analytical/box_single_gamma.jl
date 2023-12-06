"Box model with a single gamma mode"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = ArrayPartition([1.0, 2, 6])
dist_init = [GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(2))]

# Solver
kernel_func = (x, y) -> 5e-3 * (x + y)
kernel = CoalescenceTensor(kernel_func, 1, FT(500))
tspan = (FT(0), FT(1000))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, kernel = kernel, coal_data = coal_data, dt = FT(1))
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, (; pdists = dist_init); file_name = "box_single_gamma_params.pdf")
plot_moments!(sol, (; pdists = dist_init); file_name = "box_single_gamma_moments.pdf")
plot_spectra!(sol, (; pdists = dist_init); file_name = "box_single_gamma_spectra.pdf", logxrange = (-2, 10))
