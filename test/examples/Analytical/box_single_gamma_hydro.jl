"Box model with a single gamma mode"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = ArrayPartition([100.0, 10.0, 2.0])
dist_init = [GammaPrimitiveParticleDistribution(FT(100), FT(0.1), FT(1))]

# Solver
kernel_func = HydrodynamicKernelFunction(1e-4 * π)
kernel = Array{CoalescenceTensor{FT}}(undef, length(dist_init), length(dist_init))
kernel .= CoalescenceTensor(kernel_func, 4, FT(500))
tspan = (FT(0), FT(240))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), NProgMoms, kernel)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, kernel = kernel, coal_data = coal_data, dist_thresholds = [Inf], dt = FT(20))
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, (; pdists = dist_init); file_name = "box_single_gamma_hydro_params.pdf")
plot_moments!(sol, (; pdists = dist_init); file_name = "box_single_gamma_hydro_moments.pdf")
plot_spectra!(sol, (; pdists = dist_init); file_name = "box_single_gamma_hydro_spectra.pdf", logxrange = (-3, 6))
print_box_results!(sol, (; pdists = dist_init))
