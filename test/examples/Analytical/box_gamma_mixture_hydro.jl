"Box model with two gamma modes"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moment_init = ArrayPartition([100.0, 10.0, 2.0], [1e-6, 3e-5, 12e-4])
dist_init = [
    GammaPrimitiveParticleDistribution(FT(100), FT(0.1), FT(1)),
    GammaPrimitiveParticleDistribution(FT(1e-6), FT(10), FT(3)),
]

# Solver
kernel_func = HydrodynamicKernelFunction(1e-4)
kernel = Array{CoalescenceTensor{FT}}(undef, length(dist_init), length(dist_init))
kernel .= CoalescenceTensor(kernel_func, 4, FT(500))
tspan = (FT(0), FT(200))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), NProgMoms, kernel)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters =
    (; pdists = dist_init, kernel = kernel, coal_data = coal_data, dist_thresholds = [FT(0.25), Inf], dt = FT(1))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, (; pdists = dist_init); file_name = "box_gamma_mix_hydro_params.pdf")
plot_moments!(sol, (; pdists = dist_init); file_name = "box_gamma_mix_hydro_moments.pdf")
plot_spectra!(sol, (; pdists = dist_init); file_name = "box_gamma_mix_hydro_spectra.pdf", logxrange = (-2, 5))
print_box_results!(sol, (; pdists = dist_init))
