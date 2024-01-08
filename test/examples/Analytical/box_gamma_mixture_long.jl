"Box model with two gamma modes and using Long's kernel"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moment_init = ArrayPartition([10.0, 1.0, 0.2], [0.1, 0.1, 0.2])
dist_init = [
    GammaPrimitiveParticleDistribution(FT(10), FT(0.1), FT(1)),
    GammaPrimitiveParticleDistribution(FT(0.1), FT(1), FT(1)),
]

# Solver
kernel_func = LongKernelFunction(0.5236, 9.44e-3, 5.78e-3)
matrix_of_kernels = Array{CoalescenceTensor{FT}}(undef, 2, 2)
matrix_of_kernels .= CoalescenceTensor(kernel_func, 1, FT(500), lower_limit = FT(0.5))
matrix_of_kernels[1, 1] = CoalescenceTensor(kernel_func, 2, FT(0.5))
tspan = (FT(0), FT(500))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data =
    initialize_coalescence_data(AnalyticalCoalStyle(), matrix_of_kernels, NProgMoms, dist_thresholds = [FT(0.5), Inf])
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, dt = FT(1.0))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, (; pdists = dist_init); file_name = "box_gamma_mix_long_params.pdf")
plot_moments!(sol, (; pdists = dist_init); file_name = "box_gamma_mix_long_moments.pdf")
plot_spectra!(sol, (; pdists = dist_init); file_name = "box_gamma_mix_long_spectra.pdf", logxrange = (-2, 5))
