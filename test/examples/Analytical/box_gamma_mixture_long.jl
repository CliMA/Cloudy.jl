"Box model with two gamma modes and using Long's kernel"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moment_init = [1e7, 1e-3, 2e-13, 1e5, 1e-4, 2e-13]
dist_init = [
    GammaPrimitiveParticleDistribution(FT(1e7), FT(1e-10), FT(1)),
    GammaPrimitiveParticleDistribution(FT(1e5), FT(1e-9), FT(1)),
]

# Solver
kernel_func = LongKernelFunction(5.236e-10, 9.44e9, 5.78)
matrix_of_kernels = Array{CoalescenceTensor{FT}}(undef, 2, 2)
matrix_of_kernels .= CoalescenceTensor(kernel_func, 1, FT(1e-6), lower_limit = FT(5e-10))
matrix_of_kernels[1, 1] = CoalescenceTensor(kernel_func, 2, FT(5e-10))
tspan = (FT(0), FT(120))
NProgMoms = [nparams(dist) for dist in dist_init]
norms = [1e6, 1e-9]
coal_data = initialize_coalescence_data(
    AnalyticalCoalStyle(),
    matrix_of_kernels,
    NProgMoms,
    norms = norms,
    dist_thresholds = [5e-10, Inf],
)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(1.0))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "box_gamma_mix_long_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "box_gamma_mix_long_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_gamma_mix_long_spectra.pdf", logxrange = (-11, -4))
