"Box model with a single gamma mode"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = [1e8, 1e-2, 2e-12]
dist_init = [GammaPrimitiveParticleDistribution(FT(100), FT(1e-10), FT(1))]

# Solver
kernel_func = HydrodynamicKernelFunction(2e2 * π)
kernel = CoalescenceTensor(kernel_func, 4, FT(1e-6))
tspan = (FT(0), FT(240))
NProgMoms = [nparams(dist) for dist in dist_init]
norms = [1e6, 1e-9]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms, norms = norms)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(10))
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "box_single_gamma_hydro_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "box_single_gamma_hydro_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_single_gamma_hydro_spectra.pdf", logxrange = (-12, -3))
print_box_results!(sol, ODE_parameters)
