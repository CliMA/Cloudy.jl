"Box model with two gamma modes"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moment_init = [1e8, 1e-2, 2e-12, 1.0, 3e-9, 12e-18]
dist_init = [
    GammaPrimitiveParticleDistribution(FT(1e8), FT(1e-10), FT(1)),
    GammaPrimitiveParticleDistribution(FT(1), FT(1e-9), FT(3)),
]

# Solver
kernel_func = HydrodynamicKernelFunction(1e2 * π)
kernel = CoalescenceTensor(kernel_func, 4, FT(1e-6))
tspan = (FT(0), FT(240))
NProgMoms = [nparams(dist) for dist in dist_init]
norms = [1e6, 1e-9]
coal_data =
    initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms, norms = norms, dist_thresholds = [4e-9, Inf])
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(10))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; yscale = :identity, file_name = "box_gamma_mix_hydro_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "box_gamma_mix_hydro_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_gamma_mix_hydro_spectra.pdf", logxrange = (-12, -6))
print_box_results!(sol, ODE_parameters)
