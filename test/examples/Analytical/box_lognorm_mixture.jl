"Box model with two lognormal modes"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moment_init = ArrayPartition([10.0, 1.0, 0.2], [0.1, 0.1, 0.2])
dist_init = [
    LognormalPrimitiveParticleDistribution(FT(10), -1.15, 0.55),
    LognormalPrimitiveParticleDistribution(FT(0.1), -0.15, 0.55),
]

# Solver
kernel_func = (x, y) -> 5e-3 * (x + y)
kernel = CoalescenceTensor(kernel_func, 1, FT(500))
tspan = (FT(0), FT(500))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms, dist_thresholds = [FT(0.5), Inf])
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, kernel = kernel, coal_data = coal_data, dt = FT(1))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, (; pdists = dist_init); yscale = :identity, file_name = "box_lognorm_mixture_params.pdf")
plot_moments!(sol, (; pdists = dist_init); file_name = "box_lognorm_mixture_moments.pdf")
plot_spectra!(sol, (; pdists = dist_init); file_name = "box_lognorm_mixture_spectra.pdf", logxrange = (-2, 5))
