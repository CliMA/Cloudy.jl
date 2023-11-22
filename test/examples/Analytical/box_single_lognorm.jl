"Box model with a single Lognormal mode"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = ArrayPartition([1.0, 2.0, 6.0])
dist_init = [LognormalPrimitiveParticleDistribution(FT(1), 0.213, 0.42)]

# Solver
kernel_func = x -> 5e-3 * (x[1] + x[2])
kernel = CoalescenceTensor(kernel_func, 1, FT(500))
tspan = (FT(0), FT(1000))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), NProgMoms, kernel)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (;
    pdists = dist_init, 
    kernel = kernel,
    coal_data = coal_data,
    dist_thresholds = [Inf],
    dt = FT(10), 
    )
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, (;pdists = dist_init); file_name = "box_single_lognorm_params.pdf")
plot_moments!(sol, (;pdists = dist_init); file_name = "box_single_lognorm_moments.pdf")
plot_spectra!(sol, (;pdists = dist_init); file_name = "box_single_lognorm_spectra.pdf", logxrange=(-2, 10))