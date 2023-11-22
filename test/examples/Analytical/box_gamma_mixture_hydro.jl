"Box model with two gamma modes"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moment_init = ArrayPartition([10.0, 1.0, 0.2], [0.1, 0.1, 0.2])
dist_init = [
    GammaPrimitiveParticleDistribution(FT(10), FT(0.1), FT(1)),
    GammaPrimitiveParticleDistribution(FT(0.1), FT(1), FT(1))
    ]

# Solver
kernel_func = x -> HydrodynamicKernelFunction(1e-15)(x[1], x[2])
kernel = CoalescenceTensor(kernel_func, 5, FT(500))
tspan = (FT(0), FT(200))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), NProgMoms, kernel)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (;
    pdists = dist_init, 
    kernel = kernel,
    coal_data = coal_data,
    dist_thresholds = [FT(0.5), Inf],
    dt = FT(1), 
    )
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
@show sol.u
plot_params!(sol, (;pdists = dist_init); file_name = "box_gamma_mix_hydro_params.pdf")
plot_moments!(sol, (;pdists = dist_init); file_name = "box_gamma_mix_hydro_moments.pdf")
plot_spectra!(sol, (;pdists = dist_init); file_name = "box_gamma_mix_hydro_spectra.pdf", logxrange=(-2, 5))