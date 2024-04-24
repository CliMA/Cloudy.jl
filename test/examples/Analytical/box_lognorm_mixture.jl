"Box model with two lognormal modes"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
# M0: 1/m^3 - M1: kg/m^3 - M2: kg^2/m^3
moment_init = [1e7, 1e-3, 2e-13, 1e5, 1e-4, 2e-13]
dist_init = (
    LognormalPrimitiveParticleDistribution(FT(1e7), -23.37, 0.833), # 1e7/m^3; μ = -23.37; σ = 0.833
    LognormalPrimitiveParticleDistribution(FT(1e5), -21.07, 0.833), # 1e5/m^3; μ = -21.07; σ = 0.833
)

# Solver
kernel_func = (x, y) -> 5 * (x + y) # 5 m^3/kg/s; x, y in kg
kernel = CoalescenceTensor(kernel_func, 1, FT(1e-6))
tspan = (FT(0), FT(120))
NProgMoms = [nparams(dist) for dist in dist_init]
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
coal_data =
    initialize_coalescence_data(AnalyticalCoalStyle(), kernel, NProgMoms, norms = norms, dist_thresholds = [5e-10, Inf])
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(1))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; yscale = :identity, file_name = "box_lognorm_mixture_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "box_lognorm_mixture_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_lognorm_mixture_spectra.pdf", logxrange = (-11, -4))
