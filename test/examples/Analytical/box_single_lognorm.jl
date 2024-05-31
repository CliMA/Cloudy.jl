"Box model with a single Lognormal mode"

using OrdinaryDiffEq

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
# M0: 1/m^3 - M1: kg/m^3 - M2: kg^2/m^3
moments_init = [1e6, 2e-3, 6e-12]
dist_init = (LognormalPrimitiveParticleDistribution(FT(1e6), -20.233, 0.637),) # 1e6/m^3; μ = -20.233; σ = 0.637

# Solver
kernel_func = LinearKernelFunction(FT(5)) # 5 m^3/kg/s; x, y in kg
kernel = CoalescenceTensor(kernel_func, 1, FT(1e-6))
tspan = (FT(0), FT(1000))
NProgMoms = map(dist_init) do dist
    nparams(dist)
end
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
coal_data = CoalescenceData(kernel, NProgMoms, (Inf,), norms)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(10))
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "box_single_lognorm_params.pdf", yscale = :identity)
plot_moments!(sol, ODE_parameters; file_name = "box_single_lognorm_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_single_lognorm_spectra.pdf", logxrange = (-11, 1))
