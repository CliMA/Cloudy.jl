"Box model with a mixture of a monodisperse and a gamma dist"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
# M0: 1/m^3 - M1: kg/m^3 - M2: kg^2/m^3
moments_init = [1e7, 1e-3, 1e5, 1e-4, 2e-13]
dist_init = (
    MonodispersePrimitiveParticleDistribution(FT(1e7), FT(1e-10)),# 1e7/m^3; 1e-10 kg
    GammaPrimitiveParticleDistribution(FT(1e5), FT(1e-9), FT(1)),# 1e5/m^3; 1e-9 kg; k = 1
)

# Solver
kernel_func = (x, y) -> 5 * (x + y) # 5 m^3/kg/s; x, y in kg
kernel = CoalescenceTensor(kernel_func, 1, FT(1e-6))
tspan = (FT(0), FT(1000))
NProgMoms = map(dist_init) do dist
    nparams(dist)
end
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
coal_data = CoalescenceData(kernel, NProgMoms, (5e-10, Inf), norms)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(1))
prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "box_mono_gamma_mixture_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "box_mono_gamma_mixture_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_mono_gamma_mixture_spectra.pdf", logxrange = (-11, -1))
