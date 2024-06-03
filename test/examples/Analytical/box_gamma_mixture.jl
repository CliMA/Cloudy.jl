"Box model with two gamma modes"

using OrdinaryDiffEq

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")
include(joinpath(pkgdir(Cloudy), "test", "examples", "utils", "netcdf_helpers.jl"))

FT = Float64

# Initial condition
# M0: 1/m^3 - M1: kg/m^3 - M2: kg^2/m^3
moment_init = [1e8, 1e-2, 2e-12, 1, 1e-8, 2e-16]
dist_init = (
    GammaPrimitiveParticleDistribution(FT(1e8), FT(1e-10), FT(1)), # 1e8/m^3; 1e-10 kg; k = 1
    GammaPrimitiveParticleDistribution(FT(1), FT(1e-8), FT(1)), # 1/m^3; 1e-8 kg; k = 1
)

# Solver
kernel_func = LinearKernelFunction(FT(5)) # 5 m^3/kg/s; x, y in kg
kernel = CoalescenceTensor(kernel_func, 1, FT(1e-6))
tspan = (FT(0), FT(120))
NProgMoms = map(dist_init) do dist
    nparams(dist)
end
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
coal_data = CoalescenceData(kernel, NProgMoms, (5e-10, Inf), norms)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(10))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "box_gamma_mixture_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "box_gamma_mixture_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_gamma_mixture_spectra.pdf", logxrange = (-12, -3))
print_box_results!(sol, ODE_parameters)
box_output(sol, ODE_parameters, "box_double_gamma_golovin.nc", FT)
