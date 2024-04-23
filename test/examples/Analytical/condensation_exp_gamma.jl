"Box model with a single gamma mode"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
# M0: 1/m^3 - M1: kg/m^3 - M2: kg^2/m^3
moments_init = [1e8, 1e-2, 1e7, 1e-2, 2e-11]
dist_init = (
    ExponentialPrimitiveParticleDistribution(FT(1e8), FT(1e-10)), # 1e8/m^3; 1e-10 kg
    GammaPrimitiveParticleDistribution(FT(1e7), FT(1e-9), FT(1)), # 1e7/m^3; 1e-9 kg; k = 1
)

# Solver
s = 0.05
ξ = 1e-2
tspan = (FT(0), FT(120))
rhs!(dm, m, par, t) = rhs_condensation!(dm, m, par, s)
NProgMoms = [nparams(dist) for dist in dist_init]
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
ODE_parameters = (; ξ = ξ, pdists = dist_init, NProgMoms = NProgMoms, norms = norms, dt = FT(10))
prob = ODEProblem(rhs!, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "condensation_expgam_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "condensation_expgam_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "condensation_expgam_spectra.pdf", logxrange = (-12, -6))
print_box_results!(sol, ODE_parameters)
