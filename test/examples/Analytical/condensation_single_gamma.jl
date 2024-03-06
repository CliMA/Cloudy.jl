"Box model with a single gamma mode"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = [1e8, 1e-2, 2e-12]
dist_init = [GammaPrimitiveParticleDistribution(FT(1e8), FT(1e-10), FT(1))]

# Solver
s = 0.05
ξ = 1e-2
tspan = (FT(0), FT(120))
NProgMoms = [nparams(dist) for dist in dist_init]
norms = [1e6, 1e-9]
rhs!(dm, m, par, t) = rhs_condensation!(dm, m, par, s)
ODE_parameters = (; ξ = ξ, pdists = dist_init, NProgMoms = NProgMoms, norms = norms, dt = FT(10))
prob = ODEProblem(rhs!, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "condensation_single_gamma_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "condensation_single_gamma_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "condensation_single_gamma_spectra.pdf", logxrange = (-12, -3))
print_box_results!(sol, ODE_parameters)
