"Box model with a single gamma mode"

using DifferentialEquations

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
moments_init = ArrayPartition([100.0, 10.0], [10.0, 10.0, 20.0])
dist_init = [
    ExponentialPrimitiveParticleDistribution(FT(100), FT(0.1)), # 100/cm^3; 10^5 µm^3
    GammaPrimitiveParticleDistribution(FT(10), FT(1), FT(1)),   # 10/cm^3; 10^6 µm^3; k=1
    ]

# Solver
s = 0.05
ξ = 1e-2
tspan = (FT(0), FT(120))
NProgMoms = [nparams(dist) for dist in dist_init]
rhs!(dm, m, par, t) = rhs_condensation!(dm, m, par, s)
ODE_parameters = (; ξ=ξ, pdists = dist_init, dt = FT(10))
prob = ODEProblem(rhs!, moments_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, (; pdists = dist_init); file_name = "condensation_expgam_params.pdf")
plot_moments!(sol, (; pdists = dist_init); file_name = "condensation_expgam_moments.pdf")
plot_spectra!(sol, (; pdists = dist_init); file_name = "condensation_expgam_spectra.pdf", logxrange = (-3, 6))
print_box_results!(sol, (; pdists = dist_init))
