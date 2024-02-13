"Test case with a single exponential distribution"

using DifferentialEquations
using Cloudy.KernelFunctions

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Dynamical info
T_end = 1.0
coalescence_coeff = 1e-3
dt = FT(0.1)

# Initial condition 
Ndist = 1
particle_number = [100.0]
mass_scale = [100.0]

# Initialize ODE info
pdists = map(1:Ndist) do i
    ExponentialPrimitiveParticleDistribution(particle_number[i], mass_scale[i])
end
dist_moments = vcat([get_moments(dist) for dist in pdists]...)

# Set up ODE
tspan = (0.0, T_end)
kernel = LinearKernelFunction(coalescence_coeff)
NProgMoms = [nparams(dist) for dist in pdists]
coal_data = initialize_coalescence_data(NumericalCoalStyle(), kernel, NProgMoms)
rhs = make_box_model_rhs(NumericalCoalStyle())
ODE_parameters = (pdists = pdists, coal_data = coal_data, NProgMoms = NProgMoms, dt = dt)
prob = ODEProblem(rhs, dist_moments, tspan, ODE_parameters; progress = true)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
@show sol.u
plot_params!(sol, ODE_parameters; file_name = "single_particle_exp_params.png")
plot_moments!(sol, ODE_parameters; file_name = "single_particle_exp_moments.png")
plot_spectra!(sol, ODE_parameters; file_name = "single_particle_exp_spectra.png")
