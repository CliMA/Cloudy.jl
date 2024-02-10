"Test case with a single gamma distribution"


using DifferentialEquations
using Cloudy.KernelFunctions

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Dynamical Info
T_end = 1000
coalescence_coeff = 5e-3
dt = FT(50)

# Initial condition 
Ndist = 1
particle_number = [1.0]
mass_scale = [1.0]
gamma_shape = [2.0]

# Initialize ODE info
pdists = map(1:Ndist) do i
    GammaPrimitiveParticleDistribution(particle_number[i], mass_scale[i], gamma_shape[i])
end
dist_moments = vcat([get_moments(dist) for dist in pdists]...)

# Set up ODE information
tspan = (0.0, T_end)
kernel = LinearKernelFunction(coalescence_coeff)
NProgMoms = [nparams(dist) for dist in pdists]
coal_data = initialize_coalescence_data(NumericalCoalStyle(), kernel, NProgMoms)
rhs = make_box_model_rhs(NumericalCoalStyle())
ODE_parameters = (pdists = pdists, coal_data = coal_data, dt = dt)
prob = ODEProblem(rhs, dist_moments, tspan, ODE_parameters; progress = true)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
plot_params!(sol, ODE_parameters; file_name = "single_particle_gam_params.png")
plot_moments!(sol, ODE_parameters; file_name = "single_particle_gam_moments.png")
plot_spectra!(sol, ODE_parameters; file_name = "single_particle_gam_spectra.png")
