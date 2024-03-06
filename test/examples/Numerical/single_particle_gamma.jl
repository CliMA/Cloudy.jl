"Test case with a single gamma distribution"


using DifferentialEquations
using Cloudy.KernelFunctions

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Dynamical Info
T_end = 1000
coalescence_coeff = 5.0
dt = FT(50)

# Initial condition 
Ndist = 1
particle_number = [1e6]
mass_scale = [1e-9]
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
norms = [1e6, 1e-9]
coal_data = initialize_coalescence_data(NumericalCoalStyle(), kernel, NProgMoms, norms = norms)
rhs = make_box_model_rhs(NumericalCoalStyle())
ODE_parameters = (pdists = pdists, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = dt)
prob = ODEProblem(rhs, dist_moments, tspan, ODE_parameters; progress = true)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
plot_params!(sol, ODE_parameters; file_name = "single_particle_gam_params.png")
plot_moments!(sol, ODE_parameters; file_name = "single_particle_gam_moments.png")
plot_spectra!(sol, ODE_parameters; file_name = "single_particle_gam_spectra.png", logxrange = (-10, -6))
