"Test case with N exponential distributions"


using DifferentialEquations
using Cloudy.KernelFunctions

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Dynamical info
T_end = 1.0
coalescence_coeff = 1e-3
dt = FT(0.01)

# Initial condition 
Ndist = 2
N0 = 100.0
m0 = 100.0
particle_number = [1e-10 for k in 1:Ndist]
particle_number[1] = N0
mass_scale = m0 * [1000.0^(k - 1) for k in 1:Ndist]

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
ODE_parameters = (pdists = pdists, coal_data = coal_data, dt = dt)
prob = ODEProblem(rhs, dist_moments, tspan, ODE_parameters; progress = true)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
plot_params!(sol, ODE_parameters; file_name = "n_particle_exp_params.png")
plot_moments!(sol, ODE_parameters; file_name = "n_particle_exp_moments.png")
plot_spectra!(sol, ODE_parameters; file_name = "n_particle_exp_spectra.png")
