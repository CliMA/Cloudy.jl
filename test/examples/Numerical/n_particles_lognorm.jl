"Test case with N lognormal distributions"

using DifferentialEquations
using Cloudy.KernelFunctions

include("../utils/plotting_helpers.jl")
include("../utils/box_model_helpers.jl")

FT = Float64

T_end = 1000.0
coalescence_coeff = 5e-3
dt = FT(50)

# Initial condition 
Ndist = 2
particle_number = [10.0, 0.1]
mass_scale = [0.1, 1.0]
Nmom = 3

# Initialize ODE info
pdists = map(1:Ndist) do i
    LognormalPrimitiveParticleDistribution(particle_number[i], log(mass_scale[i]), log(2.0))
end
dist_moments = zeros(FT, Ndist, Nmom)
for i in 1:Ndist
    dist_moments[i,:] = get_moments(pdists[i])
end
coal_data = initialize_coalescence_data(Ndist, Nmom)

# Set up ODE information
tspan = (0.0, T_end)
kernel = LinearKernelFunction(coalescence_coeff)
rhs = make_box_model_rhs(NumericalCoalStyle())
ODE_parameters = (Ndist=Ndist, Nmom=Nmom, pdists=pdists, kernel_func=kernel, coal_data=coal_data, dt=dt)
prob = ODEProblem(rhs, dist_moments, tspan, ODE_parameters; progress=true)
sol = solve(prob, SSPRK33(), dt=ODE_parameters.dt)
@show sol.u
plot_params!(sol, ODE_parameters; file_name="n_particle_ln_params.png")
plot_moments!(sol, ODE_parameters; file_name="n_particle_ln_moments.png")
plot_spectra!(sol, ODE_parameters; file_name="n_particle_ln_spectra.png", logxrange=(-2,5))