"Test case with a single exponential distribution"

using OrdinaryDiffEq
using Cloudy.KernelFunctions

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Dynamical info
T_end = 1.0
coalescence_coeff = 1.0 # m^3/kg/s
dt = FT(0.1)

# Initial condition 
Ndist = 1
particle_number = [1e8] # 1/m^3
mass_scale = [1e-7] # kg

# Initialize ODE info
pdists = ntuple(Ndist) do i
    ExponentialPrimitiveParticleDistribution(particle_number[i], mass_scale[i])
end
dist_moments = vcat([get_moments(dist) for dist in pdists]...)

# Set up ODE
tspan = (0.0, T_end)
kernel = LinearKernelFunction(coalescence_coeff)
NProgMoms = map(pdists) do dist
    nparams(dist)
end
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
kernel_n = get_normalized_kernel_func(kernel, norms)
rhs = make_box_model_rhs(NumericalCoalStyle())
ODE_parameters = (pdists = pdists, kernel_func = kernel_n, NProgMoms = NProgMoms, norms = norms, dt = dt)
prob = ODEProblem(rhs, dist_moments, tspan, ODE_parameters; progress = true)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
@show sol.u
plot_params!(sol, ODE_parameters; file_name = "single_particle_exp_params.png")
plot_moments!(sol, ODE_parameters; file_name = "single_particle_exp_moments.png")
plot_spectra!(sol, ODE_parameters; file_name = "single_particle_exp_spectra.png", logxrange = (-9, 0))
