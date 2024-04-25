"Test case with N gamma distributions"

using DifferentialEquations
using Cloudy.KernelFunctions

include("../utils/plotting_helpers.jl")
include("../utils/box_model_helpers.jl")

FT = Float64

T_end = 200.0
coalescence_coeff = 5.0 # m^3/kg/s
dt = FT(10)

# Initial condition 
Ndist = 2
particle_number = [1e7, 1e5] # 1/m^3
mass_scale = [1e-10, 1e-9] # kg
k0 = 1.0

# Initialize ODE info
pdists = ntuple(Ndist) do i
    GammaPrimitiveParticleDistribution(particle_number[i], mass_scale[i], k0)
end
dist_moments = vcat([get_moments(dist) for dist in pdists]...)

# Set up ODE information
tspan = (0.0, T_end)
kernel = LinearKernelFunction(coalescence_coeff)
NProgMoms = map(pdists) do dist
    nparams(dist)
end
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
kernel_n = get_normalized_kernel_func(kernel, norms)
rhs = make_box_model_rhs(NumericalCoalStyle())
ODE_parameters = (pdists = pdists, kernel_func = kernel, NProgMoms = NProgMoms, norms = norms, dt = dt)
prob = ODEProblem(rhs, dist_moments, tspan, ODE_parameters; progress = true)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
@show sol.u
plot_params!(sol, ODE_parameters; file_name = "n_particle_gam_params.png")
plot_moments!(sol, ODE_parameters; file_name = "n_particle_gam_moments.png")
plot_spectra!(sol, ODE_parameters; file_name = "n_particle_gam_spectra.png", logxrange = (-11, -4))
