"1D Rainshaft model with coalescence and sedimentation for a single gamma distribution"

using DifferentialEquations

include("../utils/rainshaft_helpers.jl")
include("../utils/plotting_helpers.jl")


FT = Float64

# Build discrete domain
a = 0.0
b = 3000.0
dz = (b - a) / 60
z = (a + dz / 2):dz:b

# Initial condition
mom_max = [1, 1, 2]
nmom = length(mom_max)
ic = initial_condition(z, mom_max)
m = ic

# Solver
dist_init = [GammaPrimitiveParticleDistribution(FT(1), FT(1), FT(1))]
kernel_func = x -> 5e-3 * (x[1] + x[2])
kernel = CoalescenceTensor(kernel_func, 1, FT(500))
tspan = (FT(0), FT(1000))
NProgMoms = [nparams(dist) for dist in dist_init]
coal_data = initialize_coalescence_data(AnalyticalCoalStyle(), NProgMoms, kernel)
rhs = make_rainshaft_rhs(AnalyticalCoalStyle())
ODE_parameters = (;
    pdists = dist_init,
    kernel = kernel,
    coal_data = coal_data,
    dist_thresholds = [Inf],
    vel = [(2.0, 1.0 / 6)],
    dz = dz,
    dt = 1.0,
)
prob = ODEProblem(rhs, m, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
res = sol.u

plot_rainshaft_results(z, res, ODE_parameters, file_name = "rainshaft_single_gamma.pdf")
