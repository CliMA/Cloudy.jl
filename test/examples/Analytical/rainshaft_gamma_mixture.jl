"1D Rainshaft model with coalescence and sedimentation for two gamma modes"

using OrdinaryDiffEq

include("../utils/rainshaft_helpers.jl")
include("../utils/plotting_helpers.jl")


FT = Float64

# Build discrete domain
a = 0.0
b = 3000.0
dz = (b - a) / 20
z = (a + dz / 2):dz:b

# Initial condition
# M0: 1/m^3 - M1: kg/m^3 - M2: kg^2/m^3
mom_max = [1e7, 1e-3, 2e-13, 1.0, 1e-9, 2e-18]
nm_tot = length(mom_max)
ic = initial_condition(z, mom_max)
m = ic

# Solver
dist_init = (
    GammaPrimitiveParticleDistribution(FT(1e7), FT(1e-10), FT(1)), # 1e7/m^3; 1e-10 kg; k = 1
    GammaPrimitiveParticleDistribution(FT(1), FT(1e-9), FT(1)), # 1/m^3; 1e-9 kg; k = 1
)
kernel_func = (x, y) -> 5 * (x + y) # 5 m^3/kg/s; x, y in kg
kernel = CoalescenceTensor(kernel_func, 1, FT(1e-6))
tspan = (FT(0), FT(1000))
NProgMoms = map(dist_init) do dist
    nparams(dist)
end
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
coal_data = CoalescenceData(kernel, NProgMoms, (5e-10, Inf), norms)
rhs = make_rainshaft_rhs(AnalyticalCoalStyle())
ODE_parameters = (;
    pdists = dist_init,
    coal_data = coal_data,
    NProgMoms = NProgMoms,
    norms = norms,
    vel = ((50.0, 1.0 / 6),), # 50 m/s/kg^(1/6)
    dz = dz,
    dt = 1.0,
)
prob = ODEProblem(rhs, m, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)
res = sol.u

plot_rainshaft_results(z, res, ODE_parameters, file_name = "rainshaft_gamma_mixture.pdf", print = true)
#plot_rainshaft_contours(z, sol.t, res, ODE_parameters, file_name="rainshaft_gamma_mixture_contours.png")
