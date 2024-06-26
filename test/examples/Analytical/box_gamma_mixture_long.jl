"Box model with two gamma modes and using Long's kernel"

using OrdinaryDiffEq
using StaticArrays

include("../utils/box_model_helpers.jl")
include("../utils/plotting_helpers.jl")

FT = Float64

# Initial condition
# M0: 1/m^3 - M1: kg/m^3 - M2: kg^2/m^3
moment_init = [1e7, 1e-3, 2e-13, 1e5, 1e-4, 2e-13]
dist_init = (
    GammaPrimitiveParticleDistribution(FT(1e7), FT(1e-10), FT(1)), # 1e7/m^3; 1e-10 kg; k = 1
    GammaPrimitiveParticleDistribution(FT(1e5), FT(1e-9), FT(1)), # 1e5/m^3; 1e-9 kg; k = 1
)

# Solver
kernel_func = LongKernelFunction(5.236e-10, 9.44e9, 5.78) # 5.236e-10 kg; 9.44e9 m^3/kg^2/s; 5.78 m^3/kg/s
matrix_of_kernels = ntuple(2) do i
    ntuple(2) do j
        if i == j == 1
            CoalescenceTensor(kernel_func, 2, FT(5e-10))
        else
            CoalescenceTensor(kernel_func, 2, FT(1e-6), FT(5e-10))
        end
    end
end
tspan = (FT(0), FT(120))
NProgMoms = map(dist_init) do dist
    nparams(dist)
end
norms = (1e6, 1e-9) # 1e6/m^3; 1e-9 kg
coal_data = CoalescenceData(matrix_of_kernels, NProgMoms, (5e-10, Inf), norms)
rhs = make_box_model_rhs(AnalyticalCoalStyle())
ODE_parameters = (; pdists = dist_init, coal_data = coal_data, NProgMoms = NProgMoms, norms = norms, dt = FT(1.0))
prob = ODEProblem(rhs, moment_init, tspan, ODE_parameters)
sol = solve(prob, SSPRK33(), dt = ODE_parameters.dt)

plot_params!(sol, ODE_parameters; file_name = "box_gamma_mix_long_params.pdf")
plot_moments!(sol, ODE_parameters; file_name = "box_gamma_mix_long_moments.pdf")
plot_spectra!(sol, ODE_parameters; file_name = "box_gamma_mix_long_spectra.pdf", logxrange = (-11, -4))
