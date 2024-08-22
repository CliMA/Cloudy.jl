"Testing correctness of KernelFunctions module."

using Cloudy.KernelFunctions

# Constant kernel
# Initialization
A = 1.0
x = 2.0
y = 4.0
norms = (100.0, 0.001)
kernel = ConstantKernelFunction(A)
@test kernel.coll_coal_rate == A

# evaluation, symmetry
@test kernel(x, y) == A
@test kernel(y, x) == A

# normalization
kernel_n = get_normalized_kernel_func(kernel, norms)
@test kernel_n isa ConstantKernelFunction
@test kernel_n.coll_coal_rate == A * norms[1]

# Linear kernel
# Initialization
B = 1.0
kernel = LinearKernelFunction(B)
@test kernel.coll_coal_rate == B

# evaluation, symmetry
@test kernel(x, y) == B * (x + y)
@test kernel(y, x) == B * (x + y)

# normalization
kernel_n = get_normalized_kernel_func(kernel, norms)
@test kernel_n isa LinearKernelFunction
@test kernel_n.coll_coal_rate == B * norms[1] * norms[2]

# Hydrodynamic kernel
# Initialization
Ec = 1.0
kernel = HydrodynamicKernelFunction(Ec)
@test kernel.coal_eff == Ec

# evaluation, symmetry 
r1 = (3 / 4 / π * x)^(1 / 3)
r2 = (3 / 4 / π * y)^(1 / 3)
A1 = π * r1^2
A2 = π * r2^2
@test kernel(x, y) == Ec * (r1 + r2)^2 * abs(A1 - A2)
@test kernel(x, y) == kernel(y, x)

# normalization
kernel_n = get_normalized_kernel_func(kernel, norms)
@test kernel_n isa HydrodynamicKernelFunction
@test kernel_n.coal_eff ≈ Ec * norms[1] * norms[2]^(4.0 / 3) atol = 1e-12

# Long's kernel
# Initialization
x_th = 1.0
C_bl = 10.0
C_ab = 5.0
kernel = LongKernelFunction(x_th, C_bl, C_ab)
@test kernel.x_threshold == x_th
@test kernel.coal_rate_below_threshold == C_bl
@test kernel.coal_rate_above_threshold == C_ab

# evaluation, symmetry
x_b = 0.25
y_b = 0.5
kernel(x_b, y_b) == C_bl * (x^2 + y^2)
kernel(x, y) == C_ab * (x + y)
kernel(x_b, y_b) == kernel(y_b, x_b)
kernel(x, y) == kernel(y, x)

# normalization
kernel_n = get_normalized_kernel_func(kernel, norms)
@test kernel_n isa LongKernelFunction
@test kernel_n.x_threshold == x_th / norms[2]
@test kernel_n.coal_rate_below_threshold == C_bl * norms[1] * norms[2]^2
@test kernel_n.coal_rate_above_threshold == C_ab * norms[1] * norms[2]

# type stability
r = 1
for FT in (Float64, Float32)
    kernel_func = CL.KernelFunctions.LinearKernelFunction(FT(5e0))
    kernel_tens = CL.KernelTensors.CoalescenceTensor(kernel_func, r, FT(5e-10))
    @test kernel_tens isa CL.KernelTensors.CoalescenceTensor{r + 1, FT, (r + 1) * (r + 1)}
end
