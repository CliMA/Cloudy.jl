"Testing correctness of KernelFunctions module."

using Cloudy.KernelFunctions

# Constant kernel
# Initialization
A = 1.0
x = 2.0
y = 4.0
kernel = ConstantKernelFunction(A)
@test kernel.coll_coal_rate == A

# evaluation, symmetry
@test kernel(x, y) == A
@test kernel(y, x) == A


# Linear kernel
# Initialization
B = 1.0
kernel = LinearKernelFunction(B)
@test kernel.coll_coal_rate == A

# evaluation, symmetry
@test kernel(x, y) == B * (x + y)
@test kernel(y, x) == B * (x + y)


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
