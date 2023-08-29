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
x = 2.0
y = 4.0
kernel = LinearKernelFunction(B)
@test kernel.coll_coal_rate == A

# evaluation, symmetry
@test kernel(x, y) == B * (x + y)
@test kernel(y, x) == B * (x + y)