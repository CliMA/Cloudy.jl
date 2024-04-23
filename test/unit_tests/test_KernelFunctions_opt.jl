using Cloudy.KernelFunctions
using JET: @test_opt

rtol = 1e-3

# Constant kernel
# Initialization
@test_opt ConstantKernelFunction(FT(π))

# Evaluations
x = 2.0
y = 4.0
norms = (10.0, 0.1)
kernel = ConstantKernelFunction(FT(π))
@test_opt kernel(x, y)

# Normalization
@test_opt get_normalized_kernel_func(kernel, norms)


# Linear kernel
# Initialization
@test_opt LinearKernelFunction(FT(π))

# Evaluations
kernel = LinearKernelFunction(FT(π))
@test_opt kernel(x, y)

# Normalization
@test_opt get_normalized_kernel_func(kernel, norms)

# Hydrodynamic kernel
# Initialization
@test_opt HydrodynamicKernelFunction(FT(π))

# Evaluations
kernel = HydrodynamicKernelFunction(FT(π))
@test_opt kernel(x, y)

# Normalization
@test_opt get_normalized_kernel_func(kernel, norms)

# Long's kernel
# Initialization
@test_opt LongKernelFunction(FT(5), FT(9), FT(1))

# Evaluations
kernel = LongKernelFunction(FT(5), FT(9), FT(1))
@test_opt kernel(x, y)

# Normalization
@test_opt get_normalized_kernel_func(kernel, norms)
