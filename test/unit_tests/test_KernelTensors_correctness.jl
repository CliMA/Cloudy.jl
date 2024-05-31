"Testing correctness of KernelTensors module."

using Cloudy.KernelTensors
using StaticArrays

import Cloudy.KernelTensors: check_symmetry, polyfit

rtol = 1e-5

# test initialization with arrays
c = SA[0.1 0.0; 0.0 0.2]
ker = CoalescenceTensor(c)
@test ker.c == Array{FT}(c)

#test initialization with kernel function
kernel_func = LinearKernelFunction(FT(0.02))
ker = CoalescenceTensor(kernel_func, 1, 10.0)
@test ker.c ≈ SA[0.02 1.0; 1.0 0.0] rtol = rtol
@test ker.c isa SMatrix{2, 2}{FT}

# test auxilliary functions
# test symmetry checks
c = [1.0 0.0; 0.0 2.0]
check_symmetry(c)
c = [1.0 -0.2 0.1; -0.2 -1.0 1.1; 0.1 1.1 3.0]
check_symmetry(c)
c = [1.0 -0.2; 0.2 2.0]
@test_throws Exception check_symmetry(c)
c = [1.0 0.2 0.1; -0.2 -1.0 1.1; 0.1 1.1 3.0]
@test_throws Exception check_symmetry(c)
f = (x, y) -> x + y
check_symmetry(f)
f = (x, y) -> x - y
@test_throws Exception check_symmetry(f)

# test polynomial fitting routines
f = (x, y) -> 0.1 + 0.2 * x * y
c = polyfit(f, 1, 10.0)
@test c isa SMatrix{2, 2}{FT}
@test c ≈ [0.1 0.0; 0.0 0.2] rtol = rtol
f = (x, y) -> 0.1 - 0.23 * x - 0.23 * y + 0.2 * x * y
@test polyfit(f, 1, 10.0) ≈ [0.1 -0.23; -0.23 0.2] rtol = rtol
@test polyfit(f, 1, 100.0) ≈ [0.1 -0.23; -0.23 0.2] rtol = rtol
@test polyfit(f, 1, 1000.0) ≈ [0.1 -0.23; -0.23 0.2] rtol = rtol

# test get_normalized_kernel_tensor
c = SA[1.0 2.0; 2.0 3.0]
ker = CoalescenceTensor(c)
ker_n = get_normalized_kernel_tensor(ker, (10.0, 0.2))
@test ker_n.c ≈ [10.0 4.0; 4.0 1.2] atol = 1e-12
@test ker_n.c isa SMatrix{2, 2}{FT}
