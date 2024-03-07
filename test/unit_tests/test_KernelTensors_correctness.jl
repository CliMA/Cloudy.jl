"Testing correctness of KernelTensors module."

using Cloudy.KernelTensors

import Cloudy.KernelTensors: check_symmetry, polyfit

rtol = 1e-5

# test initialization with arrays
c = [0.1 0.0; 0.0 0.2]
ker = CoalescenceTensor(c)
@test ker.r == 1
@test ker.c == Array{FT}(c)

#test initialization with kernel function
kernel_func = (x, y) -> 0.02 + x + y
ker = CoalescenceTensor(kernel_func, 1, 10.0)
@test ker.c ≈ [0.02 1.0; 1.0 0.0] rtol = rtol

# test auxilliary functions
# test symmetry checks
c = [1.0 0.0; 0.0 2.0]
@test check_symmetry(c) == nothing
c = [1.0 -0.2 0.1; -0.2 -1.0 1.1; 0.1 1.1 3.0]
@test check_symmetry(c) == nothing
c = [1.0 -0.2; 0.2 2.0]
@test_throws Exception check_symmetry(c)
c = [1.0 0.2 0.1; -0.2 -1.0 1.1; 0.1 1.1 3.0]
@test_throws Exception check_symmetry(c)
f = (x, y) -> x + y
@test check_symmetry(f) == nothing
f = (x, y) -> x - y
@test_throws Exception check_symmetry(f)

# test polynomial fitting routines
f = (x, y) -> 0.1 + 0.2 * x * y
@test polyfit(f, 1, 10.0) ≈ [0.1 0.0; 0.0 0.2] rtol = rtol
f = (x, y) -> 0.1 - 0.23 * x - 0.23 * y + 0.2 * x * y
@test polyfit(f, 1, 10.0) ≈ [0.1 -0.23; -0.23 0.2] rtol = rtol
@test polyfit(f, 1, 100.0) ≈ [0.1 -0.23; -0.23 0.2] rtol = rtol
@test polyfit(f, 1, 1000.0) ≈ [0.1 -0.23; -0.23 0.2] rtol = rtol

# test get_normalized_kernel_tensor
c = [1.0 2.0; 2.0 3.0]
ker = CoalescenceTensor(c)
ker_n = get_normalized_kernel_tensor(ker, [10, 0.2])
@test ker_n.r == 1
@test ker_n.c ≈ [10.0 4.0; 4.0 1.2] atol = 1e-12
