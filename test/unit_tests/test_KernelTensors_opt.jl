using Cloudy.KernelTensors
import Cloudy.KernelTensors: check_symmetry, polyfit
using StaticArrays
using JET: @test_opt

# test initialization with arrays
c = SA[0.1 0.0; 0.0 0.2]
@test_opt CoalescenceTensor(c)
@test 80 >= @allocated CoalescenceTensor(c)

# test initialization with kernel function
kernel_func = (x, y) -> 0.02 + x + y
# @test_opt CoalescenceTensor(kernel_func, 1, 10.0) # TODO polyfit fails optimization

# test symmetry checks
@test_opt check_symmetry(c)
@test 0 == @allocated check_symmetry(c)

# test polynomial fitting routines
kernel_func = (x, y) -> 0.1 + 0.2 * x * y
@test_opt kernel_func(1.0, 2.0)
# @test_opt polyfit(kernel_func, 1, 10.0)  # TODO polyfit fails optimization

# test get_normalized_kernel_tensor
c = SA[1.0 2.0; 2.0 3.0]
ker = CoalescenceTensor(c)
get_normalized_kernel_tensor(ker, (10.0, 0.2))
@test_opt get_normalized_kernel_tensor(ker, (10.0, 0.2)) # TODO: fails optimization (but only called once at initialization...)
