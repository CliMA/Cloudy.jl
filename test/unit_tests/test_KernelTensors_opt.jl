using Cloudy.KernelTensors
using JET: @test_opt

# # test initialization with arrays
# c = [0.1 0.0; 0.0 0.2]
@test_opt CoalescenceTensor(c)

# #test initialization with kernel function
kernel_func = (x, y) -> 0.02 + x + y
# @test_opt CoalescenceTensor(kernel_func, 1, 10.0)

# # test symmetry checks
@test_opt check_symmetry(c)

# # test polynomial fitting routines
kernel_func = (x, y) -> 0.1 + 0.2 * x * y
# @test_opt polyfit(kernel_func, 1, 10.0)

# test get_normalized_kernel_tensor
c = [1.0 2.0; 2.0 3.0]
ker = CoalescenceTensor(c)
@test_opt get_normalized_kernel_tensor(ker, [10.0, 0.2])
