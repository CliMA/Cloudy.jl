"Testing correctness of KernelTensors module."

using Cloudy.KernelTensors

import Cloudy.KernelTensors: check_symmetry, symmetrize!,
                             unpack_vector_index_to_poly_index, polyfit

rtol=1e-5

# test initialization with arrays
c = [0.1 0.0; 0.0 0.2]
ker = CoalescenceTensor(c)
@test ker.r  == 1
@test ker.c == Array{FT}(c)

#test initialization with kernel function
kernel_func = x -> 0.02 + x[1] + x[2]
ker = CoalescenceTensor(kernel_func, 1, 10.0)
@test ker.c ≈ [0.02 1.0; 1.0 0.0] rtol=rtol

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
f = x -> x[1] + x[2]
@test check_symmetry(f) == nothing
f = x -> x[1] - x[2]
@test_throws Exception check_symmetry(f)

# test symmetrization of arrays
c = [1.0 -0.1; 0.0 2.0]
symmetrize!(c)
@test check_symmetry(c) == nothing
@test c == [1.0 -0.05; -0.05 2.0]
c = [1.0 0.2 0.1; -0.2 -1.0 1.1; 0.1 1.1 3.0]
symmetrize!(c)
@test check_symmetry(c) == nothing
@test c == [1.0 0.0 0.1; 0.0 -1.0 1.1; 0.1 1.1 3.0]

# test unpacking of vector indices
i, r = 1, 1
@test unpack_vector_index_to_poly_index(i, r) == (0, 0)
i, r = 0, 1
@test_throws Exception unpack_vector_index_to_poly_index(i, r)
i, r = 1, 0
@test_throws Exception unpack_vector_index_to_poly_index(i, r)
i, r = 2, 1
@test_throws Exception unpack_vector_index_to_poly_index(i, r)
i, r = 1, 2
@test unpack_vector_index_to_poly_index(i, r) == (0, 0)
i, r = 2, 2
@test unpack_vector_index_to_poly_index(i, r) == (0, 1)
i, r = 3, 2
@test unpack_vector_index_to_poly_index(i, r) == (1, 0)
i, r = 4, 2
@test unpack_vector_index_to_poly_index(i, r) == (1, 1)
i, r = 1, 3
@test unpack_vector_index_to_poly_index(i, r) == (0, 0)
i, r = 2, 3
@test unpack_vector_index_to_poly_index(i, r) == (0, 1)
i, r = 3, 3
@test unpack_vector_index_to_poly_index(i, r) == (0, 2)
i, r = 4, 3
@test unpack_vector_index_to_poly_index(i, r) == (1, 0)
i, r = 5, 3
@test unpack_vector_index_to_poly_index(i, r) == (1, 1)
i, r = 6, 3
@test unpack_vector_index_to_poly_index(i, r) == (1, 2)
i, r = 7, 3
@test unpack_vector_index_to_poly_index(i, r) == (2, 0)
i, r = 8, 3
@test unpack_vector_index_to_poly_index(i, r) == (2, 1)
i, r = 9, 3
@test unpack_vector_index_to_poly_index(i, r) == (2, 2)

# test polynomial fitting routines
c = x -> 0.1 + 0.2*x[1]*x[2]
@test polyfit(c, 1, 10.0) ≈ [0.1 0.0; 0.0 0.2] rtol=rtol
c = x -> 0.1 - 0.23*x[1] - 0.23*x[2] +  0.2*x[1]*x[2]
@test polyfit(c, 1, 100.0) ≈ [0.1 -0.23; -0.23 0.2] rtol=rtol
c = x -> 0.1 - 0.23*x[1] - 0.23*x[2] +  0.2*x[1]*x[2]
@test polyfit(c, 1, 1000.0) ≈ [0.1 -0.23; -0.23 0.2] rtol=rtol
c = x -> 0.1 - 0.23*x[1] - 0.23*x[2] +  0.2*x[1]*x[2]
@test polyfit(c, 1, 10000.0) ≈ [0.1 -0.23; -0.23 0.2] rtol=rtol
c = x -> 0.1 - 0.23*x[1] - 0.23*x[2] +  0.2*x[1]*x[2]
@test polyfit(c, 1, 100000.0) ≈ [0.1 -0.23; -0.23 0.2] rtol=1e-4
c = x -> 0.1 - 0.23*x[1] - 0.23*x[2] +  0.2*x[1]*x[2]
@test polyfit(c, 1, 1e6) ≈ [0.1 -0.23; -0.23 0.2] rtol=5e-3
