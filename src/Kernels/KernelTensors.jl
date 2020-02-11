"""
  kernel tensor module for microphysical process modeling.

A kernel tensor is a multi-dimensional lookup table that approximates
a continuous kernel function through a polynomial approximation. For example,
coalescence kernel tensors approximate the product of a collision kernel
function and a coalescence efficiency function.
"""
module KernelTensors

using LinearAlgebra
using HCubature

# kernel tensors available for microphysics
export KernelTensor
export CoalescenceTensor


"""
  KernelTensor{FT}

A kernel tensor approximation to a kernel function that can be used for
different microphysical processes (e.g., coalescence, breakup, sedimentation).
"""
abstract type KernelTensor{FT} end


"""
  CoalescenceTensor{FT} <: KernelTensor{FT}

Represents a Collision-Coalescence kernel.

# Constructors
  CoalescenceKernel(c::Array{FT})
  CoalescenceKernel(kernel_func, order::Int, limit:FT

# Fields

"""
struct CoalescenceTensor{FT} <: KernelTensor{FT}
  "polynomial order of the tensor"
  r::Int
  "collision-coalesence rate matrix"
  c::Array{FT}

  function CoalescenceTensor(c::Array{FT}) where {FT<:Real}
    check_symmetry(c)
    new{FT}(size(c)[1]-1, c)
  end
end


function CoalescenceTensor(kernel_func, order::Int, limit::FT) where {FT<:Real}
  coef = polyfit(kernel_func, order, limit)
  CoalescenceTensor(coef)
end


"""
  polyfit(kernel_func, r::Int, limit:FT)

  - `kernel_func` - is a collision-coalescence kernel function
  - `r` - is the order of the polynomial approximation
  - `limit` - is the upper limit of particle mass allowed for this kernel function
Returns a collision-coalescence rate matrix (as array) approximating the specified
kernel function `kernel_func` to order `2r` using a monomial basis in two dimensions.
"""
function polyfit(kernel_func, r::Int, limit::FT) where {FT<:Real}
  check_symmetry(kernel_func)
  # Build a coefficient matrix for the two-dimensional monomial
  # basis function set used for the polynomial approximation.
  Q = Array{FT}(undef, (r+1)^2, (r+1)^2)
  for i in 1:(r+1)^2
    for j in 1:(r+1)^2
      a, b = unpack_vector_index_to_poly_index(i, r+1)
      c, d = unpack_vector_index_to_poly_index(j, r+1)
      Q[i,j] = limit^(a+b+c+d+2) * (a+c+1.0)^(-1) * (b+d+1.0)^(-1)
    end
  end

  # Build the right hand side vector for the least-squares polynomial
  # approximation problem (projects of kernel_func onto polynomial basis).
  F = Array{FT}(undef, (r+1)^2)
  for i in 1:(r+1)^2
    a, b = unpack_vector_index_to_poly_index(i, r+1)
    integrand = x -> kernel_func(x) * x[1]^a * x[2]^b
    F[i], err = hcubature(integrand, [0.0, 0.0], [limit, limit], rtol=1e-4, atol=1e-6)
  end

  # Build vector of coefficients that measure importance of each monomial
  # in the polynomial approximation.
  K = inv(Q) * F

  # Because polynomial regression was done in vector-matrix setup K is now a
  # one-dimensional array. The desired coefficient matrix corresponding to K
  # is just an unpacked version of K, where each entry of K is mapped onto a
  # matrix element.
  C = Array{FT}(undef, r+1, r+1)
  for (i, kk) in enumerate(K)
    a, b = unpack_vector_index_to_poly_index(i, r+1)
    C[a+1, b+1] = K[i]
  end

  # Symmetrize collision-coalescence coefficient array to have numerically exact
  # symmetry.
  symmetrize!(C)

  return C
end


"""
  unpack_vector_index_to_poly_index(ind::Int, array_size:Int)

  - `ind` - vector index of vector that is being mapped to array
  - `array_size` - size of array the vector is being mapped to
Returns the double index of a matrix that a vector is being mapped to.
"""
function unpack_vector_index_to_poly_index(ind::Int, array_size::Int)
  if ind < 1 || array_size < 1
    error("Both $ind and $array_size need to be >= 1.")
  end
  if ind > array_size^2
    error("$array_size^2 needs to be >= $ind.")
  end
  i = ind
  r = array_size
  div(i,r) + sign(mod(i,r)) - 1, mod(i,r) - sign(mod(ind,r)) * r + r - 1
end


"""
  symmetrize!(array::Array{FT})
  - `array` - array that is being symmetrized
"""
function symmetrize!(array::Array{FT}) where {FT<:Real}
  ndim = size(array)[1]
  for i in 1:ndim
    for j in 1:ndim
      temp = 0.5 * (array[i, j] + array[j, i])
      array[i, j] = temp
      array[j, i] = temp
    end
  end
  nothing
end


"""
  check_symmetry(array)
  check_symmetry(func)

  - `array` - array that is being checked for symmety
  - `func` - function that is being checked for symmety
Throws an exception if `array` is not symmetric.
"""
function check_symmetry(array::Array{FT}) where {FT <: Real}
  if length(array) > 1
    n, m = size(array)
    if n != m
      error("array needs to be quadratic in order to be symmetric.")
    end
    for i in 1:n
      for j in i+1:n
        if array[i, j] != array[j, i]
          error("array not symmetric at ($i, $j).")
        end
      end
    end
  end
  nothing
end

function check_symmetry(func)
  n_test = 1000
  test_numbers = rand(n_test, 2)
  for i in 1:n_test
    if abs(func(test_numbers[i, :]) - func(test_numbers[i, end:-1:1])) > 1e-6
      error("function likely not symmetric.")
    end
  end
  nothing
end

end
