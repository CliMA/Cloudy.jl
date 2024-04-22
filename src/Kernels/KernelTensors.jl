"""
  kernel tensor module for microphysical process modeling.

A kernel tensor is a multi-dimensional lookup table that approximates
a continuous kernel function through a polynomial approximation. For example,
coalescence kernel tensors approximate the product of a collision kernel
function and a coalescence efficiency function.
"""
module KernelTensors

using LinearAlgebra
using Optim
using StaticArrays
import ..rflatten

# kernel tensors available for microphysics
export KernelTensor
export CoalescenceTensor
export get_normalized_kernel_tensor


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
    c::AbstractArray{FT}

    function CoalescenceTensor(c::SMatrix{N, N, FT, M}) where {N, M, FT <: Real}
        check_symmetry(c)
        new{FT}(size(c)[1] - 1, c)
    end
end


function CoalescenceTensor(kernel_func, order::Int, limit::FT; lower_limit::FT = FT(0)) where {FT <: Real} # TODO: kwarg?
    coef = polyfit(kernel_func, order, limit, lower_limit = lower_limit)
    CoalescenceTensor(SMatrix{order + 1, order + 1}(coef))
end

Base.broadcastable(ct::CoalescenceTensor) = Ref(ct)

"""
  polyfit(kernel_func, r::Int, limit:FT)

  - `kernel_func` - is a collision-coalescence kernel function
  - `r` - is the order of the polynomial approximation
  - `limit` - is the upper limit of particle mass allowed for this kernel function
Returns a collision-coalescence rate matrix (as array) approximating the specified
kernel function `kernel_func` to order `r` using a monomial basis in two dimensions.
"""
# Plenty of this code is likely not GPU compatible... but it's only run at initialization
function polyfit(
    kernel_func, # TODO: kwarg?
    r::Int,
    limit::FT;
    lower_limit = FT(0),
    npoints = 10,
    opt_tol = 10 * eps(FT),
    opt_max_iter = 100000,
) where {FT <: Real}
    check_symmetry(kernel_func)
    if limit < lower_limit || lower_limit < FT(0)
        error("polyfit limits improperly specified")
    end

    # use a 2d grid (with 0 < x < y and lower_limit < y < limit)
    # to fit a 2d polynomial function to kernel_func
    Δ = limit / (npoints - 1)
    x_ = map(i -> i % npoints * Δ, 0:(npoints * npoints - 1))
    y_ = map(i -> floor(i / npoints) * Δ, 0:(npoints * npoints - 1))
    ind1_ = map(s -> s > lower_limit, y_)
    ind2_ = map(s -> s < 0, y_ - x_)
    inds_ = map(i -> ind1_[i] && ind2_[i], 1:(npoints * npoints))
    x = x_[inds_]
    y = y_[inds_]

    # find the first element of the coefficients matrix - constant term in the approximation
    C_1_1 = max(eps(FT), kernel_func(FT(0), FT(0)))
    if r == 0
        return SA[C_1_1]
    end

    # define loss function
    function f(c_vec_)
        # convert the vector of unknowns to the symmetric matrix of coefficients
        c_vec = [C_1_1; c_vec_...]
        C = [
            i <= j ? c_vec[Int(j * (j - 1) / 2 + i)] : c_vec[Int(i * (i - 1) / 2 + j)] for i in 1:(r + 1),
            j in 1:(r + 1)
        ]

        z = map(x_i -> map(y_i -> kernel_func(x_i, y_i), y), x)
        for i in 1:(r + 1)
            for j in 1:(r + 1)
                z -= map(x_i -> map(y_i -> C[i, j] * x_i^(i - 1) * y_i^(j - 1), y), x)
            end
        end

        return norm(z)
    end

    c_vec0 = zeros(Int((r + 1) * (r + 2) / 2) - 1)
    res_ = optimize(f, c_vec0, g_abstol = opt_tol, iterations = opt_max_iter).minimizer
    res = [C_1_1; res_...]
    return SMatrix{r + 1, r + 1}([
        i <= j ? res[Int(j * (j - 1) / 2 + i)] : res[Int(i * (i - 1) / 2 + j)] for i in 1:(r + 1), j in 1:(r + 1)
    ])
end

"""
  check_symmetry(array)
  check_symmetry(func)

  - `array` - array that is being checked for symmety
  - `func` - function that is being checked for symmety
Throws an exception if `array` is not symmetric.
"""
# only called once at initialization, so performance not strictly necessary
function check_symmetry(array::AbstractArray{FT}) where {FT <: Real}
    if length(array) > 1
        n, m = size(array)
        if n != m
            error("array needs to be quadratic in order to be symmetric.")
        end
        for i in 1:n
            for j in (i + 1):n
                if array[i, j] != array[j, i]
                    error("array not symmetric.")
                end
            end
        end
    end
end
# only called once at initialization, so performance not strictly necessary
function check_symmetry(func)
    n_test = 1000
    test_numbers = rand(n_test, 2)
    for i in 1:n_test
        if abs(func(test_numbers[i, :]...) - func(test_numbers[i, end:-1:1]...)) > 1e-6
            error("function likely not symmetric.")
        end
    end
end

"""
  get_normalized_kernel_tensor(kernel::CoalescenceTensor{FT}, norms::Vector{FT})
  `kernel` - Coalescence kernel tensor
  `norms` - vector containing scale of number and mass/volume of particles
Returns normalized kernel tensor by using the number and mass/volume scales
"""
function get_normalized_kernel_tensor(kernel::CoalescenceTensor{FT}, norms::Tuple{FT, FT}) where {FT <: Real}
    r = kernel.r
    c = map(1:(r + 1)) do i
        map(1:(r + 1)) do j
            kernel.c[i, j] * (norms[1] * norms[2]^(FT(i + j - 2)))
        end
    end
    return CoalescenceTensor(SMatrix{r + 1, r + 1}(mapreduce(permutedims, vcat, c)))
end

end
