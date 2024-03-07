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
    c::Array{FT}

    function CoalescenceTensor(c::Array{FT}) where {FT <: Real}
        check_symmetry(c)
        new{FT}(size(c)[1] - 1, c)
    end
end


function CoalescenceTensor(kernel_func, order::Int, limit::FT; lower_limit::FT = FT(0)) where {FT <: Real}
    coef = polyfit(kernel_func, order, limit, lower_limit = lower_limit)
    CoalescenceTensor(coef)
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
function polyfit(
    kernel_func,
    r::Int,
    limit::FT;
    lower_limit = FT(0),
    npoints = 10,
    opt_tol = 10 * eps(FT),
    opt_max_iter = 100000,
) where {FT <: Real}
    check_symmetry(kernel_func)
    @assert FT(0) <= lower_limit < limit

    # use a 2d grid (with 0 < x < y and lower_limit < y < limit)
    # to fit a 2d polynomial function to kernel_func
    Δ = limit / (npoints - 1)
    x_ = collect(0:(npoints * npoints - 1)) .% npoints * Δ
    y_ = floor.(collect(0:(npoints * npoints - 1)) / npoints) * Δ
    inds_ = intersect(findall(s -> lower_limit <= s, y_), findall(s -> 0 <= s, y_ - x_))
    x = x_[inds_]
    y = y_[inds_]

    # find the first element of the coefficients matrix - constant term in the approximation
    C_1_1 = max(eps(FT), kernel_func(FT(0), FT(0)))
    if r == 0
        return [C_1_1]
    end

    # define loss function
    function f(c_vec_)
        # convert the vector of unknowns to the symmetric matrix of coefficients
        c_vec = [C_1_1; c_vec_...]
        C = [
            i <= j ? c_vec[Int(j * (j - 1) / 2 + i)] : c_vec[Int(i * (i - 1) / 2 + j)] for i in 1:(r + 1),
            j in 1:(r + 1)
        ]

        z = kernel_func.(x, y)
        for i in 1:(r + 1)
            for j in 1:(r + 1)
                z -= C[i, j] .* x .^ (i - 1) .* y .^ (j - 1)
            end
        end

        return norm(z)
    end

    c_vec0 = zeros(Int((r + 1) * (r + 2) / 2) - 1)
    res_ = optimize(f, c_vec0, g_abstol = opt_tol, iterations = opt_max_iter).minimizer
    res = [C_1_1; res_...]
    return [i <= j ? res[Int(j * (j - 1) / 2 + i)] : res[Int(i * (i - 1) / 2 + j)] for i in 1:(r + 1), j in 1:(r + 1)]
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
            for j in (i + 1):n
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
        if abs(func(test_numbers[i, :]...) - func(test_numbers[i, end:-1:1]...)) > 1e-6
            error("function likely not symmetric.")
        end
    end
    nothing
end

"""
  get_normalized_kernel_tensor(kernel::CoalescenceTensor{FT}, norms::Vector{FT})
  `kernel` - Coalescence kernel tensor
  `norms` - vector containing scale of number and mass/volume of particles
Returns normalized kernel tensor by using the number and mass/volume scales
"""
function get_normalized_kernel_tensor(kernel::CoalescenceTensor{FT}, norms::Vector{FT}) where {FT <: Real}
    r = kernel.r
    kernel_norm = [FT(1) / (norms[1] * norms[2]^(i + j)) for i in 0:r for j in 0:r]
    kernel_norm = reshape(kernel_norm, r + 1, r + 1)
    c = zeros(FT, r + 1, r + 1)
    for i in 1:(r + 1)
        for j in 1:(r + 1)
            c[i, j] = kernel.c[i, j] / kernel_norm[i, j]
        end
    end
    return CoalescenceTensor(c)
end

end
