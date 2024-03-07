"""
  kernel function module for microphysical process modeling.

A kernel function is a multi-dimensional function representing interaction rates. For example,
a coalescence kernel is the product of a collision kernel
function and a coalescence efficiency function.
"""
module KernelFunctions

using DocStringExtensions

# kernel functions available for microphysics
export KernelFunction
export CoalescenceKernelFunction
export ConstantKernelFunction
export LinearKernelFunction
export HydrodynamicKernelFunction
export LongKernelFunction
export get_normalized_kernel_func

"""
  KernelFunction

A kernel kernel function that can be used for
different microphysical processes (e.g., coalescence, breakup, sedimentation).
"""
abstract type KernelFunction{FT} end
abstract type CoalescenceKernelFunction{FT} <: KernelFunction{FT} end


"""
  ConstantKernelFunction <: CoalescenceKernelFunction

Represents a constant collision-coalescence kernel.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ConstantKernelFunction{FT} <: CoalescenceKernelFunction{FT}
    "collision-coalesence rate"
    coll_coal_rate::FT
end


"""
  LinearKernelFunction <: CoalescenceKernelFunction

Represents a linear collision-coalescence kernel.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LinearKernelFunction{FT} <: CoalescenceKernelFunction{FT}
    "collision-coalesence rate"
    coll_coal_rate::FT
end

"""
  HydrodynamicKernelFunction <: CoalescenceKernelFunction

Represents a hydrodynamic collision-coalescence kernel.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct HydrodynamicKernelFunction{FT} <: CoalescenceKernelFunction{FT}
    "coalescence efficiency"
    coal_eff::FT
end

"""
  LongKernelFunction <: CoalescenceKernelFunction

Represents the Long's collision-coalescence kernel.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LongKernelFunction{FT} <: CoalescenceKernelFunction{FT}
    "mass threshold"
    x_threshold::FT
    "collision-coalesence rate below threshold"
    coal_rate_below_threshold::FT
    "collision-coalesence rate above threshold"
    coal_rate_above_threshold::FT
end


"""
    (kernel::KernelFunction)(x::FT, y::FT)
    
Returns evaluations of kernel function at vector of paired locations (x[1], x[2]).
"""
function (kern::ConstantKernelFunction)(x::FT, y::FT) where {FT <: Real}
    return kern.coll_coal_rate
end

function (kern::LinearKernelFunction{FT})(x::FT, y::FT) where {FT <: Real}
    return kern.coll_coal_rate * (x + y)
end

function (kern::HydrodynamicKernelFunction{FT})(x::FT, y::FT) where {FT <: Real}
    r1 = (3 / 4 / π * x)^(1 / 3)
    r2 = (3 / 4 / π * y)^(1 / 3)
    A1 = π * r1^2
    A2 = π * r2^2
    return kern.coal_eff * (r1 + r2)^2 * abs(A1 - A2)
end

function (kern::LongKernelFunction{FT})(x::FT, y::FT) where {FT <: Real}
    if x < kern.x_threshold && y < kern.x_threshold
        return kern.coal_rate_below_threshold * (x^2 + y^2)
    else
        return kern.coal_rate_above_threshold * (x + y)
    end
end

"""
    get_normalized_kernel_func(kern::ConstantKernelFunction, norms::Vector{FT})
    `kern` - kernel function
    `norms` - vector containing scale of number and mass/volume of particles
Returns normalized kernel
"""
function get_normalized_kernel_func(kern::ConstantKernelFunction, norms::Vector{FT}) where {FT <: Real}
    return ConstantKernelFunction(kern.coll_coal_rate * norms[1])
end

function get_normalized_kernel_func(kern::LinearKernelFunction, norms::Vector{FT}) where {FT <: Real}
    return LinearKernelFunction(kern.coll_coal_rate * norms[1] * norms[2])
end

function get_normalized_kernel_func(kern::HydrodynamicKernelFunction, norms::Vector{FT}) where {FT <: Real}
    return HydrodynamicKernelFunction(kern.coal_eff * norms[1] * norms[2]^FT(4 / 3))
end

function get_normalized_kernel_func(kern::LongKernelFunction, norms::Vector{FT}) where {FT <: Real}
    return LongKernelFunction(
        kern.x_threshold / norms[2],
        kern.coal_rate_below_threshold * norms[1] * norms[2]^2,
        kern.coal_rate_above_threshold * norms[1] * norms[2],
    )
end

end
