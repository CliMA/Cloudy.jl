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
    return kern.coal_eff * π * (x + y)^2 * abs(x^2 - y^2)
end

end
