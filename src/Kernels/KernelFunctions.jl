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
    (kernel::KernelFunction)(x::FT, y::FT)
    
Returns evaluations of kernel function at locations `x` and `y`.
"""
function (kern::ConstantKernelFunction)(x::FT, y::FT) where {FT<:Real}
  return kern.coll_coal_rate
end

function (kern::LinearKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
  return kern.coll_coal_rate * (x + y)
end

end