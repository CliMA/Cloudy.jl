"""
  kernel function module for microphysical process modeling.

A kernel function is a multi-dimensional function representing interaction rates. For example,
a coalescence kernel is the product of a collision kernel
function and a coalescence efficiency function.
"""
module KernelFunctions

using LinearAlgebra
using DocStringExtensions

# kernel functions available for microphysics
export KernelFunction
export CoalescenceKernelFunction
export ConstantKernelFunction
export LinearKernelFunction
export ProductKernelFunction
export LongKernelFunction
export GravitationalKernelFunction


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
  ProductKernelFunction <: CoalescenceKernelFunction

Represents a product collision-coalescence kernel.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ProductKernelFunction{FT} <: CoalescenceKernelFunction{FT}
  "collision-coalesence rate"
  coll_coal_rate::FT
end


"""
  LongKernelFunction <: CoalescenceKernelFunction

Represents a Long-1974 collision-coalescence kernel.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LongKernelFunction{FT} <: CoalescenceKernelFunction{FT}
  "collision-coalesence rate for cloud droplets"
  cloud_coll_coal_rate::FT
  "collision-coalesence rate for rain droplets"
  rain_coll_coal_rate::FT
  "cloud-rain threshold"
  threshold::FT
end

"""
  GravitationalKernelFunction <: CoalescenceKernelFunction

Represents a generalized gravitational kernel with Ec=1.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GravitationalKernelFunction{FT} <: CoalescenceKernelFunction{FT}
  "collision-coalesence rate"
  coll_coal_rate::FT

  "maximum particle size"
  xmax::FT
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

function (kern::ProductKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
  return kern.coll_coal_rate * x * y
end

function (kern::LongKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
  if x < kern.threshold && y < kern.threshold 
    return kern.cloud_coll_coal_rate * (x.^2 + y.^2)
  else
    return kern.rain_coll_coal_rate * (x + y)
  end
end

function (kern::GravitationalKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
  return kern.coll_coal_rate * π / kern.xmax^(2/3) * abs(x^(2/3) - y^(2/3)) * (x+y)^2
end

end # module KernelFunctions.jl