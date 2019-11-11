"""
  kernel tensor module for microphysical process modeling.

A kernel tensor is a multi-dimensional lookup table that approximates
a continuous kernel function through a polynomial approximation. For example,
coalescence kernel tensors approximate the product of a collision kernel
function and a coalescence efficiency function.
"""
module KernelTensors

using LinearAlgebra

# kernel tensors available for microphysics
export KernelTensor
export ConstantCoalescenceTensor
export LinearCoalescenceTensor


"""
  KernelTensor{FT}

A kernel tensor approximation to a kernel function that can be used for
different microphysical processes (e.g., coalescence, breakup, sedimentation).
"""
abstract type KernelTensor{FT} end


"""
  ConstantCoalescenceTensor{FT} <: KernelTensor

Represents a collision-coalescence tensor that is constant with respect to
internal variables (e.g., mass or size, shape).

# Constructors
  ConstantCoalescenceTensor(c::Real)

# Fields

"""
struct ConstantCoalescenceTensor{FT} <: KernelTensor{FT}
  "polynomial order of the tensor"
  r::Int
  "collision-coalesence rate matrix (1x1)"
  c::Array{FT}

  function ConstantCoalescenceTensor(c::FT) where {FT<:Real}
    if c < 0
      error("Collision-coalescence coefficient must be nonnegative.")
    end
    new{FT}(0, Array{FT}([c]))
  end
end


"""
  LinearCoalescenceTensor{FT} <: KernelTensor

Corresponds to a collision-coalescence tensor that is a linear function
with respect to internal variables (e.g., mass or size, shape).

# Constructors
  LinearCoalescenceTensor(c::Array{Real})

# Fields

"""
struct LinearCoalescenceTensor{FT} <: KernelTensor{FT}
  "polynomial order of the tensor"
  r::Int
  "collision-coalesence rate matrix (2x2)"
  c::Array{FT}

  function LinearCoalescenceTensor(c::Array{FT}) where {FT<:Real}
    if any(c .< 0)
      error("Collision-coalescence coefficient array must be nonnegative.")
    end
    if ndims(c) > 2 || length(c) != 4
      error("Collision-coalescence coefficient array must be 2x2.")
    end
    if c[2,1] != c[1,2]
      error("Collision-coalescence coefficient array must be symmetric.")
    end
    new{FT}(1, c)
  end
end

end #module KernelTensors.jl
