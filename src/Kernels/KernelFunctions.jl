"""
  kernel functions module

Kernel functions module for microphysical process modeling.
"""
module KernelFunctions

export linear

"""
  linear(x::FT, y::FT)

  - `x` - array of particle masses
Return interaction rate based on particle masses.
"""
function linear(x::Array{FT}) where {FT<:Real}
  x[1] + x[2]
end

end
