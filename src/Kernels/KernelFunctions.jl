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
export GolovinConstantKernelFunction
export ProductGolovinConstantKernelFunction
export GravitationalWithSigmoidECollKernelFunction
export GravitationalWithProductECollKernelFunction


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
  ProductGolovinConstantKernelFunction <: CoalescenceKernelFunction

Represents a sum of a product, a Golovin, and a constant kernel function; has the form K(x, y) = Cxy + B(x+y) + A

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ProductGolovinConstantKernelFunction{FT} <: CoalescenceKernelFunction{FT}
    "A"
    A::FT
    "B"
    B::FT
    "C"
    C::FT
end

"""
  GolovinConstantKernelFunction <: CoalescenceKernelFunction

Represents a sum of a Golovin and a constant kernel function; has the form 
K(x, y) = A + B(x + y)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GolovinConstantKernelFunction{FT} <: CoalescenceKernelFunction{FT}
    "additive rate A"
    additive_rate::FT
    "multiplicative rate B"
    multiplicative_rate::FT
end


"""
  HydrodynamicKernel <: CoalescenceKernelFunction

Represents hydrodynamic capture with a collection efficiency E_coll that
approximates the tabulated values of Shafrir and Neiburger (1962). See
Berry (1967), "Cloud Droplet Growth by Collection", which introduces the
approximation used here, even though Berry provides no derivation whatsoever.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct HydrodynamicKernelFunction{FT} <: CoalescenceKernelFunction{FT}
    A::FT
    B::FT
    D1::FT
    D2::FT
    E1::FT
    E2::FT
    F1::FT
    F2::FT
    G1::FT
    G2::FT
    G3::FT
    M_f::FT
    M_g::FT

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
#  "cloud-rain threshold"
#  threshold::FT
end

"""
  PolyDeg2KernelFunction <: CoalescenceKernelFunction

Represents a polynomial kernel of deg 2: K(x,y) = A + B(x+y) + Cxy + D(x^2+y^2)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct PolyDeg2KernelFunction{FT} <: CoalescenceKernelFunction{FT}
    "A (const)"
    A::FT
    "B (coefficient of (x+y) term)"
    B::FT
    "C (coefficient of xy term)"
    C::FT
    "D (coefficient of (x^2 + y^2) term)"
    D::FT
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


function (kern::GolovinConstantKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
    return kern.additive_rate + kern.multiplicative_rate * (x + y)
end

function (kern::HydrodynamicKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
  # dynamic viscosity of air at 10 deg Celsius, g cm^-1 s^-1
  μ_a = 1.778e-4
  # density of air at 10 deg Celsius, g cm^-3
  ρ_a = 1.2 * 1e-3
  # density of particle (water droplet), g cm^-3
  ρ_p = 1.0
  # gravitational acceleration, cm s^-2
  g = 9.81 * 1e2
  # conversion from mass (in g) to radius (in μm)
  mass2rad = m -> (3 * m / (4 * π * ρ_p))^(1/3) * 1e4
  r1 = mass2rad(x)
  r2 = mass2rad(y)
  println("r1: ", r1)
  println("r2: ", r2)
  if r1 >= r2
    r = r1
    r_s = r2
  else
    r = r2
    r_s = r1
  end
  p = r_s / r
  D = kern.D1 / (r^(kern.D2))
  E = kern.E1 / (r^(kern.E2))
  F = (kern.F1 / r)^(kern.M_f) + kern.F2
  G = (kern.G1 / r)^(kern.M_g) + kern.G2 + kern.G3 * r
#  println("----------------")
#  println("D: $D")
#  println("E: $E")
#  println("F: $F")
#  println("G: $G")
#  println("D/(p^F)")
#  println(D/(p^F))
#  println("1-p")
#  println(1.0-p)
#  println("(1-p)^G")
#  println(((1-p)^G))
#  println("E / (1-p)^G")
#  println(E / ((1-p)^G))
  Yc =  kern.A + kern.B * p + D / (p^F) + E / ((1.0 - p)^G)
  if Yc < 0
      Yc = 0
      println("Yc: ", Yc)
  end
  E_coll = (r / (r + r_s))^2 * Yc
  println("E_coll: ", E_coll)
  # terminal velocity of falling droplet of radius rad (in cm) in Stokes
  #regime, cm s^-1
  v_T(rad) = 2/9 * (ρ_p - ρ_a) / μ_a * g * rad^2
  # K(r, r') = π * (r + r')^2 * |v(r) - v(r')| * E_coll(r, r')
  v_T_diff = abs(v_T(r*1e-4) - v_T(r_s*1e-4))
  # cylindrical volume swept out per unit time, cm^3 s^-1
  cyl_vol = π * (mass2rad(x) + mass2rad(y))^2 * v_T_diff

  return cyl_vol * E_coll
end


function (kern::LongKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
  threshold = 5.0e-7
  if x < threshold && y < threshold 
    return kern.cloud_coll_coal_rate * (x.^2 + y.^2)
  else
    return kern.rain_coll_coal_rate * (x + y)
  end
end

function (kern::PolyDeg2KernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
    return kern.A + kern.B * (x + y) + kern.C * x * y + kern.D * (x^2 + y^2)
end

function (kern::ProductGolovinConstantKernelFunction{FT})(x::FT, y::FT) where {FT<:Real}
    return kern.A + kern.B * (x + y) + kern.C * x * y
end

end # module KernelFunctions.jl
