"""
  Microphysical source functions

"""
module Sources

using HCubature
using NIntegration
using NLsolve
using Distributions
using SpecialFunctions: gamma
using ..ParticleDistributions
using ..KernelFunctions
# methods that compute source terms from microphysical parameterizations
export get_coalescence_integral
export get_coalescence_integral_moment
export get_breakup_integral_moment
export Beard_Ochs_coalescence_efficiency
export constant_coalescence_efficiency

"""
get_coalescence_integral(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int)

Returns the collision-coalescence integral at points `x`.
"""
function get_coalescence_integral(x::Array{FT}, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, n_samples::Int) where {FT<:Real}
  # monte carlo samples
  y = sort(sample(pdist, n_samples))

  out = zeros(size(x))
  # source contribution to collision integral
  for (i, xx) in enumerate(x)
    for yy in  y[y .<= xx]
      out[i] += 0.5 * pdist(xx - yy) * kernel(xx - yy, yy)
    end
  end

  # sink contribution to collision integral
  out -= pdist.(x) .* sum(kernel.(x, y'), dims=2)

  # normalize to get correct monte carlo average
  out *= pdist.n / length(y) 

end


"""
get_coalescence_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
    n_samples_mc::Int, coalescence_efficiency::Function)

Returns the collision-coalescence integral at moment of order `k`, using
Monte Carlo integration
"""
function get_coalescence_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
    n_samples_mc::Int, coalescence_efficiency::Function
    ) where {FT<:Real}

  # monte carlo samples
  x = zeros(n_samples_mc)
  y = zeros(n_samples_mc)
  for i in 1:n_samples_mc
    x[i], y[i] = ParticleDistributions.sample(pdist, 2)
  end

  # collision integral for the k-th moment
  out = 0.0
  for i in 1:n_samples_mc
    out += 0.5 * coalescence_efficiency(x[i], y[i])* kernel(x[i], y[i]) * ((x[i]+y[i])^k - x[i]^k - y[i]^k)
  end

  # normalize to get correct monte carlo average
  out *= (pdist.n)^2 / n_samples_mc

  return out
end


"""
get_coalescence_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}
    )

Returns the collision-coalescence integral at moment of order `k`, using
quadrature
"""
function get_coalescence_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}
    ) where {FT<:Real}

  function coal_integrand(x)
      integrand = 0.5 * pdist(x[1]) * pdist(x[2]) + kernel(x[1], x[2]) * ((x[1] + x[2])^k -x[1]^k - x[2]^k)
      return integrand
  end
  max_mass = ParticleDistributions.moment(pdist, 1.0)
  out = hcubature(coal_integrand, [0, 0], [max_mass, max_mass]; rtol=1e-4)

  return out[1]
end


"""
get_breakup_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
    M0_init::FT, M1_init::FT, n_samples_mc::Int, coal_eff::Function
    )

Returns the breakup integral at moment of order `k`, using Monte Carlo
integration.
"""
function get_breakup_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
    M0_init::FT, M1_init::FT, n_samples_mc::Int,
    coalescence_efficiency::Function
    ) where {FT<:Real}

  # The full equation for the time rate of change of the k-th moment, M_k, of
  # the droplet mass distribution due to collisional breakup is given by
  # (e.g., Hu, Z., and Srivastava, R. C. (1995). Evolution of Raindrop Size
  # Distribution by Coalescence, Breakup, and Evaporation: Theory and
  # Observations, Journal of Atmospheric Sciences, 52(10), 1761-1783):
  # ∂M_k/∂t = 0.5 ∫x^k P(x;y,z) dx ∫f(y) dy ∫f(z) B(y, z) dz
  #           - ∫x^k f(x) dx ∫f(z) B(x, z) / (x + z) dz ∫y P(y;x,z) dy
  # See also CliMA microphyscis documentation
  #
  # Idea: Define G(x) = P(x,y,z) / (y+z). Then, the term x^k * G(x) can
  # be turned into a Gamma distribution with shape parameter k+1 and scale
  # parameter θ := M1 / (n*M0) by multiplying with α := θ^(1-k) / gamma(k+1) 
  # This means that we can remove α * G(x) from the source term below
  # (but have to multiply by 1/α to not change its value) and sample the
  # x_i from a Gamma(k+1, θ) distribution.
  # A similar trick is possible for the sink term, where the term y * G(y)
  # can be turned into a Gamma(2, θ) by dividing (and re-multiplying)
  # by Γ(2)=1.

  # Fragment distribution function P
  # The fragment distribution function implemented here is taken from
  # Feingold, G., Tzivion (Tzitzvashvili), S., & Leviv, Z. (1988). Evolution
  # of Raindrop Spectra. Part I: Solution to the Stochastic
  # Collection/Breakup Equation Using the Method of Moments, Journal of
  # Atmospheric Sciences, 45(22), 3387-3399.

  # n is a positive integer that characterizes the fragment concentration
  # (n=1 means no evolution with respect to the droplet concentration)
  n = 2
  function P(x, y, z, n, M0, M1)
      return (n * M0 / M1)^2 * (y + z) * exp(-n * M0 / M1 * x)
  end
  P(x, y, z) = P(x, y, z, n, M0_init, M1_init)

  # Defining these parameters will save us some typing
  θ = M1_init / (n * M0_init)
  α = θ^(1-k)/ gamma(k+1)

  # Breakup kernel
  function B(x, y, kernel, coalescence_efficiency::Function)
      return kernel(x, y) * (1.0 - coalescence_efficiency(x, y))
  end
  B(x, y) = B(x, y, kernel, coalescence_efficiency)

  # Monte Carlo samples for source term
  x = zeros(n_samples_mc)
  y = zeros(n_samples_mc)
  z = zeros(n_samples_mc)

  for i in 1:n_samples_mc
      x[i], y[i], z[i] = ParticleDistributions.sample(pdist, 3)
  end

  # Source breakup integral for the k-th moment
  source = 0.0
  for i in 1:n_samples_mc
      source += 0.5 * B(y[i], z[i]) * (y[i] + z[i]) * 1/α
  end

  # Sink breakup integral for the k-th moment
  sink = 0.0
  for i in 1:n_samples_mc
      sink += x[i]^k * B(x[i], z[i])
  end

  # Normalize to get correct Monte Carlo average
  out = source - sink
  out *= (pdist.n)^2 / n_samples_mc

   return out
end


"""
get_breakup_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
    M0_init::FT, M1_init::FT
    )

Returns the breakup integral at moment of order `k`, using quadrature.
"""
function get_breakup_integral_moment(
    k::Int, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT},
    M0_init::FT, M1_init::FT
    ) where {FT<:Real}

  # The full equation for the time rate of change of the k-th moment, M_k, of
  # the droplet mass distribution due to collisional breakup is given by
  # (e.g., Hu, Z., and Srivastava, R. C. (1995). Evolution of Raindrop Size
  # Distribution by Coalescence, Breakup, and Evaporation: Theory and
  # Observations, Journal of Atmospheric Sciences, 52(10), 1761-1783):
  # ∂M_k/∂t = 0.5 ∫x^k P(x;y,z) dx ∫f(y) dy ∫f(z) B(y, z) dz
  #           - ∫x^k f(x) dx ∫f(z) B(x, z) / (x + z) dz ∫y P(y;x,z) dy
  # See also CliMA microphyscis documentation
  #
  # Idea: Define G(x) = P(x,y,z) / (y+z). Then, the term x^k * G(x) can
  # be turned into a Gamma distribution with shape parameter k+1 and scale
  # parameter θ := M1 / (n*M0) by multiplying with α := θ^(1-k) / gamma(k+1) 
  # This means that we can remove α * G(x) from the source term below
  # (but have to multiply by 1/α to not change its value) and sample the
  # x_i from a Gamma(k+1, θ) distribution.
  # A similar trick is possible for the sink term, where the term y * G(y)
  # can be turned into a Gamma(2, θ) by dividing (and re-multiplying)
  # by Γ(2)=1.

  # Fragment distribution function P
  # The fragment distribution function implemented here is taken from
  # Feingold, G., Tzivion (Tzitzvashvili), S., & Leviv, Z. (1988). Evolution
  # of Raindrop Spectra. Part I: Solution to the Stochastic
  # Collection/Breakup Equation Using the Method of Moments, Journal of
  # Atmospheric Sciences, 45(22), 3387-3399.

  # n is a positive integer that characterizes the fragment concentration
  # (n=1 means no evolution with respect to the droplet concentration)
  n = 3
  function P(x, y, z, n, M0, M1)
      return (n * M0 / M1)^2 * (y + z) * exp(-n * M0 / M1 * x)
  end
  P(x, y, z) = P(x, y, z, n, M0_init, M1_init)

  # Breakup kernel
  function B(x, y, kernel, coalescence_efficiency::Function)
      return kernel(x, y) * (1.0 - coalescence_efficiency(x, y))
  end
  B(x, y) = B(x, y, kernel, coalescence_efficiency)

#  function source_integrand(x)
#     0.5 * x[1]^k * P(x[1], x[2], x[3]) * pdist(x[2]) * pdist(x[3]) * B(x[2], x[3]) 
#  end
#
#  function sink_integrand(x)
#      x[1]^k * pdist(x[1]) * pdist(x[3]) * B(x[1], x[3]) / (x[1] + x[3]) * x[2] * P(x[2], x[1], x[3])
#  end
  function source_integrand(x, y, z)
      0.5 * x^k * P(x, y, z) * pdist(y) * pdist(z) * B(y, z) 
  end

  function sink_integrand(x, y, z)
      x^k * pdist(x) * pdist(z) * B(x, z) / (x + z) * y * P(y, x, z)
  end

  max_mass = ParticleDistributions.moment(pdist, 1.0)
  println("total mass: $max_mass")

  (I_source, E_source, n_source, R_source) = nintegrate(
        source_integrand, (0.0, 0.0, 0.0), (max_mass, max_mass, max_mass);
        reltol=1e-6, abstol=eps(), maxevals=1000000
  )
  (I_sink, E_sink, n_sink, R_sink) = nintegrate(
        sink_integrand, (0.0, 0.0, 0.0), (max_mass, max_mass, max_mass);
        reltol=1e-6, abstol=eps(), maxevals=1000000
  )
  #source = hcubature(source_integrand, zeros(3), [max_mass, max_mass, max_mass]; rtol=1e-8, maxevals=1000)
  #sink = hcubature(sink_integrand, zeros(3), [max_mass, max_mass, max_mass]; rtol=1e-8)

  #println("breakup source: $(I_source)")
  #println("breakup sink: $(I_sink)")
  return I_source - I_sink

end


function Beard_Ochs_coalescence_efficiency(x, y)

    # Coalescence efficiency given in Beard and Ochs (1995):
    # Beard, K. V., , and H. T. Ochs, 1995: Collisions between small
    # precipitation drops. Part II: Formulas for coalescence, temporary
    # coalescence, and satellites. J. Atmos. Sci., 52 , 3977–3996.
    #
    # It is valid for droplets with radii smaller than 300 μm. If coalescence
    # efficiencies for droplets greater than 300 μm are needed, see e.g.
    # section 2.a) of the following paper for a parameterization that covers
    # the size spectrum from < 300 μm to 600 μm and beyond:
    # Seifert, A., Khain, A., Blahak, U. and Beheng, K.D., 2005. Possible
    # effects of collisional breakup on mixed-phase deep convection simulated
    # by a spectral (bin) cloud model. Journal of the Atmospheric Sciences,
    # 62(6), pp.1917-1931.

    # dynamic viscosity of air at 10 deg Celsius, g cm^-1 s^-1
    μ_a = 1.778e-4
    # density of air at 10 deg Celsius, g cm^-3
    ρ_a = 1.2 * 1e-3
    # density of particle (water droplet), g cm^-3
    ρ_p = 1.0
    # surface tension of water in g/s^2
    σ_w = 72.8
    # gravitational acceleration, cm s^-2
    g = 9.81 * 1e2
    # conversion from mass (in g) to radius (in μm)
    mass2rad = m -> (3 * m / (4 * π * ρ_p))^(1/3) * 1e4
    r1 = mass2rad(x)
    r2 = mass2rad(y)
    # R: radius of the bigger droplet; r: radius of the smaller droplet
    if r1 >= r2
        R = r1
        r = r2
    else
        R = r2
        r = r1
    end

    # Check if either of the two colliding droplets has a radius greater than
    # 300 μm -- the formula for coalescence efficiency used here is only valid
    # for droplets with radii < 300 μm.
    if !all([R, r] .<= 300)
        println("Droplets exceeding a radius of 300 microns.")
    end
    # Calculate E1
    q = r/R
    # terminal velocity of falling droplet of radius rad (in cm) in Stokes
    # regime, cm s^-1
    v_T(rad) = 2/9 * (ρ_p - ρ_a) / μ_a * g * rad^2
    v_T_diff = abs(v_T(R * 1e-4)- v_T(r * 1e-4))
    We = ρ_p * r * 1e-4 * v_T_diff^2 / σ_w  # Weber number
    b0 = 0.767
    b1 = -10.14 * 2^(3/2) / (6 * π)
    E1 = b0 + b1 * q^4 * (1 + q) / ((1 + q^2) * (1 + q^3)) * We^(1/2)

    # Calculate E2, which is given as an implicit equation
    a0 = 5.07
    a1 = -5.94
    a2 = 7.27
    a3 = -5.29
    function f!(F, E2)
        F[1] = (a0 + a1 * E2[1] + a2 * E2[1]^2 + a3 * E2[1]^3
                - log(r) - log(R/200))
    end
    function j!(J, E2)
        J[1, 1] = a1 + 2.0 * a2 * E2[1] + 3.0 * a3 * E2[1]^2
    end

    E2 = try
        nlsolve(f!, j!, [0.5]).zero[1]
    catch IsFiniteException
        println("Couldn't solve root-finding problem to determine E2
                for r=$r, R=$R -- setting E2 to 1.")
        E2 = 1.0
    end
    if E2 > 1.0
        E2 = 1.0
    end
    if E2 < 0.0
        E2 = 0.0
    end

    # The coalescence efficiency is the maximum of E1 and E2
    return maximum([E1, E2])
end

function constant_coalescence_efficiency(x, y, val)
    return val
end

end #module Sources.jl
