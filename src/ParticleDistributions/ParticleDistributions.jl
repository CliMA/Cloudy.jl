"""
  particle mass distribution module

Particle mass distribution functions for microphysical process modeling:
  - computation of moments of distributions
  - computation of densities of distributions
  - creating distributions given a set of parameters
  - creating distributions given a set of moments
"""
module ParticleDistributions

using SpecialFunctions: gamma, gamma_inc, gamma_inc_inv
using DocStringExtensions
using QuadGK
using StaticArrays

# particle mass distributions available for microphysics
export AbstractParticleDistribution
export PrimitiveParticleDistribution
export ExponentialPrimitiveParticleDistribution
export GammaPrimitiveParticleDistribution
export MonodispersePrimitiveParticleDistribution
export LognormalPrimitiveParticleDistribution

# methods that query particle mass distributions
export moment
export get_moments
export density
export normed_density
export nparams
export update_dist_from_moments
export moment_source_helper
export get_standard_N_q
export compute_thresholds

"""
  AbstractParticleDistribution{FT}

A particle mass distribution function, which can be initialized
for various subtypes of assumed shapes in the microphysics parameterization.
"""
abstract type AbstractParticleDistribution{FT} end


"""
  PrimitiveParticleDistribution{FT}

A particle mass distribution that has support on the positive real
axis and analytic expressions for moments and partial moments.
"""
abstract type PrimitiveParticleDistribution{FT} <: AbstractParticleDistribution{FT} end

Base.eltype(::PrimitiveParticleDistribution{FT}) where {FT} = FT

"""
  ExponentialPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents particle mass distribution function of exponential shape.

# Constructors
  ExponentialPrimitiveParticleDistribution(n::Real, θ::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ExponentialPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
    "normalization constant (e.g., droplet number concentration)"
    n::FT
    "scale parameter"
    θ::FT

    function ExponentialPrimitiveParticleDistribution(n::FT, θ::FT) where {FT <: Real}
        if n < 0 || θ <= 0
            error("n needs to be nonnegative. θ needs to be positive.")
        end

        new{FT}(n, θ)
    end
end


"""
  GammaPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents particle mass distribution function of gamma shape.

# Constructors
  GammaPrimitiveParticleDistribution(n::Real, θ::Real, k::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GammaPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
    "normalization constant (e.g., droplet number concentration)"
    n::FT
    "scale parameter"
    θ::FT
    "shape parameter"
    k::FT

    function GammaPrimitiveParticleDistribution(n::FT, θ::FT, k::FT) where {FT <: Real}
        if n < 0 || θ <= 0 || k <= 0
            error("n needs to be nonnegative. θ and k need to be positive.")
        end
        new{FT}(n, θ, k)
    end
end

"""
  MonodispersePrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents monodisperse particle size distribution function.

# Constructors
  MonodispersePrimitiveParticleSizeDistribution(n::Real, m::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct MonodispersePrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
    "normalization constant (e.g., droplet number concentration)"
    n::FT
    "particle diameter"
    θ::FT

    function MonodispersePrimitiveParticleDistribution(n::FT, θ::FT) where {FT <: Real}
        if n < 0 || θ <= 0
            error("n needs to be nonnegative. θ needs to be positive.")
        end
        new{FT}(n, θ)
    end
end

"""
  LognormalPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}

Represents lognormal particle size distribution function.

# Constructors
  LognormalPrimitiveParticleSizeDistribution(n::Real, μ::Real, σ::Real)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct LognormalPrimitiveParticleDistribution{FT} <: PrimitiveParticleDistribution{FT}
    "normalization constant (e.g., droplet number concentration)"
    n::FT
    "logarithmic mean size"
    μ::FT
    "logarithmic standard deviation"
    σ::FT

    function LognormalPrimitiveParticleDistribution(n::FT, μ::FT, σ::FT) where {FT <: Real}
        if n < 0 || σ <= 0
            error("n needs to be nonnegative. σ needs to be positive.")
        end
        new{FT}(n, μ, σ)
    end
end

"""
  (pdist::ParticleDistribution{FT}(x::FT)

  - `x` - is an array of points to evaluate the density of `pdist` at
Returns the particle mass density evaluated at `x`.
"""
function (pdist::AbstractParticleDistribution{FT})(x::FT) where {FT <: Real}
    return density(pdist, x)
end

"""
  moment_func(dist)

  `dist` - particle mass distribution function
Returns a function that computes the moments of `dist`.
"""
function moment_func(dist::ExponentialPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * θ^q * Γ(q+1)
    function f(q)
        dist.n * dist.θ^q * gamma(q + FT(1))
    end
    return f
end

function moment_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * θ^q * Γ(q+k) / Γ(k)
    function f(q)
        dist.n * dist.θ^q * gamma(q + dist.k) / gamma(dist.k)
    end
    return f
end

function moment_func(dist::MonodispersePrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * θ^(q)
    function f(q)
        dist.n * dist.θ^q
    end
    return f
end

function moment_func(dist::LognormalPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * exp(q * μ + 1/2 * q^2 * σ^2)
    function f(q)
        dist.n * exp(q * dist.μ + q^2 * dist.σ^2 / 2)
    end
    return f
end

"""
  moment(dist, q)

  - `dist` - distribution of which the partial moment `q` is taken
  - `q` - is a potentially real-valued order of the moment
Returns the q-th moment of a particle mass distribution function.
"""
function moment(dist::AbstractParticleDistribution{FT}, q::FT) where {FT <: Real}
    moment_func(dist)(q)
end

"""
  partial_moment_func(dist)

  `dist` - particle mass distribution function
Returns a function that computes the moments of `dist`, integrated up to some threhold
"""
function partial_moment_func(dist::ExponentialPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * θ^q * Γ(q+1)
    function f(q, x_threshold)
        dist.n * dist.θ^q * gamma_inc(q + FT(1), x_threshold / dist.θ)[1]
    end
    return f
end

function partial_moment_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * θ^q / Γ(k) * (Γ(k+q) - Γ(k+q, x/θ))
    function f(q, x_threshold)
        dist.n * dist.θ^q * gamma_inc(q + dist.k, x_threshold / dist.θ)[1] / gamma(dist.k)
    end
    return f
end

function partial_moment_func(dist::MonodispersePrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * θ^(q)
    function f(q, x_threshold)
        if x_threshold < dist.θ
            FT(0)
        else
            dist.n * dist.θ^q
        end
    end
    return f
end

function partial_moment_func(dist::LognormalPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # moment_of_dist = n * exp(q * μ + 1/2 * q^2 * σ^2)
    function f(q, x_threshold)
        quadgk(x -> x^q * dist(x), FT(0.0), FT(x_threshold))[1]
    end
    return f
end

"""
  partial_moment(dist, q, x_threshold)

  - `dist` - distribution of which the partial moment `q` is taken
  - `q` - is a potentially real-valued order of the moment
  - `x_threshold` - is the integration limit for the moment computation
Returns the q-th moment of a particle mass distribution function integrated up to some threshold size.
"""
function partial_moment(dist::AbstractParticleDistribution{FT}, q::FT, x_threshold::FT) where {FT <: Real}
    return partial_moment_func(dist)(q, x_threshold)
end

# TODO: Move to examples
"""
    get_moments(pdist::GammaParticleDistribution{FT})
Returns the first P (0, 1, 2) moments of the distribution where P is the innate
numer of prognostic moments
"""
function get_moments(pdist::GammaPrimitiveParticleDistribution{FT}) where {FT <: Real}
    return [pdist.n, pdist.n * pdist.k * pdist.θ, pdist.n * pdist.k * (pdist.k + 1) * pdist.θ^2]
end

function get_moments(pdist::LognormalPrimitiveParticleDistribution{FT}) where {FT <: Real}
    return [pdist.n, pdist.n * exp(pdist.μ + pdist.σ^2 / 2), pdist.n * exp(2.0 * pdist.μ + 2.0 * pdist.σ^2)]
end

function get_moments(pdist::ExponentialPrimitiveParticleDistribution{FT}) where {FT <: Real}
    return [pdist.n, pdist.n * pdist.θ]
end

function get_moments(pdist::MonodispersePrimitiveParticleDistribution{FT}) where {FT <: Real}
    return [pdist.n, pdist.n * pdist.θ]
end

"""
  density_func(dist)

  - `dist` - is a particle mass distribution
Returns the particle mass density function.
"""
function density_func(dist::ExponentialPrimitiveParticleDistribution{FT}) where {FT <: Real}
    function f(x)
        dist.n ./ dist.θ .* exp.(-x ./ dist.θ)
    end
    return f
end

function density_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # density = n / θ^k / Γ(k) * x^(k-1) * exp(-x/θ)
    function f(x)
        dist.n .* x .^ (dist.k .- 1) ./ dist.θ .^ dist.k ./ gamma.(dist.k) .* exp.(-x ./ dist.θ)
    end
    return f
end

function density_func(dist::LognormalPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # density = n * 1 / (x σ √2π) exp((-ln(x) - μ)^2 / 2σ^2 )
    function f(x)
        dist.n .* exp.(-((log.(x) - dist.μ) .^ 2 ./ (2 * dist.σ^2))) ./ (x .* dist.σ .* sqrt(2 * π))
    end
    return f
end

function density_func(dist::MonodispersePrimitiveParticleDistribution{FT}) where {FT <: Real}
    # density = n δ(θ); here we return a rectangular pulse only for visualizations: n/(2Δx) * [H(x-θ+Δx) - H(x-θ-Δx)]
    # where 2Δx = 2θ/10 is the pulse width and H represents the heaviside function
    function f(x)
        return (abs(x - dist.θ) < dist.θ / 10.0) ? dist.n / (2 * dist.θ / 10.0) : FT(0)
    end
    return f
end

"""
  normed_density_func(dist)

  - `dist` - is a particle mass distribution
Returns the normalized particle mass density function.
"""
function normed_density_func(dist::ExponentialPrimitiveParticleDistribution{FT}) where {FT <: Real}
    function f(x)
        1 ./ dist.θ .* exp.(-x ./ dist.θ)
    end
    return f
end

function normed_density_func(dist::GammaPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # density = n / θ^k / Γ(k) * x^(k-1) * exp(-x/θ)
    function f(x)
        x .^ (dist.k .- 1) ./ dist.θ .^ dist.k ./ gamma.(dist.k) .* exp.(-x ./ dist.θ)
    end
    return f
end

function normed_density_func(dist::LognormalPrimitiveParticleDistribution{FT}) where {FT <: Real}
    # density = n * 1 / (x σ √2π) exp((-ln(x) - μ)^2 / 2σ^2 )
    function f(x)
        exp.(-(log.(x) - dist.μ) .^ 2 ./ (2 * dist.σ^2)) ./ (x .* dist.σ .* sqrt(2 * π))
    end
    return f
end

"""
  density(dist, x)

  - `dist` - is a particle mass distribution
  - `x` - is a point to evaluate the density of `dist` at
Returns the particle mass density evaluated at point `x`.
"""
function density(dist::AbstractParticleDistribution{FT}, x::FT) where {FT <: Real}
    if any(x .< zero(x))
        error("Density can only be evaluated at nonnegative values.")
    end
    density_func(dist)(x)
end

"""
  normed_density(dist, x)

  - `dist` - is a particle mass distribution
  - `x` - is a point to evaluate the density of `dist` at
Returns the particle normalized mass density evaluated at point `x`.
"""
function normed_density(dist::AbstractParticleDistribution{FT}, x::FT) where {FT <: Real}
    if any(x .< zero(x))
        error("Density can only be evaluated at nonnegative values.")
    end
    normed_density_func(dist)(x)
end


"""
  nparams(dist)

  - `dist` - is a particle mass distribution
Returns the number of settable parameters of dist.
"""
function nparams(dist::PrimitiveParticleDistribution{FT}) where {FT <: Real}
    length(propertynames(dist))
end

"""
  check_moment_consistency(m::NTuple{N, T})

  - `m` - is an array of moments

Checks if moments are nonnegative and whether even-ordered central moments implied
by moments vector are all positive.
"""
function check_moment_consistency(m::NTuple{N, FT}) where {N, FT <: Real}
    # check if  moments are nonnegative
    any(m .< 0.0) && error("all moments need to be nonnegative.")

    # check if even-ordered central moments are positive (e.g., variance, etc.)
    # non-positivity  would be inconsistent with a well-defined distribution.
    for order in 2:2:(length(m) - 1)
        cm = mapreduce(+, 0:order) do i
            binomial(order, i) * (-1)^i * (m[2] / m[1])^i * (m[order - i + 1] / m[1])
        end
        cm < 0.0 && error("order central moment needs to be nonnegative.")
    end
end

"""
    update_dist_from_moments(pdist::GammaPrimitiveParticleDistribution{FT}, moments::Tuple{FT, FT, FT})

Returns a new gamma distribution given the first three moments
"""
function update_dist_from_moments(
    pdist::GammaPrimitiveParticleDistribution{FT},
    moments::Tuple{FT, FT, FT};
    param_range = (; :k => (eps(FT), 10.0)),
) where {FT <: Real}
    if moments[1] > eps(FT) && moments[2] > eps(FT)
        n = moments[1]
        k = max(
            param_range.k[1],
            min(param_range.k[2], (moments[2] / moments[1]) / (moments[3] / moments[2] - moments[2] / moments[1])),
        )
        θ = moments[2] / moments[1] / k
        return GammaPrimitiveParticleDistribution(n, θ, k)
    else # make sure θ and k are physical
        GammaPrimitiveParticleDistribution(FT(0), FT(1), FT(1))
    end
end

"""
    update_dist_from_moments(pdist::LognormalPrimitiveParticleDistribution{FT}, moments::Tuple{FT, FT})

Returns a new lognormal distribution given the first three moments
"""
function update_dist_from_moments(
    pdist::LognormalPrimitiveParticleDistribution{FT},
    moments::Tuple{FT, FT, FT};
    param_range = (; :μ => (-Inf, Inf), :σ => (eps(FT), Inf)),
) where {FT <: Real}
    if moments[1] > eps(FT) && moments[2] > eps(FT) && moments[3] > eps(FT)
        μ = max(param_range.μ[1], min(param_range.μ[2], log(moments[2]^2 / moments[1]^(3 / 2) / moments[3]^(1 / 2))))
        σ = max(param_range.σ[1], min(param_range.σ[2], sqrt(log(moments[1] * moments[3] / moments[2]^2))))
        n = moments[2] / exp(μ + 1 / 2 * σ^2)
        return LognormalPrimitiveParticleDistribution(n, μ, σ)
    else # make sure μ and σ are physical
        return LognormalPrimitiveParticleDistribution(FT(0), FT(1), FT(1))
    end
end

"""
    update_dist_from_moments(pdist::ExponentialPrimitiveParticleDistribution{FT}, moments::Tuple{FT, FT})

Returns a new exponential distribution given the first two moments
"""
function update_dist_from_moments(
    pdist::ExponentialPrimitiveParticleDistribution{FT},
    moments::Tuple{FT, FT},
) where {FT <: Real}
    if moments[1] > eps(FT) && moments[2] > eps(FT)
        n = moments[1]
        θ = moments[2] / moments[1]
        return ExponentialPrimitiveParticleDistribution(n, θ)
    else # make sure θ is physical
        return ExponentialPrimitiveParticleDistribution(FT(0), FT(1))
    end
end

"""
    update_dist_from_moments(pdist::MonodispersePrimitiveParticleDistribution{FT}, moments::Tuple{FT, FT})

Returns a new monodisperse distribution given the first two moments
"""
function update_dist_from_moments(
    pdist::MonodispersePrimitiveParticleDistribution{FT},
    moments::Tuple{FT, FT},
) where {FT <: Real}
    if moments[1] > eps(FT) && moments[2] > eps(FT)
        n = moments[1]
        θ = moments[2] / moments[1]
        return MonodispersePrimitiveParticleDistribution(n, θ)
    else # make sure θ is physical
        return MonodispersePrimitiveParticleDistribution(FT(0), FT(1))
    end
end

"""
  moment_source_helper(dist, p1, p2, x_threshold)

  - `dist` - AbstractParticleDistribution
  - `p1` - power of particle mass
  - `p2` - power of particle mass
  - `x_threshold`- particle mass threshold

Returns ∫_0^x_threshold ∫_0^(x_threshold-x') x^p1 x'^p2 f(x) f(x') dx dx' for computations of the source of moments of the distribution below
the given threshold x_threshold. For MonodispersePrimitiveParticleDistribution The integral can be computed analytically: 
∫_0^x_threshold ∫_0^(x_threshold-x') x^p1 x'^p2 f(x) f(x') dx dx = n^2 * θ^(p1+p2) if θ < x_threshold/2, and equals zero otherwise. For
ExponentialPrimitiveParticleDistribution and GammaPrimitiveParticleDistribution the two-dimensional integral reduces to a one-dimensional 
integral over incomplete gamma functions.
"""
function moment_source_helper(
    dist::MonodispersePrimitiveParticleDistribution{FT},
    p1::FT,
    p2::FT,
    x_threshold::FT,
) where {FT <: Real}
    return (dist.θ < x_threshold / 2) ? dist.n^2 * dist.θ^(p1 + p2) : FT(0)
end

logx(x_min::FT, j::Int, dx::FT) where {FT} = x_min + (j - 1) * dx
function moment_source_helper(
    dist::ExponentialPrimitiveParticleDistribution{FT},
    p1::FT,
    p2::FT,
    x_threshold::FT,
    n_bins_per_log_unit::Int = 15,
) where {FT <: Real}
    (; n, θ) = dist
    γ_p2k = gamma(p2 + 1)

    f(x) = x^p1 * exp(-x / θ) * gamma_inc(p2 + 1, (x_threshold - x) / θ)[1] * γ_p2k

    x_lowerbound = FT(min(FT(1e-5), FT(1e-5) * x_threshold))
    n_bins = floor(Int, n_bins_per_log_unit * log10(x_threshold / x_lowerbound))
    x_min = log(x_lowerbound)
    dx = (log(x_threshold) - log(x_lowerbound)) / n_bins
    y_func(j) = j <= n_bins ? exp(logx(x_min, j, dx)) * f(exp(logx(x_min, j, dx))) : zero(typeof(dx))
    return n^2 * θ^(p2 - 1) * integrate_SimpsonEvenFast(n_bins, dx, y_func)
end

function moment_source_helper(
    dist::GammaPrimitiveParticleDistribution{FT},
    p1::FT,
    p2::FT,
    x_threshold::FT,
    n_bins_per_log_unit::Int = 15,
) where {FT <: Real}
    (; n, θ, k) = dist
    γ_k = gamma(k)
    γ_p2k = gamma(p2 + k)

    # Note that gamma_inc is the source of evil allocations this time
    f(x) = x^(p1 + k - 1) * exp(-x / θ) * gamma_inc(p2 + k, (x_threshold - x) / θ)[1] * γ_p2k

    x_lowerbound = FT(min(FT(1e-5), FT(1e-5) * x_threshold))
    n_bins = floor(Int, n_bins_per_log_unit * log10(x_threshold / x_lowerbound))
    x_min = log(x_lowerbound)
    dx = (log(x_threshold) - log(x_lowerbound)) / n_bins
    y_func(j) = j <= n_bins ? exp(logx(x_min, j, dx)) * f(exp(logx(x_min, j, dx))) : zero(typeof(dx))
    return n^2 * θ^(p2 - k) / γ_k^2 * integrate_SimpsonEvenFast(n_bins, dx, y_func)
end

function moment_source_helper(
    dist::LognormalPrimitiveParticleDistribution{FT},
    p1::FT,
    p2::FT,
    x_threshold::FT;
) where {FT <: Real}

    g(x, y) = x .^ (p1) .* y .^ (p2) .* density_func(dist).(x) .* density_func(dist).(y)
    f(y) = quadgk(xx -> g(xx, y), FT(0), x_threshold - y)[1]

    return quadgk(f, FT(0), x_threshold)[1]
end

"""
  get_standard_N_q(pdists; size_cutoff, rtol)
  `pdists` - tuple of particle size distributions
  `size_cutoff` - size distinguishing between cloud and rain
Returns a named tuple (N_liq, N_rai, M_liq, M_rai) of the number and mass densities of liquid (cloud) and rain computed
from the current pdists given a size cutoff
"""
function get_standard_N_q(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, size_cutoff = 1e-6) where {FT, N}
    N_liq = get_standard_N_liq(pdists, size_cutoff)
    M_liq = get_standard_M_liq(pdists, size_cutoff)
    N_rai = get_standard_N_rai(pdists, size_cutoff)
    M_rai = get_standard_M_rai(pdists, size_cutoff)
    return (; N_liq, N_rai, M_liq, M_rai)
end

function get_standard_N_liq(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, size_cutoff = 1e-6) where {FT, N}
    return mapreduce(j -> partial_moment(pdists[j], FT(0), size_cutoff), +, ntuple(identity, N))
end

function get_standard_N_rai(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, size_cutoff = 1e-6) where {FT, N}
    return mapreduce(
        j -> moment(pdists[j], FT(0)) - partial_moment(pdists[j], FT(0), size_cutoff),
        +,
        ntuple(identity, N),
    )
end

function get_standard_M_liq(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, size_cutoff = 1e-6) where {FT, N}
    return mapreduce(j -> partial_moment(pdists[j], FT(1), size_cutoff), +, ntuple(identity, N))
end

function get_standard_M_rai(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, size_cutoff = 1e-6) where {FT, N}
    return mapreduce(
        j -> moment(pdists[j], FT(1)) - partial_moment(pdists[j], FT(1), size_cutoff),
        +,
        ntuple(identity, N),
    )
end

"""
  integrate_SimpsonEvenFast(x::AbstractVector, y::AbstractVector)

  `n_bins` - number of evaluation points
  `dx` - spacing of domain x
  `y` - desired function evaluated at the domain points x
Returns the numerical integral, assuming evenly spaced points x. 
This is a reimplementation from NumericalIntegration.jl which has outdated dependencies.
"""
function integrate_SimpsonEvenFast(n_bins::Int, dx::FT, y::F) where {FT <: Real, F}
    n_bins ≥ 3 || error("n_bins must be at least 3")
    e = n_bins + 1
    retval =
        sum(j -> y(j), 5:(n_bins - 3); init = FT(0)) +
        (17 * (y(1) + y(e)) + 59 * (y(2) + y(e - 1)) + 43 * (y(3) + y(e - 2)) + 49 * (y(4) + y(e - 3))) / 48
    return dx * retval
end


"""
  compute_threshold(pdists; percentile)
  `pdists` - tuple of particle size distributions
  `percentile` - mass percentile
Returns a tuple of new integral thresholds, one for each pdist, computed using the given percentile
"""
function compute_thresholds(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, percentile::FT = 0.97) where {FT, N}
    return ntuple(N) do i
        if i == N 
            Inf
        else
            FT(compute_threshold(pdists[i], percentile))
        end
    end
end

function compute_thresholds(pdists::NTuple{N, PrimitiveParticleDistribution{FT}}, percentiles::NTuple{N, FT}) where {FT, N}
    return ntuple(N) do i
        if i == N 
            Inf
        else
            FT(compute_threshold(pdists[i], percentiles[i]))
        end
    end
end

function compute_threshold(pdist::ExponentialPrimitiveParticleDistribution{FT}, percentile::FT = 0.97, minx::FT = 1e-18) where {FT <: Real}
    return max(-pdist.θ * log(1 - percentile), minx)
end

function compute_threshold(pdist::GammaPrimitiveParticleDistribution{FT}, percentile::FT = 0.97, minx::FT = 1e-18) where {FT <: Real}
    return max(pdist.θ * gamma_inc_inv(pdist.k, percentile, 1 - percentile), minx)
end


end #module ParticleDistributions.jl
