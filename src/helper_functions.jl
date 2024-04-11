export get_dist_moment_ind
export get_dist_moments_ind_range
export get_moments_normalizing_factors
export integrate_SimpsonEvenFast

"""
  get_dist_moment_ind(NProgMoms::Vector{Int}, i::Int, m::Int)

  `NProgMoms` - vector containing number of prognostic moments for each distribution
  `i` - distribution index
  `m` - moment index
Returns index of the m'th moment of the i'th distribution in the long vector containing moments of all distributions
"""
function get_dist_moment_ind(NProgMoms::Vector{Int}, i::Int, m::Int)
    @assert 0 < m <= NProgMoms[i]
    return sum(NProgMoms[1:(i - 1)]) + m
end

"""
  get_dist_moments_ind_range(NProgMoms::Vector{Int}, i::Int)

  `NProgMoms` - vector containing number of prognostic moments for each distribution
  `i` - distribution index
Returns range of indecies of the i'th distribution's moments in the long vector containing moments of all distributions
"""
function get_dist_moments_ind_range(NProgMoms::Vector{Int}, i::Int)
    last_ind = sum(NProgMoms[1:(i - 1)])
    return (last_ind + 1):(last_ind + NProgMoms[i])
end

"""
  get_moments_normalizing_factors(NProgMoms::Vector{Int}, n::FT, θ::FT)
  `NProgMoms` - vector containing number of prognostic moments for each distribution
  `norms` - vector containing scale of number and mass/volume of particles
Returns normalizing factors of the vector of moments based on given scales of number and mass/volume of particles
"""
function get_moments_normalizing_factors(NProgMoms::Vector{Int}, norms::Vector{FT}) where {FT <: Real}
    @assert all(norms .> FT(0))
    norm = zeros(FT, sum(NProgMoms))
    n_dist = length(NProgMoms)
    for (i, n_mom) in enumerate(NProgMoms)
        for j in 1:n_mom
            norm[get_dist_moment_ind(NProgMoms, i, j)] = norms[1] * norms[2]^(j - 1)
        end
    end
    return norm
end

"""
  integrate_SimpsonEvenFast(x::AbstractVector, y::AbstractVector)

  `x` - evenly spaced domain x
  `y` - desired function evaluated at the domain points x
Returns the numerical integral, assuming evenly spaced points x. 
This is a reimplementation from NumericalIntegration.jl which has outdated dependencies.
"""
function integrate_SimpsonEvenFast(x::Vector{FT}, y::Vector{FT}) where {FT <: Real}
    length(x) == length(y) || error("x and y vectors must be of the same length!")
    length(x) ≥ 4 || error("vectors must contain at least 4 elements")
    dx = x[2:end] - x[1:(end - 1)]
    minimum(dx) ≈ maximum(dx) || error("x must be evenly spaced")

    @inbounds retval =
        (17 * (y[1] + y[end]) + 59 * (y[2] + y[end - 1]) + 43 * (y[3] + y[end - 2]) + 49 * (y[4] + y[end - 3])) / 48
    @fastmath @inbounds for i in 5:(length(y) - 4)
        retval += y[i]
    end
    @inbounds return (x[2] - x[1]) * retval
end
