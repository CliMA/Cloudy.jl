export get_dist_moment_ind
export get_dist_moments_ind_range
export get_moments_normalizing_factors

"""
  get_dist_moment_ind(NProgMoms::Vector{Int}, i::Int, m::Int)

  `NProgMoms` - vector containing number of prognostic moments for each distribution
  `i` - distribution index
  `m` - moment index
Returns index of the m'th moment of the i'th distribution in the long vector containing moments of all distributions
"""
function get_dist_moment_ind(NProgMoms::NTuple, i::Int, m::Int)
    if ~(0 < m <= NProgMoms[i])
        error("moment index must be positive integer and equal or smaller than the dist number of prognostic moments!!!")
    end
    return i == 1 ? m : sum(NProgMoms[1:(i - 1)]) + m
end

"""
  get_dist_moments_ind_range(NProgMoms::Vector{Int}, i::Int)

  `NProgMoms` - vector containing number of prognostic moments for each distribution
  `i` - distribution index
Returns range of indecies of the i'th distribution's moments in the long vector containing moments of all distributions
"""
function get_dist_moments_ind_range(NProgMoms::NTuple, i::Int)
    last_ind = sum(NProgMoms[1:(i - 1)])
    return (last_ind + 1):(last_ind + NProgMoms[i])
end

"""
  get_moments_normalizing_factors(NProgMoms::Vector{Int}, n::FT, θ::FT)
  `NProgMoms` - vector containing number of prognostic moments for each distribution
  `norms` - vector containing scale of number and mass/volume of particles
Returns normalizing factors of the vector of moments based on given scales of number and mass/volume of particles
"""
function get_moments_normalizing_factors(NProgMoms::AbstractArray{Int}, norms::Tuple{FT, FT}) where {FT <: Real}
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

rflatten(tup::Tuple) = (rflatten(Base.first(tup))..., rflatten(Base.tail(tup))...)
rflatten(tup::Tuple{<:Tuple}) = rflatten(Base.first(tup))
rflatten(arg) = arg
rflatten(tup::Tuple{}) = ()
