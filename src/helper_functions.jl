export get_dist_moment_ind
export get_dist_moments_ind_range

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
