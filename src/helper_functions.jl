export get_dist_moment_ind
export get_dist_moments_ind_range

function get_dist_moment_ind(NProgMoms::Vector{Int}, i::Int, m::Int)
    @assert 0 < m <= NProgMoms[i]
    return sum(NProgMoms[1:(i - 1)]) + m
end

function get_dist_moments_ind_range(NProgMoms::Vector{Int}, i::Int)
    last_ind = sum(NProgMoms[1:(i - 1)])
    return (last_ind + 1):(last_ind + NProgMoms[i])
end
