"""
  Microphysical source functions involving interactions between particles
  
  Includes terms such as collisional coalescence
"""
module MultiParticleSources

using QuadGK
using Cloudy.ParticleDistributions

export update_coal_ints!
export initialize_coalescence_data


"""
get_coalescence_integral_moment_qrs(moment_order::FT, kernel::KernelFunction{FT}, pdist::ParticleDistribution{FT}, 
    Q::Array{FT}, R::Array{FT}, S::Array{FT})

Updates the collision-coalescence integrals.
Q: source term to particle k via collisions with smaller particles j
R: sink term to particle k via collisions with particles j
S: source terms to particles j and j+1 through internal collisions
"""
function update_coal_ints!(
    Nmom, kernel_func, pdists, coal_data
)
    # check that all pdists are of the same type
    if typeof(pdists) == Vector{PrimitiveParticleDistribution{Float64}}
        throw(ArgumentError("All particle size distributions must be the same type"))
    end

    for m in 1:Nmom
        coal_data.coal_ints[:,m] .= 0.0
        get_coalescence_integral_moment_qrs!(Float64(m-1), kernel_func, pdists, coal_data.Q, coal_data.R, coal_data.S)
        for k in 1:length(pdists)
            coal_data.coal_ints[k,m] += sum(@views coal_data.Q[k,:])
            coal_data.coal_ints[k,m] -= sum(@views coal_data.R[k,:])
            coal_data.coal_ints[k,m] += coal_data.S[k,1]
            if k > 1
                coal_data.coal_ints[k,m] += coal_data.S[k-1, 2]
            end
        end
    end
end


"""
initialize_coalescence_data(Ndist::FT, dist_moments_init::Array{FT})

Initializes the collision-coalescence integral matrices as zeros.
coal_ints contains all three matrices (Q, R, S) and the overall coal_int summation term
"""
function initialize_coalescence_data(Ndist, Nmom; FT=Float64)
    Q = zeros(FT, Ndist, Ndist)
    R = zeros(FT, Ndist, Ndist)
    S = zeros(FT, Ndist, 2)
    coal_ints = zeros(FT, Ndist, Nmom)
    return (Q=Q, R=R, S=S, coal_ints=coal_ints)
end


function get_coalescence_integral_moment_qrs!(
  moment_order, kernel, pdists, Q, R, S)
  update_Q_coalescence_matrix!(moment_order, kernel, pdists, Q)
  update_R_coalescence_matrix!(moment_order, kernel, pdists, R)
  update_S_coalescence_matrix!(moment_order, kernel, pdists, S)
end


function update_Q_coalescence_matrix!(
    moment_order, kernel, pdists, Q
)
    Ndist = length(pdists)
    for j in 1:Ndist
        for k in 1:Ndist
            if j < k
                Q[j,k] = quadgk(x -> q_integrand_outer(x, j, k, kernel, pdists, moment_order), 0.0, Inf; rtol=1e-8, maxevals=1000)[1]
            else
                Q[j,k] = 0.0
            end
        end
    end
end

function update_R_coalescence_matrix!(
    moment_order, kernel, pdists, R
)
    Ndist = length(pdists)
    for j in 1:Ndist
        for k in 1:Ndist
            R[j,k] = quadgk(x -> r_integrand_outer(x, j, k, kernel, pdists, moment_order), 0.0, Inf; rtol=1e-8, maxevals=1000)[1]
        end
    end
end

function update_S_coalescence_matrix!(
    moment_order, kernel, pdists, S
)
    Ndist = length(pdists)
    for j in 1:Ndist 
        S[j,1] = quadgk(x -> s_integrand1(x, j, kernel, pdists, moment_order), 0.0, Inf; rtol=1e-8, maxevals=1000)[1]
        S[j,2] = quadgk(x -> s_integrand2(x, j, kernel, pdists, moment_order), 0.0, Inf; rtol=1e-8, maxevals=1000)[1]
    end
end

function weighting_fn(x::FT, k::Int64, pdists) where {FT<:Real}
    denom = 0.0
    num = 0.0
    Ndist = length(pdists)
    if k > Ndist
        throw(AssertionError("k out of range"))
    end
    for j=1:Ndist
      denom += normed_density(pdists[j], x) #pdists[j](x) / pdists[j].n
      if j<= k
        num += normed_density(pdists[j], x) #pdists[j](x) / pdists[j].n
      end
    end
    if denom == 0.0
      return 0.0
    else
      return num / denom
    end
end

function q_integrand_inner(x, y, j, k, kernel, pdists)
    if j==k
       throw(AssertionError("q_integrand called on j==k, should call s instead"))
    elseif y > x
        throw(AssertionError("x <= y required in Q integrals"))
    end
    integrand = 0.5 * kernel(x - y, y) * (pdists[j](x-y) * pdists[k](y) + pdists[k](x-y) * pdists[j](y))
    return integrand
end

function q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    if j==k
        throw(AssertionError("q_integrand called on j==k, should call s instead"))
    end
    outer = x.^moment_order * quadgk(yy -> q_integrand_inner(x, yy, j, k, kernel, pdists), 0.0, x; rtol=1e-8, maxevals=1000)[1]
    return outer
end

function r_integrand_inner(x, y, j, k, kernel, pdists)
    integrand = kernel(x, y) * pdists[j](x) * pdists[k](y)
    return integrand
end

function r_integrand_outer(x, j, k, kernel, pdists, moment_order)
    outer = x.^moment_order * quadgk(yy -> r_integrand_inner(x, yy, j, k, kernel, pdists), 0.0, Inf; rtol=1e-8, maxevals=1000)[1]
    return outer
end

function s_integrand_inner(x, k, kernel, pdists, moment_order)
    integrand_inner = y -> 0.5 * kernel(x - y, y) * pdists[k](x-y) * pdists[k](y)
    integrand_outer = x.^moment_order * quadgk(yy -> integrand_inner(yy), 0.0, x; rtol=1e-8, maxevals=1000)[1]
    return integrand_outer
  end
  
function s_integrand1(x, k, kernel, pdists, moment_order)
    integrandj = weighting_fn(x, k, pdists) * s_integrand_inner(x, k, kernel, pdists, moment_order)
    return integrandj
end
  
function s_integrand2(x, k, kernel, pdists, moment_order)
    integrandk = (1 - weighting_fn(x, k, pdists)) * s_integrand_inner(x, k, kernel, pdists, moment_order)
    return integrandk
end

end # module
