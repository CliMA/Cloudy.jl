"""
  multi-moment bulk microphysics implementation of coalescence

  Includes 3 variations on how to solve the SCE:
    - Numerical approach (all numerical integration)
    - Analytical approach (all power-law derived)
    - Hybrid approach (analytical except for the autoconversion integrals)
"""
module Coalescence

using Cloudy.ParticleDistributions
using Cloudy.KernelTensors
using Cloudy.EquationTypes
using QuadGK


# methods that compute source terms from microphysical parameterizations
export get_int_coalescence
export update_coal_ints!
export initialize_coalescence_data

"""
get_int_coalescence(::OneModeCoalStyle, mom_p::Array{Real}, par::Dict, ker::KernelTensor{Real})

  - `mom_p` - prognostic moments of particle mass distribution
  - `par` - ODE parameters, a Dict containing a key ":dist" whose value is a list of ParticleDistribution;
            it is used to calculate the diagnostic moments.
  - `ker` - coalescence kernel tensor
Returns the coalescence integral for all moments in `mom_p`.
"""
function get_int_coalescence(::OneModeCoalStyle, mom_p::Array{FT}, par::Dict, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r
  s = length(mom_p)

  # Need to build diagnostic moments
  update_dist_from_moments!(par[:dist][1], mom_p)
  # Update the distribution that is carried along in par for use in next time steps
  mom_d = Array{FT}(undef, r)
  for k in 0:r-1
    mom_d[k+1] = moment(par[:dist][1], FT(s+k))
  end
  mom = vcat(mom_p, mom_d)

  # only calculate coalescence integral for prognostic moments
  coal_int = similar(mom_p)
  for k in 0:s-1
    if k == 1
      # implies conservation of 1st moment (~mass) under coalescence processes
      temp = 0.0
    end
    if k != 1
      temp = 0.0
      # coalescence integral for kth moment (k = 0 or k > 1)
      # see design document's microphysics for details of derivation
      for a in 0:r
        for b in 0:r
          coef = ker.c[a+1, b+1]
          temp -= coef * mom[a+k+1] * mom[b+1]
          for c in 0:k
            temp += 0.5 * coef * binomial(k, c) * mom[a+c+1] * mom[b+k-c+1]
          end
        end
      end
    end

  coal_int[k+1] = temp
  end

  return coal_int
end

function get_int_coalescence(::TwoModesCoalStyle, mom_p::Array{FT}, par::Dict, ker::KernelTensor{FT}) where {FT <: Real}
  r = ker.r

  n_params = [nparams(par[:dist][i]) for i in 1:2]
  mom_p_ = [mom_p[1:n_params[1]], mom_p[n_params[1]+1:end]]
  s = [length(mom_p_[i]) for i in 1:2]

  # update distributions from moments
  for i in 1:2
    update_dist_from_moments!(par[:dist][i], mom_p_[i])
  end
  
  # Need to build diagnostic moments
  n_mom = maximum(s) + r
  mom = zeros(FT, 2, n_mom)
  for i in 1:2
    for j in 1:n_mom
      if j <= s[i]
        mom[i, j] = mom_p_[i][j]
      else
        mom[i, j] = moment(par[:dist][i], FT(j-1))
      end
    end
  end

  # Compute the integral term with threshold (mode1-mode1 coalescence contributing to both mode 1 and 2)
  int_w_thrsh = zeros(FT, n_mom, n_mom)
  mom_times_mom = zeros(FT, n_mom, n_mom)
  for i in 1:n_mom
    for j in i:n_mom
      mom_times_mom[i, j] = mom[1, i] * mom[1, j]
      tmp = (mom_times_mom[i, j] < eps(FT)) ? FT(0) : moment_source_helper(par[:dist][1], FT(i-1), FT(j-1), par[:x_th])
      int_w_thrsh[i, j] = min(mom_times_mom[i, j], tmp)
      mom_times_mom[j, i] = mom_times_mom[i, j]
      int_w_thrsh[j, i] = int_w_thrsh[i, j]
    end
  end

  # only calculate coalescence integral for prognostic moments
  coal_int = [similar(mom_p_[1]), similar(mom_p_[2])]
  for i in 1:2
    j = (i==1) ? 2 : 1
    for k in 0:s[i]-1
      temp = 0.0

      for a in 0:r
        for b in 0:r
          coef = ker.c[a+1, b+1]
          temp -= coef * mom[i, a+k+1] * mom[i, b+1]
          temp -= coef * mom[i, a+k+1] * mom[j, b+1]
          for c in 0:k
            coef_binomial = coef * binomial(k, c)
            if i == 1
              temp += 0.5 * coef_binomial * int_w_thrsh[a+c+1, b+k-c+1]
            elseif i == 2
              temp += 0.5 * coef_binomial * (mom_times_mom[a+c+1, b+k-c+1] - int_w_thrsh[a+c+1, b+k-c+1])
              temp += 0.5 * coef_binomial * mom[i, a+c+1] * mom[i, b+k-c+1]
              temp += coef_binomial * mom[j, a+c+1] * mom[i, b+k-c+1]
            end
          end
        end
      end

    coal_int[i][k+1] = temp
    end
  end

  return vcat(coal_int[1], coal_int[2])
end


"""
update_coal_ints!(Nmom::FT, kernel_func::KernelFunction{FT}, pdists::Array{ParticleDistribution{FT}},
    coal_data::Dict)

Updates the collision-coalescence integrals.
Nmom: number of prognostic moments per particle distribution
kernel_func: K(x,y) function that determines rate of coalescence based on size of particles x, y
pdists: array of PSD subdistributions
coal_data: Dictionary carried by ODE solver that contains all dynamical parameters, including the 
    coalescence integrals
"""
function update_coal_ints!(::NumericalCoalStyle,
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
