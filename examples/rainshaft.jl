# 1D Rainshaft model
using LinearAlgebra
using DifferentialEquations
using Plots

using Cloudy.KernelTensors
using Cloudy.Distributions
using Cloudy.Sources

import Cloudy.Distributions: density, nparams


FT = Float64


"""
  sedi_flux(mom_p::Array{Real}, dist::Distribution{Real}, vel::Array{Real})

  - `mom_p` - prognostic moments of particle mass distribution
  - `dist` - particle mass distribution used to calculate diagnostic moments
  _ `coef` - settling velocity coefficient
Returns the sedimentation flux for all moments in `mom_p`.

"""
function sedi_flux(mom_p::Array{FT}, dist::Distribution{FT}, coef::FT) where {FT <: Real}
  s = length(mom_p)
  dist = update_params_from_moments(dist, mom_p)
   
  sedi_int = similar(mom_p)
  for k in 0:s-1
    sedi_int[k+1] = -coef*moment(dist, FT(k)+1/6)
  end
  
  return sedi_int
end


"""
  main(t_min, t_max, z_min, z_max, n_z, mass_min, mass_max, n_mass)

  - `t_min` - minimum time
  - `t_max` - maximum time
  - `z_min` - minimum height
  - `z_max` - maximum height
  - `n_z` - number of vertical levels
  - `mass_min` - minimum droplet mass
  - `mass_max` - maximum droplet mass
  - `n_mass` - number of grid points in mass spass 
Returns the results from a 1D rainshaft model run with collisions and sedimentation. 

"""
function main(t_min=0.0,
              t_max=1.0,
              z_min=0.0,
              z_max=1.0,
              n_z=50,
              mass_min=0.0, 
              mass_max=10.0,
              n_mass=10)
  
  # Physicsal parameters
  coal_kernel = CoalescenceTensor([0.01]) # constant coalescence kernel
  sedi_coef = 0.2 # sedimentation flux coefficient 
  distribution = Exponential(1.0, 1.0) # Size distribution function

  # Initial condition
  d1 = Exponential(100.0, 1.25) # Used to init upper half of domain
  d2 = Exponential(1e-1, 10.0) # Used to init lower half of domain
  n_z_zero = div(n_z, 2)
  n_mom = nparams(d1)
  moments_init = zeros(n_z, n_mom)
  for i in 1:n_z
    if i > n_z_zero
      moments_init[i,:] = [moment(d1, FT(j)) for j in 0:n_mom-1]
    else
      moments_init[i,:] = [moment(d2, FT(j)) for j in 0:n_mom-1]
    end
  end

  # First-order finite difference matrix with no-flux boundary conditions
  # This is used to calculate the flux divergence for the sedimentation flux
  δz = (z_max-z_min) / (n_z - 1)
  height = z_min:δz:z_max
  dd1 = Tridiagonal(zeros(n_z-1), 1.0*ones(n_z), -1.0.*ones(n_z-1))
  dd1[1,1] = -1.0
  dd1[1,2] = 0.0
  dd1[end,end-1] = 0.0
  dd1[end,end] = 1.0
  dd1 /= δz

  # Set up the right hand side of ODE
  function rhs!(dm, m, p, t)
    flux_sedi = Array{FT}(undef, n_z, n_mom)
    for i in 1:n_z
      flux_sedi[i, :] = sedi_flux(m[i,:], distribution, sedi_coef)
    end
    for i in 1:n_mom
      dm[:, i] = dd1 * flux_sedi[:, i]
    end
    for i in 2:n_z
      dm[i, :] += get_int_coalescence(m[i,:], distribution, coal_kernel)
    end
  end
  
  # Solve the ODE 
  prob = ODEProblem(rhs!, moments_init, (t_min, t_max))
  sol = solve(prob)

  # Unpack ODE solution into an array
  time = sol.t
  n_time = length(time)
  moments = Array{FT}(undef, n_time, n_z, n_mom)
  for (i, slice) in enumerate(sol.u)
    moments[i,:,:] = slice
  end

  # Turn time series of moments into time series of density
  δm = mass_max/(n_mass-1)
  mass = mass_min:δm:mass_max
  dists = Array{FT}(undef, n_time, n_z, n_mass)
  for i in 1:n_time
    for j in 1:n_z
      d = update_params_from_moments(d1, moments[i,j,:])
      for k in 1:n_mass
        dists[i,j,k] = density(d, mass[k])
      end
    end
  end

  time, height, mass, moments, dists
end


"""
  plotting(time, height, mass, moments, dist)

  - `time` - time grid
  - `height` - height grid
  - `mass` - mass grid
  - `moments` - moments on time, height, mass grid
  - `dists` - droplet mass distribution on time, height, mass grid 
Generates various plots for the ODE solution.

"""
function plotting(time, height, mass, moments, dists)
  gr()
  plot(
    moments[1, :, 1], 
    height,
    lw=3,
    xaxis="Number density",
    yaxis="Height",
    xlims=(0, 150.0),
    ylims=(height[1], height[end]),
    label="Initial condition",
    title="Zeroth moment"
  )
  plot!(
    moments[end, :, 1], 
    height, 
    lw=3,
    label="Final condition"
  )
  savefig("zeroth_moment.png")

  plot(
    moments[1, :, 2], 
    height,
    lw=3,
    xaxis="Number density",
    yaxis="Height",
    xlims=(0, 150.0),
    ylims=(height[1], height[end]),
    label="Initial condition",
    title="First moment"
  )
  plot!(
    moments[end, :, 2], 
    height, 
    lw=3,
    label="Final condition"
  )
  savefig("first_moment.png")
  
  plot(
    time,
    moments[:, 1, 1], 
    lw=3,
    xaxis="Time",
    yaxis="Moments",
    xlims=(time[1], time[end]),
    ylims=(0.0, 1000.0),
    label="Zeroth moment",
    title="Surface"
  )
  plot!(
    time,
    moments[:, 1, 2], 
    lw=3,
    label="First moment"
  )
  savefig("surface.png")
end

# Run everything!
plotting(main()...)

