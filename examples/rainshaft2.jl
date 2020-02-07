#1D Rainshaft model
using LinearAlgebra
using DifferentialEquations
using Plots

using Cloudy.KernelTensors
using Cloudy.Distributions
using Cloudy.Sources

import Cloudy.Distributions: density, nparams


FT = Float64


function sedi_flux(mom)
  """Physical parameters"""
  cd = 0.55 # drag coefficient for droplets
  grav = 9.81 # gravitational constant
  ρ_l = 1e3 # liquid water density
  ρ_a = 1.225 # air density
  
  sedi_coef = (8*grav/3/cd*(ρ_l/ρ_a-1)*(3/4/π/ρ_l)^(1/3))^(1/2) * 0.01
  flux = mom -> mom
  dflux = mom -> zeros(size(mom)) 
end


"""
  initial_condition(z, nmom)

  - `z` - height coordinate 
  - `nmom` - number of moments
Returns array of initial values.

"""
function initial_condition(z, nmom)
  zmax = findmax(z)[1]
  zs1 = 2 .* (z .- 0.5 .* zmax) ./ zmax .* 500.0
  zs2 = 2 .* (z .- 0.75 .* zmax) ./ zmax .* 500.0
  at1 = 0.5 .* (1 .+ atan.(zs1) .* 2 ./ pi)
  at2 = 0.5 .* (1 .+ atan.(zs2) .* 2 ./ pi)
  at = 1e-6 .+ at1 .- at2

  ic = zeros(length(z), nmom)
  ic[:, 1] = 1.0e3 .* at
  ic[:, 2] = 1.0e3 .* at

  return ic 
end


"""
  weno(m, flux, dflux, source, dz)
  
  - `m` - number of layers 
  - `flux` - sedimentation flux function
  - `dflux` - derivative of sedimentation flux function
  - `source` - sources and sinks function of RHS
  - `dz` - vertical grid spacing
  Returns WENO (weighted essentially non-oscillatory) scheme rhs.
  This scheme is strongly mass conserving and allows for sharp gradients in the fields.

"""
function weno(m, flux, dflux, source, dz)
  # Lax-Friedrich flux splitting
  a = findmax(abs.(dflux(m)), dims=1)[1]
  v = 0.5 .* (flux(m) .+ a .* m) 
  u = 0.5 .* (flux(m) .- a .* m) 
  u = circshift(u, [-1, 0])

  # Right Flux
  # Choose the positive fluxes, 'v', to compute the right cell boundary flux:
  # $u_{i+1/2}^{-}$
  vm = circshift(v, [1, 0])
  vp = circshift(v, [-1, 0])

  # Polynomials
  p0n = 0.5 .* (-vm .+ 3 .* v)
  p1n = 0.5 .* (v  .+ vp)
  
  # Smooth Indicators (Beta factors)
  B0n = (vm .- v).^2 
  B1n = (v .- vp).^2
  
  # Constants
  d0n = 1/3 
  d1n = 2/3 
  epsilon = 1E-6
  
  # Alpha weights 
  alpha0n = d0n ./ (epsilon .+ B0n).^2
  alpha1n = d1n ./ (epsilon .+ B1n).^2
  alphasumn = alpha0n .+ alpha1n
  
  # ENO stencils weigths
  m0n = alpha0n ./ alphasumn
  m1n = alpha1n ./ alphasumn
  
  # Numerical Flux at cell boundary, $u_{i+1/2}^{-}$
  hn = m0n .* p0n .+ m1n .* p1n

  # Left Flux 
  # Choose the negative fluxes, 'u', to compute the left cell boundary flux:
  # $u_{i-1/2}^{+}$ 
  um  = circshift(u, [1, 0])
  up  = circshift(u, [-1, 0])

  # Polynomials
  p0p = 0.5 .* (um .+ u)
  p1p = 0.5 .* (3 .* u .- up)
  
  # Smooth Indicators (Beta factors)
  B0p = (um .- u).^2 
  B1p = (u .- up).^2
  
  # Constants
  d0p = 2/3 
  d1p = 1/3 
  epsilon = 1E-6
  
  # Alpha weights 
  alpha0p = d0p ./ (epsilon .+ B0p).^2
  alpha1p = d1p ./ (epsilon .+ B1p).^2
  alphasump = alpha0p .+ alpha1p
  
  # ENO stencils weigths
  m0p = alpha0p ./ alphasump
  m1p = alpha1p ./ alphasump
  
  # Numerical Flux at cell boundary, $u_{i-1/2}^{+}$
  hp = m0p .* p0p .+ m1p .* p1p

  # Positive and negative flux componetns
  flux_right = hp .+ hn
  flux_left = circshift(hp, [1, 0]) .+ circshift(hn, [1, 0])
  flux_right[end, :] = zeros(size(m)[2])
  flux_left[1, :] = zeros(size(m)[2]) 

  # Compute finite volume rhs term, df/dz.
  rhs = (flux_right .- flux_left) ./ dz - source(m)

  return rhs
end


"""

"""
function weno2(m, flux, dflux, source, dz)
  # Parameters
  n = size(m)[1]	
  i = 2:(n+1)
  
  # Ghost cell values
  lower = m[[2], :]
  upper = m[[end], :]
  upper2 = m[[end], :]

  # Extended field
  u = [lower; lower; m[i[1:n-1], :]; upper; upper2]
  
  # Lax-Friedrichs (LF) flux splitting
  a = findmax(abs.(dflux(u)), dims=1)[1]
  fp = 0.5 * (flux(u) .+ a .* u) 
  fm = 0.5 * (flux(u) .- a .* u)
  fp[end, :] -= flux(u[end, :])
  fp[end-1, :] -= flux(u[end-1, :])
  fp[1, :] -= flux(u[1, :])

  alpha1 = zeros(size(u)) 
  alpha2 = zeros(size(u)) 
  r_minus = zeros(size(u)) 
  r_plus = zeros(size(u)) 
  
  # WENO3 "right" flux reconstruction
  epsilon = 1e-16
  alpha1[i, :] = (1/3) ./ (epsilon .+ (fp[i, :] .- fp[i.-1, :]).^2).^2   
  alpha2[i, :] = (2/3) ./ (epsilon .+ (fp[i.+1, :] .- fp[i, :]).^2).^2
  alphasum = alpha1 .+ alpha2
  omega1 = alpha1 ./ alphasum 
  omega2 = alpha2 ./ alphasum 
  r_plus[i, :] = omega1[i, :] .* (3/2 .* fp[i, :] .- 1/2 .* fp[i.-1, :]) .+ omega2[i, :] .* (1/2 .* fp[i, :] .+ 1/2 .* fp[i.+1, :])
  
  # WENO3 "left" flux reconstruction
  alpha1[i, :] = (1/3) ./ (epsilon .+ (fm[i.+2] .- fm[i.+1, :]).^2).^2   
  alpha2[i, :] = (2/3) ./ (epsilon .+ (fm[i.+1, :] .- fm[i, :]).^2).^2   
  alphasum = alpha1 .+ alpha2
  omega1 = alpha1 ./ alphasum 
  omega2 = alpha2 ./ alphasum 
  r_minus[i, :] = omega1[i, :] .* (3/2 .* fm[i.+1, :] .- 1/2 .* fm[i.+2]) .+ omega2[i, :] .* (1/2 .* fm[i, :] .+ 1/2 .* fm[i.+1, :])
  
  # Compute Residual
  rhs_right = r_plus[i, :] .+ r_minus[i, :]
  rhs_left = r_plus[i.-1, :] .+ r_minus[i.-1, :]
  #rhs_left[1, :] = rhs_left[2, :] 
  rhs_left[1, :] = zeros(1, size(m)[2]) 

  rhs = (rhs_right .- rhs_left) ./ dz .- source(m)

  return rhs
end


"""
  main(nz, zmax, tend, cfl)

  - `nz` - number of vertical levels
  - `zmax` - maximum height
  - `tend` - maximum time
  - `cfl` - cfl_max number
Returns the results from a 1D rainshaft model run with collisions and sedimentation. 

"""
function main(nmom=2, nz=50, zmax=2000.0, tend=7200.0, cfl=5.0)
  # Flux and source terms
  coef = -0.5
  flux = m -> coef * m
  dflux = m -> coef * ones(size(m))
  source = m -> zeros(size(m))

  # Build discrete domain
  a = 0.0
  b = zmax
  dz = (b-a) / nz 
  height = a+dz/2:dz:b
  
  # Initial condition
  m0 = initial_condition(height, nmom)
  m = m0
  moments = reshape(m, size(m)..., 1)

  # Rhs function
  get_rhs = weno2 
 
  # Solver loop
  t = 0.0
  it = 0
  time = [t]

  # Let's keep track of the integral of the field to track conservation
  column_mass = [sum(m[2:end, 2] .* dz)]
  surface_mass = [m[1, 2] .* dz]
  minimum = [findmin(m)[1]]
  
  while t < tend
    # Update/correct time step
    dt = cfl * dz / findmax(abs.(m[2:end-1, :]))[1] 
    if t + dt > tend
      dt = tend - t 
    end
    
    # Update iteration counter
    it = it + 1
    t = t + dt
    append!(time, t)

    # RK step
    mo = m
    
    # 1st stage
    df = get_rhs(m, flux, dflux, source, dz)
    m = mo .- dt .* df
    
    # 2nd Stage
    df = get_rhs(m, flux, dflux, source, dz)
    m = 0.75 .* mo .+ 0.25 .* (m .- dt .* df)

    # 3rd stage
    df = get_rhs(m, flux, dflux, source, dz)
    m = (mo .+ 2 .* (m .- dt .* df)) ./ 3.0

    append!(column_mass, sum(m[2:end, 2] .* dz))
    append!(surface_mass, m[1, 2] .* dz)
    append!(minimum, findmin(m)[1])
    moments = cat(moments, reshape(m, size(m)..., 1), dims=3)
  end
  minimum = findmin(minimum)[1]

  # Report minimum during integration
  println("Minimum value of moments was: $minimum.")

  return time, height, moments, column_mass, surface_mass
end


"""
  plotting(time, height, moments, column_mass, surface_mass)

  - `time` - time grid
  - `height` - height grid
  - `moments` - moments on time, height grid
  - `column_mass` - column integrated water mass 
  - `surface_mass` - surface integrated water mass 
Generates various plots for the ODE solution.

"""
function plotting(time, height, moments, column_mass, surface_mass)
  gr()
  plot(
    moments[2:end, 1, 1], 
    height[2:end],
    lw=3,
    xaxis="Number density [1/cm^3]",
    yaxis="Height [m]",
    xlims=(0, 1500.0),
    ylims=(height[1], height[end]),
    label="Initial condition",
    title="Zeroth moment - particle number"
  )
  plot!(
    moments[2:end, 1, end],
    height[2:end], 
    lw=3,
    label="Final condition"
  )
  savefig("zeroth_moment.png")

  plot(
    moments[2:end, 2, 1] * 1e-9 * 1e6 / 1.225, # nanograms for droplets and conversion to spec. hum.
    height[2:end],
    lw=3,
    xaxis="Liquid water specific humidity [g/kg]",
    yaxis="Height [m]",
    xlims=(0, 1500.0 * 1e-3),
    ylims=(height[1], height[end]),
    label="Initial condition",
    title="First moment - mass"
  )
  plot!(
    moments[2:end, 2, end] * 1e-9 * 1e6 / 1.225,
    height[2:end], 
    lw=3,
    label="Final condition"
  )
  savefig("first_moment.png")
  
  plot(
    time / 60.0,
    surface_mass * 1e-12 * 1e6, 
    lw=3,
    xaxis="Time [min]",
    yaxis="Mass [kg]",
    xlims=(time[1] / 60.0, time[end] / 60.0),
    ylims=(0.0, 2 * findmax(column_mass)[1] *1e-12 * 1e6),
    label="Surface mass (fallen precipitation)",
    title="Surface"
  )
  plot!(
    time / 60.0,
    column_mass * 1e-12 * 1e6,
    lw=3,
    label="Column-integrated mass"
  )
  plot!(
    time / 60.0,
    (column_mass .+ surface_mass) .* 1e-12 * 1e6,
    lw=1,
    linecolor=:black,
    linestyle=:dash,
    label=""
  )
  savefig("surface.png")
end

# Run everything!
plotting(main()...)
