
function weno(w, flux, dflux, S, dx)
  # Lax-Friedrichs Flux Splitting
  a = findmax(abs.(dflux(w)))[1]
  v = 0.5 .* (flux(w) .+ a .* w) 
  u = circshift(0.5 .* (flux(w) .- a .* w), -1)
  
  # Right Flux
  # Choose the positive fluxes, 'v', to compute the right cell boundary flux:
  # $u_{i+1/2}^{-}$
  vm = circshift(v, 1)
  vm[1] = 0.0
  vp = circshift(v, -1)
  vm[end] = 0.0

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
  w0n = alpha0n ./ alphasumn
  w1n = alpha1n ./ alphasumn
  
  # Numerical Flux at cell boundary, $u_{i+1/2}^{-}$
  hn = w0n .* p0n .+ w1n .* p1n
  hn[end] = 0.0

  # Left Flux 
  # Choose the negative fluxes, 'u', to compute the left cell boundary flux:
  # $u_{i-1/2}^{+}$ 
  um  = circshift(u, 1)
  up  = circshift(u, -1)
  up[end] = 0.0
  up[end-1] = 0.0
  u[end] = 0.0

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
  w0p = alpha0p ./ alphasump
  w1p = alpha1p ./ alphasump
  
  # Numerical Flux at cell boundary, $u_{i-1/2}^{+}$
  hp = w0p .* p0p .+ w1p .* p1p
  hp[1] = 0.0

  # Compute finite volume residual term, df/dx.
  res = (hp .- circshift(hp, 1) .+ hn .- circshift(hn, 1)) ./ dx - S(w)

  return res
end


function main()
  # Parameters
  nx = 31 #number of cells
  CFL = 0.31 # Courant Number
  tEnd = 0.80 # End time
  # Source term
  
  # Build discrete domain
  a = -1 
  b = 1 
  dx = (b-a) / nx 
  x = a+dx/2:dx:b 
  
  # Solver loop
  t = 0 
  it = 0 
  u = u0

  # Let's keep track of the integral of the field to track conservation
  int_u = [sum(u .* dx)]
  min_u = [findmin(u)[1]]

  while t < tEnd
    # Update/correct time step
    dt = CFL * dx / findmax(abs.(u))[1] 
    if t + dt > tEnd
      dt = tEnd - t 
    end
    
    # Update iteration counter
    it = it + 1
    t = t + dt
    
    # RK step
    uo = u
    
    # 1st stage
    # Fix negative values (WARNING: MASSIVE HACK!)
    dF = weno(u, flux, dflux, S, dx)
    u = uo .- dt .* dF
    
    # 2nd Stage
    # Fix negative values (WARNING: MASSIVE HACK!)
    dF = weno(u, flux, dflux, S, dx)
    u = 0.75 .* uo .+ 0.25 .* (u .- dt .* dF)

    # 3rd stage
    # Fix negative values (WARNING: MASSIVE HACK!)
    dF = weno(u, flux, dflux, S, dx)
    u = (uo .+ 2 .* (u .- dt .* dF)) ./ 3.0

    # Fix negative values (WARNING: MASSIVE HACK!)
    append!(int_u, sum(u .* dx))
    append!(min_u, findmin(u)[1])
  end
  plot!(x, u)
  savefig("state.png")

  # Check for conservation property and negative values
  println(findmin(int_u / int_u[1])[1])
  println(findmax(int_u / int_u[1])[1])
  println(findmin(min_u)) 
end

main()
