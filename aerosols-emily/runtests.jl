# Single aerosol species MOM evolution

using Test
using DifferentialEquations
using Plots

using Cloudy.ParticleDistributions

function main()
  # Numerical parameters
  FT = Float64
  tol = FT(1e-8)

  # Initial condition:
  S_init = FT(0.05)
  dist_init = GammaPrimitiveParticleDistribution(FT(340), FT(28.0704505), FT(3.8564))
  v_up = FT(1)
  tspan = (FT(0), FT(1))

  moments_S_init = FT[0.0, 0.0, 0.0, S_init]
  println("Initializing with moments:")
  for k in 0:2
    moments_S_init[k+1] = moment(dist_init, FT(k))
    println(moments_S_init[k+1])
  end
  println("Supersaturation: ", S_init)
  println()

  ODE_parameters = Dict(:dist => dist_init)
  # implement callbacks to halt the integration
  function condition(m,t,integrator) 
    println("Condition checked")
    t>=1e-9
  end

  function affect!(integrator) 
    terminate!(integrator)
  end
  
  cb=DiscreteCallback(condition, affect!)
  # set up ODE
  rhs(m, par, t) = get_aerosol_growth_3mom(m, par, t, v_up)

  # solve the ODE
  println("Solving ODE...")
  prob = ODEProblem(rhs, moments_S_init, tspan, ODE_parameters)
  alg = Tsit5()
  sol = solve(prob, alg, dt=1e-6, callback=cb)
  #sol = solve(prob, reltol = tol, abstol = tol, isoutofdomain = (m,par,t) -> any(x->x<0, m))

  # Plot the solution for the 0th moment
  pyplot()
  gr()
  time = sol.t
  mom = vcat(sol.u'...)
  moment_0 = mom[:, 1]
  moment_1 = mom[:, 2]
  moment_2 = mom[:, 3]
  S = vcat(sol.u'...)[:,4]

  plot(time,
      moment_0,
      linewidth=3,
      xaxis="time",
      yaxis="M\$_k\$(time)",
      xlims=(0, 1.0),
      ylims=(0, 600.0),
      label="M\$_0\$ CLIMA"
  )
  plot!(time,
      moment_1,
      linewidth=3,
      label="M\$_1\$ CLIMA"
  )
  plot!(time,
      moment_2,
      linewidth=3,
      label="M\$_2\$ CLIMA"
  )
  savefig("aerosol_growth.png")

  pyplot()
  gr()
  plot(time,
      S,
      linewidth=3,
      label="S CLIMA")
  savefig("aerosol_growth_S.png")
end

"""
get_aerosol_growth_3mom(mom_p::Array{Float64}, v_up::Float64=1)

  - 'T' - temperature in K
  - 'P' - pressure in Pa
  - 'V' - volume of box, m3
  - 'kappa' - hygroscopicity of particles
  - 'rd' - dry radius of particles, m
  Returns the coefficients for aerosol growth.
  - 'a' = [G, GA, -G k rd^3, alpha, gamma] [=] [m2/sec, m3/sec, m5/sec, 1/m2, 1/m3]

"""
function get_aerosol_growth_3mom(mom_p::Array{FT}, ODE_parameters::Dict, t::FT, v_up::FT=1.0) where {FT}
  println("time: ", t)
  println("prognostic moments: ", mom_p)

  try
    dist = update_params_from_moments(ODE_parameters, mom_p[1:3])
    ODE_parameters[:dist] = dist
    println("Distribution: ", dist)
  
    mom_d = Array{FT}(undef, 4)
    S = mom_p[end]
  
    # compute the diagnostic moments: M-1 through M-4
    s = 5; #add to moment indexing
    for k in -4:-1
      mom_d[k+s] = moment(dist, FT(k))
    end
    mom = vcat(mom_d, mom_p)
    println("diagnostic moments: ", mom_d)
  
    coeffs = get_aerosol_coefficients(FT)
    ddt = Array{FT}(undef,4)
  
    # compute the time rate of change
    ddt[1] = 0;
    ddt[2] = coeffs[1]*S*mom[-1+s] + coeffs[2]*mom[-2+s] + coeffs[3]*mom[-4+s]
    ddt[3] = 2*(coeffs[1]*S*mom[0+s] + coeffs[2]*mom[-1+s] + coeffs[3]*mom[-3+s])
    # dS/dt
    ddt[end] = coeffs[4]*v_up - coeffs[5]*(coeffs[1]*S*mom[1+s] + coeffs[2]*mom[0+s] + coeffs[3]*mom[-2+s])
    println("Derivative wrt  time: ", ddt)
    println()
    return ddt
  catch e
    ddt = [Inf, Inf, Inf, Inf]
    return ddt
  end

end

"""
    get_aerosol_coefficients(kappa::FT=0.6, rd::FT=1; T::FT=285, P::FT=95000, V::FT=1) where {FT}

 - 'T' - temperature in K
 - 'P' - pressure in Pa
 - 'V' - volume of box, m3
 - 'kappa' - hygroscopicity of particles
 - 'rd' - dry radius of particles, m
 Returns the coefficients for aerosol growth.
 - 'a' = [G, GA, -G k rd^3, alpha, gamma] [=] [m2/sec, m3/sec, m5/sec, 1/m2, 1/m3]

"""
function get_aerosol_coefficients(::Type{FT};
  kappa::FT=0.6,
  rd::FT=10.0,
  T::FT=285.0,
  P::FT=95000.0,
  V::FT=1.0e-6
  ) where {FT}

  # specify physical constants
  cp = 1005               # Specific heat of air: J/kg K
  Mw = 0.018              # Molecular weight of water: kg/mol
  Ma = 0.029              # Molecular weight of dry air: kg/mol
  g = 9.8                 # acceleration due to gravity: m/s^2
  R = 8.314               # Universal gas constant: J/K mol
  sigma_water = 0.072225  # surface tension of water (N/m)
  rho_w = 997             # density of water (kg/m3)

  P_atm = P/101325        # Pressure (atm)
  Dv = (0.211/P_atm) * (T/273)^(1.94)*1e-4 # Mass diffusivity of water in air (m2/s or J/kg)

  # temperature-dependent parameters
  temp_c = T - 273.15
  a0 = 6.107799
  a1 = 4.436518e-1
  a2 = 1.428945e-2
  a3 = 2.650648e-4
  a4 = 3.031240e-6
  a5 = 2.034081e-8
  a6 = 6.136829e-11
  # vapor pressure of water (Pa)
  Po = 100*(a0+a1*temp_c+a2*(temp_c^2)+a3*(temp_c^3)+a4*(temp_c^4)+a5*(temp_c^5)+a6*(temp_c^6))
  # thermal conductivity of air (W/m K)
  ka = 1e-3*(4.39+0.071*T)
  # density of air (kg/m3)
  rho_a = P/(287.058*T)
  # latent heat of vaporization: J/kg
  Hv = (2.5*((273.15/T)^(0.167+3.67e-4*T)))*1e6

  # Generalized coefficients
  G = 1/((rho_w*R*T/Po/Dv/Mw) + (Hv*rho_w/ka/T/(Hv*Mw/T/R - 1))) * 1e18     #nm2/sec
  A = 2*Mw*sigma_water/R/T/rho_w *1e9                                       #nm
  alpha = Hv*Mw*g/cp/R/T^2 - g*Ma/R/T                                       #1/m
  gamma = P*Ma/Po/Mw + Hv^2*Mw/cp/R/T^2
  gamma2 = gamma*4*pi/rho_a/rho_w/V*1e-27                                   #1/nm3

  # 3-moment ODE coefficients
  a = Array{FT}(undef, 5)
  a[1] = G;               #nm2/sec
  a[2] = G*A;             #nm3/sec
  a[3] = -G*kappa*rd^3;   #nm5/sec
  a[4] = alpha;           #1/m
  a[5] = gamma2;          #1/nm3

  return a
end

@time main()
