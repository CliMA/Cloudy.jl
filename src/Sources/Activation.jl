"""
  multi-moment bulk droplet activation
  - condensation/evaporation as relaxation to equilibrium
"""

module Activation

using ..ParticleDistributions

export get_aerosol_growth_3mom

"""
get_aerosol_growth_3mom(mom_p::Array{Float64}, ODE_parameters::Dict, t::FT, v_up::Float64=1)

- `mom_p` - prognostic moments of particle mass distribution
- `ODE_parameters` - ODE parameters, a Dict containing a key ":dist" whose 
                     is the distribution at the previous time step. dist is a 
                     ParticleDistribution; it is used to calculate the 
                     diagnostic moments and also to dispatch to the
                     moments-to-parameters mapping (done by the function
                     moments_to_params) for the given type of distribution
- `t' - current time step, used for print statement purposes only
- 'v_up' - updraft velocity for the parcel, defaults to 1 m/s

"""
function get_aerosol_growth_3mom(mom_p::Array{FT}, ODE_parameters::Dict, t::FT, v_up::FT=1.0) where {FT}
  println("time: ", t)
  println("prognostic moments: ", mom_p)

  dist = update_params_from_moments(ODE_parameters, mom_p[1:3])
  ODE_parameters[:dist] = dist
  #println("Distribution: ", dist)

  mom_d = Array{FT}(undef, 4)
  S = mom_p[end]

  # compute the diagnostic moments: M-1 through M-4
  s = 5; #add to moment indexing
  for k in -4:-1
    mom_d[k+s] = moment(dist, FT(k))
  end
  mom = vcat(mom_d, mom_p)
  #println(mom)

  coeffs = get_aerosol_coefficients(FT)
  ddt = Array{FT}(undef,4)

  # compute the time rate of change
  ddt[1] = 0;
  ddt[2] = coeffs[1]*S*mom[-1+s] + coeffs[2]*mom[-2+s] + coeffs[3]*mom[-4+s]
  ddt[3] = 2*(coeffs[1]*S*mom[0+s] + coeffs[2]*mom[-1+s] + coeffs[3]*mom[-3+s])
  # dS/dt
  ddt[end] = coeffs[4]*v_up - coeffs[5]*(coeffs[1]*S*mom[1+s] + coeffs[2]*mom[0+s] + coeffs[3]*mom[-2+s])
  #println("Derivative wrt  time: ", ddt)
  println()
  return ddt
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

end