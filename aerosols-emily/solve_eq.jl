using NLsolve

function f!(F,x)
    k = x[2]
    θ = x[1]
    
    S = x[3]
    kappa = 0.6
    M3_dry = 7.701e6
    coeffs = get_aerosol_coefficients(kappa, M3_dry)

    F[1] = coeffs[1]*k*(k-1)*(k-2)*θ^3*S + coeffs[2]*(k-1)*(k-2)*θ^2 + coeffs[3]
    F[2] = coeffs[1]*(k-1)*(k-2)*(k-3)*θ^3*S + coeffs[2]*(k-2)*(k-2)*θ^2 + coeffs[3]
    #F[3] = coeffs[1]*(k-2)*(k-3)*(k-4)*θ^3*S + coeffs[2]*(k-3)*(k-4)*θ^2 + coeffs[3]
end

function get_aerosol_coefficients(kappa::FT, M3_dry::FT;
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
    a[3] = -G*kappa*M3_dry; #nm5/sec
    a[4] = alpha;           #1/m
    a[5] = gamma2;          #1/nm3
  
    return a
  end

initial_x = [28.0, 3.0, 0.001]
nlsolve(f!, initial_x)