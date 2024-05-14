# Cloudy.jl
`Cloudy.jl` is a flexible N-moment microphysics scheme which currently handles warm-rain processes including coalescence, condensation/evaporation, and sedimentation (terminal velocity). Unlike typical moment-based schemes which distinguish between categories such as rain and cloud, and which determine rates of conversion between categories (the canonical autoconversion, accretion, and self-collection), this package gives the user the flexibility to define as many or as few moments as they please, with these coalescence-based processes being solved directly without relying on conversion rates. Likewise, the rate of condensation/evaporation is defined through the rate of diffusion of water vapor to/from the surface of droplets defined by the subdistributions which underpin the method. The user has not only the flexibility to specify the number of moments (and therefore the complexity/accuracy) to use, but also the assumed size distributions corresponding to these moments. For instance, one might define a 5-moment implementation using an Exponential mode for smaller cloud droplets, plus a Gamma mode for larger rain droplets. Or, more creatively, perhaps a 12-moment implementation comprised of four Gamma modes.

This package contains the source code implementation of this new method-of-moments as well as a set of simple examples in a box, parcle, and rainshaft setting. More complex simulations utilizing `Cloudy.jl` are currently under development in `KinematicDriver.jl` and `ClimaAtmos.jl`. 

## Prognostic Variables

The prognostic variables of this method are a set of N moments, which can be further divided into P sets of moments, each of which correponds to a subdistribution p. By design these moments begin at order 0 and increase as integers up to the maximum number of parameters for the chosen subdistribution. The first three such default moments have interpretable meanings:
  - ``M_0`` - the number density of droplets [1/m^3]
  - ``M_1`` - the mass density of droplets [kg/m^3]
  - ``M_2`` - proportional to the radar reflectivity [kg^2/m^3]
and can be converted to more canonical definitions of `q_liq` and `q_rai` through numerical integration.

When the user wishes to use more than 2 or 3 total variables to represent the system, these moments must be divided between ``P > 1`` sub-distributions, each of which assumes the form of a particular mathematical distribution, such as an Exponential, Lognormal, or Monodisperse (each of which has two parameters), or a Gamma distribution (which takes 3 parameters). 
