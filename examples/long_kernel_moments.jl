"Linear coalescence kernel example"

using DifferentialEquations
using LinearAlgebra
using Plots
using Random: seed!

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.Sources

seed!(126)


function main()
  # Numerical parameters
  tol = 1e-4
  n_samples = 100
 
  # Physicsal parameters
  # Time has been rescaled below by a factor of 1e1 so that 1 sec = 10 deciseconds
  time_scale = 1e6

  T_end = 15 * 60 * time_scale
  cloud_coalescence_coeff = 9.44e9 / time_scale #9.44e9 cm^3 g^-2 s-1
  rain_coalescence_coeff = 5.78e3 / time_scale #5.78e3 cm^3 g^-1 s-1
  mass_threshold = 5e-7 #5e-7 g
  kernel_func = LongKernelFunction(cloud_coalescence_coeff, rain_coalescence_coeff, mass_threshold)
  
  # Parameter transform used to transform native distribution
  # parameters to moments and back
  trafo = native_state -> [native_state[1], native_state[1]*native_state[2]*native_state[3], native_state[1]*native_state[2]*(native_state[2]+1)*native_state[3]^2]
  inv_trafo = state -> [state[1], (state[2]/state[1])/(state[3]/state[2]-state[2]/state[1]), state[3]/state[2]-state[2]/state[1]]

  # Initial condition
  particle_number = 1e4
  mean_particles_mass = 1.2e-9 #1.2e-9 g
  particle_mass_std = 1.2e-9 #1.2e-9 g
  pars_init = [particle_number; (mean_particles_mass/particle_mass_std)^2; particle_mass_std^2/mean_particles_mass]
  state_init = trafo(pars_init)

  # Set up the ODE problem
  # Step 1) Define termination criterion: stop integration when one of the 
  #         distribution parameters leaves its allowed domain (which can 
  #         happen before the end of the time period defined below by tspan)
  nothing

  # Step 2) Set up the right hand side of ODE


  function rhs!(dstate, state, p, t)
    # Transform state to native distribution parameters
    native_state = inv_trafo(state)

    # Evaluate processes for moments using a closure distribution
    pdist = GammaParticleDistribution(native_state[1], native_state[2], native_state[3])
    coal_int = similar(state)
    for k in 1:length(coal_int)
        coal_int[k] = get_coalescence_integral_moment(k-1, kernel_func, pdist, n_samples)
    end

    # Assign time derivative
    for i in 1:length(dstate)
        dstate[i] = coal_int[i]
    end
  end

  # Step 3) Solve the ODE
  tspan = (0.0, T_end)
  prob = ODEProblem(rhs!, state_init, tspan)
  sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

  # Step 4) Plot the results
  time = sol.t / time_scale

  # Get the native distribution parameters
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = vcat(sol.u'...)[:, 3]

  # Calculate the rain rain fraction
  rain_threshold = 5e-8 #5e-8 g
  rain_frac = similar(moment_0)
  for i in 1:length(rain_frac)
    n, k, θ = inv_trafo([moment_0[i], moment_1[i], moment_2[i]])
    pdist = GammaParticleDistribution(n, k, θ)
    rain_frac[i] = moment(pdist, 1, rain_threshold, 100.0) / moment_1[i]
  end

  println("Rain fraction at end of simulation:")
  println(rain_frac[end])

  p1 = plot(time,
      moment_0,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M0 [1/cm^3]",
      xlims=(0, maximum(time)),
      ylims=(0, 1.5*maximum(moment_0)),
      label="M0 CLIMA"
  )
  p2 = plot(time,
      moment_1,
      linewidth=3,
      xaxis="time",
      yaxis="M1 [grams / cm^3]",
      ylims=(0, 1.5*maximum(moment_1)),
      label="M1 CLIMA"
  )
  p3 = plot(time,
      moment_2,
      linewidth=3,
      xaxis="time",
      yaxis="M2 [grams^2 / cm^3]",
      ylims=(0, 1.5*maximum(moment_2)),
      label="M2 CLIMA"
  )
  plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
  savefig("long_kernel_moment_test.png")
end

main()
