"Linear coalescence kernel example"

using DifferentialEquations
using LinearAlgebra
using Plots
using Random: seed!
using JLD2

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.Sources

seed!(123)


function main()

  # Numerical parameters
  tol = 1e-4
  n_samples = 300

  # Physicsal parameters
  # Time has been rescaled below by a factor of 1e1 so that
  # 1 sec = 10 deciseconds
  time_scale = 1e6

  T_end = 120 * time_scale
  coalescence_coeff = 2.0e3 / time_scale # 1.5e3 cm^3 g^-1 s-1
  kernel_func = LinearKernelFunction(coalescence_coeff)

  # Parameter transform used to transform native distribution
  # parameters to moments and back
  trafo = native_state -> [native_state[1], native_state[1]*native_state[2]*native_state[3], native_state[1]*native_state[2]*(native_state[2]+1)*native_state[3]^2]
  inv_trafo = state -> [state[1], (state[2]/state[1])/(state[3]/state[2]-state[2]/state[1]), state[3]/state[2]-state[2]/state[1]]

  # Initial condition
  particle_number = 1e4
  mean_particles_mass = 0.33e-9 #0.33e-9 g = 2.5e-6 m radius
  particle_mass_std = 0.33e-9 #0.33e-9 g
  pars_init = [particle_number; (mean_particles_mass/particle_mass_std)^2;
               particle_mass_std^2/mean_particles_mass]
  state_init = trafo(pars_init)


  # Set up the right hand side of ODE
  function rhs!(dstate, state, p, t)
    # Set value of the coalescence efficiency (assumed to be constant for now)
    coalescence_efficiency = 0.8

    # Transform state to native distribution parameters
    native_state = inv_trafo(state)

    # Evaluate processes for moments using a closure distribution
    pdist = GammaParticleDistribution(native_state[1],
                                      native_state[2],
                                      native_state[3])
    coal_int = similar(state)
    for k in 1:length(coal_int)
        coal_int[k] = get_coalescence_integral_moment(k-1, kernel_func, pdist,
                                                      n_samples, 
                                                      coalescence_efficiency)
    end

    breakup_int = similar(state)
    for k in 1:length(breakup_int)
        breakup_int[k] = get_breakup_integral_moment(k-1, kernel_func, pdist,
                                                     coalescence_efficiency,
                                                     state_init[1],
                                                     state_init[2],
                                                     n_samples)
    end


    
    # Assign time derivative
    for i in 1:length(dstate)
        dstate[i] = coal_int[i] + breakup_int[i]
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
      label="M0"
  )

  p2 = plot(time,
      moment_1,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M1 [grams/cm^3]",
      ylims=(0, 1.5*maximum(moment_1)),
      label="M1"
  )

  p3 = plot(time,
      moment_2,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M2 [grams^2/cm^3]",
      ylims=(0, 1.5*maximum(moment_2)),
      label="M2"
  )

  plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
  savefig("golovin_kernel_moment_test_hcubature.png")
end

main()
