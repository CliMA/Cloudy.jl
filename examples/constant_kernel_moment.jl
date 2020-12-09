"Constant coalescence kernel example"

using DifferentialEquations
using LinearAlgebra
using Plots
using Random: seed!

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.Sources

seed!(123)


function main()
  # Numerical parameters
  tol = 1e-4
  n_samples = 125

  # Physicsal parameters
  # Rescale time and mass to get better stability properties 
  time_scale = 1e-1
  
  T_end = 200.0 / time_scale #300 s
  coalescence_coeff = 1.0e-6 * time_scale # 1.0e-6 cm^3 s-1  
  kernel_func = ConstantKernelFunction(coalescence_coeff)

  # Parameter transform used to transform native distribution
  # parameters to moments and back
  ϵ = eps(Float64)
  trafo = native_state -> [native_state[1], native_state[1]*native_state[2]*native_state[3], native_state[1]*native_state[2]*(native_state[2]+1)*native_state[3]^2]
  inv_trafo = state -> [ϵ + state[1], ϵ + (state[2]/state[1])/(state[3]/state[2]-state[2]/state[1]), ϵ + state[3]/state[2]-state[2]/state[1]]

  # Initial condition
  particle_number = 1e4 # 1e4 g^-1
  mean_particles_mass = 1e-8 # 1e-8 g
  particle_mass_std = 0.5e-8 # 0.5e-8 g
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
  time = sol.t

  # Get the native distribution parameters
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = vcat(sol.u'...)[:, 3]

  p1 = plot(time,
      moment_0,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M0 [1/cm^3]",
      xlims=(0, maximum(time)),
      ylims=(0, 1.5*maximum(moment_0)),
      label="M0 CLIMA"
  )
  plot!(p1, time,
      t-> (1 / moment_0[1] + 0.5 * coalescence_coeff * t)^(-1),
      lw=3,
      ls=:dash,
      label="M0 Exact"
  )

  p2 = plot(time,
      moment_1,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M1 [grams/cm^3]",
      ylims=(0, 1.5*maximum(moment_1)),
      label="M1 CLIMA"
  )
  plot!(p2, time,
      t-> moment_1[1],
      lw=3,
      ls=:dash,
      label="M1 Exact"
  )
  p3 = plot(time,
      moment_2,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M2 [grams^2/cm^3]",
      ylims=(0, 1.5*maximum(moment_2)),
      label="M2 CLIMA"
  )
  plot!(p3, time,
      t-> moment_2[1] + moment_1[1]^2 * coalescence_coeff * t,  lw=3,
      ls=:dash,
      label="M2 Exact"
  )
  plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
  savefig("constant_kernel_moment_test.png")
end

main()
