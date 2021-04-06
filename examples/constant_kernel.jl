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
  tol = 2e-4
  n_samples = 10
  n_inducing = 4

  # Physicsal parameters
  coalescence_coeff = 1e-4
  kernel_func = ConstantKernelFunction(coalescence_coeff)
  
  # Parameter transform used to transform native distribution
  # parameters to the real axis
  trafo = native_state -> native_state > 10.0 ? native_state : log.(exp.(native_state) - 1.0)
  inv_trafo = state -> state > 10.0 ? state : log.(exp.(state) + 1.0)
  inv_trafo_der = state -> 1.0 ./ (1.0 + exp.(-state))

  # Initial condition
  # We carrry transformed parameters in our time stepper for
  # stability purposes
  #particle_number = 100.0
  #mean_particles_mass = 10.0
  #particle_mass_std = 5.0
  #pars_init = [particle_number; (mean_particles_mass/particle_mass_std)^2; particle_mass_std^2/mean_particles_mass]
  pars_init = [100.0, 3.0, 100.0]
  state_init = trafo.(pars_init) 

  # Set up the ODE problem
  # Step 1) Define termination criterion: stop integration when one of the 
  #         distribution parameters leaves its allowed domain (which can 
  #         happen before the end of the time period defined below by tspan)
  nothing

  # Step 2) Set up the right hand side of ODE
  function rhs!(dstate, state, p, t)
    # Transform state to native distribution parameters
    native_state = inv_trafo.(state)

    # Evaluate processes at inducing points using a closure distribution
    pdist = GammaParticleDistribution(native_state[1], native_state[2], native_state[3])
    inducing_points = sample(pdist, n_inducing)
    coal_int = get_coalescence_integral(inducing_points, kernel_func, pdist, n_samples)

    # Obtain time derivatve of the transformed distribution parameters
    jacobian = density_gradient(pdist, inducing_points) * diagm(inv_trafo_der.(state))
    transformed_int = inv(jacobian'*jacobian)*jacobian'*coal_int

    # Projection to enforce mass conservation in transformed space
    normal = normal_mass_constraint(pdist)
    transformed_normal = diagm(inv_trafo_der.(state)) * normal
    unit_normal = transformed_normal / norm(transformed_normal)
    transformed_int = (I - unit_normal * unit_normal') * transformed_int

    # Assign time derivative
    for i in 1:length(dstate)
        dstate[i] = transformed_int[i]
    end
  end

  # Step 3) Solve the ODE
  tspan = (0.0, 360.0)
  prob = ODEProblem(rhs!, state_init, tspan)
  sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

  # Step 4) Plot the results
  time = sol.t

  # Get the native distribution parameters
  n = inv_trafo.(vcat(sol.u'...)[:, 1])
  k = inv_trafo.(vcat(sol.u'...)[:, 2])
  θ = inv_trafo.(vcat(sol.u'...)[:, 3])
  println("n =", n[end])
  println("k =", k[end])
  println("theta =", θ[end])

  t_plot = collect(range(0.0, stop=360.0, length=21))
  ns = inv_trafo.(vcat(sol.(t_plot)'...)[:, 1])
  ks = inv_trafo.(vcat(sol.(t_plot)'...)[:, 2])
  θs = inv_trafo.(vcat(sol.(t_plot)'...)[:, 3])
  m0 = ns
  m1 = ns.*ks.*θs
  m2 = ns.*ks.*(ks.+1.0).*θs.^2
  println("times = ", t_plot)
  println("M0 = ", m0)
  println("M1 = ", m1)
  println("M2 = ", m2)

  # Calculate moments for plotting
  moment_0 = n
  moment_1 = n.*k.*θ
  moment_2 = n.*k.*(k.+1.0).*θ.^2

  p1 = plot(time,
      moment_0,
      linewidth=3,
      xaxis="time",
      yaxis="M0",
      xlims=tspan,
      ylims=(0, 1.5*maximum(moment_0)),
      label="M0 CLIMA"
  )
  plot!(p1, time,
      t-> (1 / moment_0[1] + 0.5 * kernel_func.coll_coal_rate * t)^(-1),
      lw=3,
      ls=:dash,
      label="M0 Exact"
  )

  p2 = plot(time,
      moment_1,
      linewidth=3,
      xaxis="time",
      yaxis="M1",
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
      xaxis="time",
      yaxis="M2",
      ylims=(0, 1.5*maximum(moment_2)),
      label="M2 CLIMA"
  )
  plot!(p3, time,
      t-> moment_2[1] + moment_1[1]^2 * kernel_func.coll_coal_rate * t,
      lw=3,
      ls=:dash,
      label="M2 Exact"
  )
  plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
  savefig("examples/constant_kernel_test.png")
end

main()
