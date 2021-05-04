"Linear (Golovin) coalescence kernel example"

using DifferentialEquations
using LinearAlgebra
using Plots
using Random: seed!

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.Sources

using QuadGK

seed!(123)


function main()
  # Numerical parameters
  tol = 1e-4
  n_samples = 25
  n_inducing = 5

  # Physicsal parameters
  # Mass has been rescaled below by a factor of 1e3 so that 1 gram = 1e3 milligram 
  # Time has been rescaled below by a factor of 1e1 so that 1 sec = 10 deciseconds
  mass_scale = 1 #1e3
  time_scale = 1 #1e1
  
  T_end = 4*3600.0
  coalescence_coeff = 1500 * (1e-18) * (1e6) 
  kernel_func = LinearKernelFunction(coalescence_coeff)
  
  # Parameter transform used to transform native distribution
  # parameters to the real axis
  trafo = native_state -> native_state > 10.0 ? native_state : log.(exp.(native_state) - 1.0)
  inv_trafo = state -> state > 10.0 ? state : log.(exp.(state) + 1.0)
  inv_trafo_der = state -> 1.0 ./ (1.0 + exp.(-state))

  # Initial condition
  # Mode 1: Gaussian, radius 8 um, N = 10, theta = 2um
  # Mode 2: Gaussian, radius 4 um, N = 90, theta = 1um
  μ_r_1 = 16.0/2
  θ_r_1 = 2.0/2
  N_1   = 10.0
  μ_r_2 = 8.0/2
  θ_r_2 = 1.0/2
  N_2   = 90.0
  r = v->(3/4/pi*v)^(1/3)
  drdv = v->(1/4/pi/r(v)^2)
  mode1_r = r->(N_1/θ_r_1/sqrt(2*pi)*exp(-(r - μ_r_1)^2/2/θ_r_1^2))
  mode2_r = r->(N_2/θ_r_2/sqrt(2*pi)*exp(-(r - μ_r_2)^2/2/θ_r_2^2))
  n_v_init = v->(mode1_r(r(v)) + mode2_r(r(v)))*drdv(v)
  vmin = 1.0
  vmax = 1e4
  
  # We carrry transformed parameters in our time stepper for
  # stability purposes
  M0_0 = quadgk((x->n_v_init(x)), vmin, vmax)[1]
  M1_0 = quadgk((x->x .* n_v_init(x)), vmin, vmax)[1]
  M2_0 = quadgk((x->(x.^2) .* n_v_init(x)), vmin, vmax)[1]
  println(M0_0,", ", M1_0, ", ", M2_0)
  N_init = M0_0
  k_init = M1_0^2/(M2_0*M0_0 - M1_0^2)
  θ_init = M2_0/M1_0/(k_init + 1)
  pars_init = [N_init; k_init; θ_init]
  println(pars_init)
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
  tspan = (0.0, T_end)
  prob = ODEProblem(rhs!, state_init, tspan)
  sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

  # Step 4) Plot the results
  time = sol.t / time_scale

  # Get the native distribution parameters
  n = inv_trafo.(vcat(sol.u'...)[:, 1])
  k = inv_trafo.(vcat(sol.u'...)[:, 2])
  θ = inv_trafo.(vcat(sol.u'...)[:, 3])

  println("n =", n[end])
  println("k =", k[end])
  println("theta =", θ[end])

  t_plot = collect(range(0.0, stop=T_end, length=21))
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
      xaxis="time [s]",
      yaxis="M0 [1/cm^3]",
      xlims=(0, maximum(time)),
      ylims=(0, 1.5*maximum(moment_0)),
      label="M0 CLIMA"
  )
  plot!(p1, time,
      t-> (moment_0[1] * exp(-moment_1[1] * coalescence_coeff * t * time_scale)),
      lw=3,
      ls=:dash,
      label="M0 Exact"
  )

  p2 = plot(time,
      moment_1,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M1 [milligrams/cm^3]",
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
      yaxis="M2 [milligrams^2/cm^3]",
      ylims=(0, 1.5*maximum(moment_2)),
      label="M2 CLIMA"
  )
  plot!(p3, time,
  t-> (moment_2[1] * exp(2 * moment_1[1] * coalescence_coeff * t * time_scale)),
  lw=3,
      ls=:dash,
      label="M2 Exact"
  )
  plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
  savefig("examples/two_modes_golovin_kernel_test.png")
end

main()
