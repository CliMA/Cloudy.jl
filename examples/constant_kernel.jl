"Constant coalescence kernel example"

using DifferentialEquations
using Plots

using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources


function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-8

  # Physicsal parameters
  coalescence_coeff = 1/3.14/4
  kernel_func = x -> coalescence_coeff
  kernel = CoalescenceTensor(kernel_func, 0, 100.0)

  # Initial condition
  moments_init = [150.0, 30.0, 200.0]
  dist_init = GammaPrimitiveParticleDistribution(150.0, 6.466666667, 0.03092815)

  # Set up the ODE problem
  # Step 1) Define termination criterion: stop integration when one of the 
  #         distribution parameters leaves its allowed domain (which can 
  #         happen before the end of the time period defined below by tspan)
  """
  function out_of_bounds(m, t, integrator, lbound, ubound)

  - `m` - unknown function of the ODE; moments as a function of t
  - `t` - ODE time
  - `integrator` - ODE integrator (see DifferentialEquations.jl docs)
                   Its user-provided data (integrator.p) has to be a
                   Dict with a key ":dist", whose value is the 
                   ParticleDistribution at the previous time step.
  - `lbound` - the lower bounds of the parameters. Default to -Inf if left
               unspecified.
  - `ubound` - the upper bounds of the parameters. Default to Inf if left
               unspecified.

  Returns true if one or more of the distribution parameters are outside their
  bounds. Returns false otherwise.
  """
  function out_of_bounds(m, t, integrator; 
                         lbound::Union{AbstractVector, Nothing}=nothing, 
                         ubound::Union{AbstractVector, Nothing}=nothing)

    dist = integrator.p[:dist]
    dist_params = reduce(vcat, get_params(dist)[2])
    n_params = length(dist_params)

    lbound == nothing && (lbound = -Inf * ones(n_params))
    ubound == nothing && (ubound = Inf * ones(n_params))

    if sum(lbound .<= dist_params .<= ubound) < n_params
      println("Exiting integration at time t=$(integrator.t)")
      println("At least one of the distribution parameters has left the ",
              "allowed domain")
      return true
    else
      return false
    end
  end

  # Define bounds: n ≧ 1.0, θ > 0.0, k > 0.0)
  lbound = [1.0, 1e-5, 1e-5]
  condition(m, t, integrator) = out_of_bounds(m, t, integrator; lbound=lbound)
  affect!(integrator) = terminate!(integrator)
  cb = DiscreteCallback(condition, affect!)

  # Step 2) Set up the right hand side of ODE
  rhs(m, par, t) = get_int_coalescence(m, par, kernel)

  tspan = (0.0, 1.0)
  # Make the initial distribution a parameter of the ODE, so that it can get 
  # updated and carried along over the entire integration time.
  ODE_parameters = Dict(:dist => dist_init)
  prob = ODEProblem(rhs, moments_init, tspan, ODE_parameters)
  sol = solve(prob, Tsit5(), callback=cb, reltol=tol, abstol=tol)

  # Plot the solution for the 0th moment and compare to analytical solution
  time = sol.t
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = vcat(sol.u'...)[:, 3]
  plot(time,
      moment_0,
      linewidth=3,
      title="\$C(m, m') = k\$ (Smolu. 1916) vs. Climate Machine",
      xaxis="time",
      yaxis="M\$_k\$(time)",
      xlims=tspan,
      ylims=(0, 400), 
      label="M\$_0\$ CLIMA"
  )
  plot!(time,
      t-> (1 / moments_init[1] + 0.5 * coalescence_coeff * t)^(-1),
      lw=3,
      ls=:dash,
      label="M\$_0\$ Exact"
  )
  plot!(time,
      moment_1,
      linewidth=3,
      label="M\$_1\$ CLIMA"
  )
  plot!(time,
      t-> moments_init[2],
      lw=3,
      ls=:dash,
      label="M\$_1\$ Exact"
  )
  plot!(time,
      moment_2,
      linewidth=3,
      label="M\$_2\$ CLIMA"
  )
  plot!(time,
      t-> moments_init[3] + moments_init[2]^2 * coalescence_coeff * t,
      lw=3,
      ls=:dash,
      label="M\$_2\$ Exact"
  )
  savefig("constant_kernel_example.png")
end

main()
