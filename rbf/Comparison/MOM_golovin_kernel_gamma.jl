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
  coalescence_coeff = 8e-7
  kernel_func = x -> (coalescence_coeff*(x[1]+x[2]))
  kernel = CoalescenceTensor(kernel_func, 0, 100.0)

  # Initial condition:
  N0 = 1000.0
  mu = 15.0
  sigma = 5.0
  dist_init = GammaPrimitiveParticleDistribution(N0, mu, sigma)
  moments_init = [1e5, 1.5e6, 2.5e7]

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

  # Plot the solution for the moments and compare to analytical solution
  pyplot()
  time = sol.t
  moment_0 = vcat(sol.u'...)[:, 1]/moments_init[1]
  moment_1 = vcat(sol.u'...)[:, 2]/moments_init[2]
  moment_2 = vcat(sol.u'...)[:, 3]/moments_init[3]
  plot(t_coll,
  t-> (moments_init[1]*exp(-b*moments_init[2]*t))/moments_init[1],
      linewidth=3,
      title="\$C(m, m') = k\$ (Smolu. 1916) vs. GP-Collocation with 3 RBF",
      xaxis="time",
      yaxis="M\$_k\$(time)",
      xlims=tspan,
      ylims=(0, 10), 
      ls=:dash,
      label="M\$_0\$ Exact"
  )
  
  plot!(t_coll,
      t-> 1,
      lw=3,
      ls=:dash,
      label="M\$_1\$ Exact"
  )
  
  plot!(t_coll,
      t-> (moments_init[3]*exp(2*b*moments_init[2]*t))/moments_init[3],
      lw=3,
      ls=:dash,
      label="M\$_2\$ Exact"
  )
  plot!(time,
      moment_0,
      linewidth=3,
      title="\$C(m, m') = k\$ (Smolu. 1916) vs. Climate Machine",
      xaxis="time",
      yaxis="M\$_k\$(time)/M\$_k\$(0)",
      xlims=tspan,
      ylims=(0, 10), 
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

  savefig("rbf/Comparison/MOM_golovin_kernel_gamma.png")

# plot the actual distribution
tplot = [0.0, 0.5, 1.0]
x = collect(range(eps(), stop=250.0, length=1000))
plot(x, 
    density_eval(dist_init, x)/N0,
    linewidth=2,
    title="\$C(m, m') = k\$ (Smolu. 1916) vs. Climate Machine",
    xaxis="mass",
    yaxis="Probability distribution (normalized)",
    label="t = 0"
)

for i=2:length(tplot)
  moments = sol(tplot[i])[1:3]
  dist = update_params_from_moments(ODE_parameters, moments)
  plot!(x,
      density_eval(dist, x)/moments[1],
      linewidth=2,
      label=string("t = ", tplot[i]))
end
savefig("rbf/Comparison/MOM_golovin_kernel_gammadist.png")

end

main()
