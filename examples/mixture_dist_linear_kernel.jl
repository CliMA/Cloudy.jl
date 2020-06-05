"Linear coalescence kernel example"

using DifferentialEquations
using Plots

using Cloudy.KernelTensors
using Cloudy.ParticleDistributions
using Cloudy.Sources


function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-5

  # Physical parameters
  kernel_func = x -> 1/3.14/4/1e3 * (x[1] + x[2])
  kernel = CoalescenceTensor(kernel_func, 1, 700.0)

  # Initial condition
  moments_init = [1000.0, 6020.0, 56506.9, 724712.35, 1.18e7]
  dist_init = GammaAdditiveParticleDistribution(
                    GammaPrimitiveParticleDistribution(500.0, 3.5, 2.0), 
                    GammaPrimitiveParticleDistribution(500.0, 2.8, 1.8)
              )
                  
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
  lbound = [1.0, 1e-5, 1e-5, 1.0, 1e-5, 1e-5]
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

  # Plot the solution for the first 4 moments
  pyplot()
  time = sol.t
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = vcat(sol.u'...)[:, 3]
  moment_3 = vcat(sol.u'...)[:, 4]

  p0 = plot(time,
            moment_0,
            linewidth=3,
            xlims=tspan,
            xaxis="time",
            yaxis="M0"
  )
  p1 = plot(time,
            moment_1,
            linewidth=3,
            xlims=tspan,
            xaxis="time",
            yaxis="M1"
  )
  p2 = plot(time,
            moment_2,
            xlims=tspan,
            linewidth=3,
            xaxis="time",
            yaxis="M2"
  )
  p3 = plot(time,
            moment_3,
            xlims=tspan,
            linewidth=3,
            xaxis="time",
            yaxis="M3",
  )         
  plot(p0, p1, p2, p3, layout=(2, 2), legend=false)

  savefig("mixture_dist_linear_kernel_example.png")
end

main()
