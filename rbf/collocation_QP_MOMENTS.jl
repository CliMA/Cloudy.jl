"Constant coalescence kernel example"

using Plots
using Cloudy.BasisFunctions
using Cloudy.Collocation
using Cloudy.ParticleDistributions
using Cloudy.KernelTensors
using Cloudy.Sources
using QuadGK
using DifferentialEquations

function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-8

  # Physicsal parameters
  coalescence_coeff = 1/3.14/4
  kernel_func = x -> coalescence_coeff


  ################## COLLOCATION APPROACH ###################
  # Initial condition: gamma distribution
  gamma_dist = GammaPrimitiveParticleDistribution(150.0, 6.466666667, 0.03092815)
  dist_init = x -> density_eval(gamma_dist, x)

  # Choose the basis functions
  Nb = 10
  mu_start = 0.1
  mu_stop = 25.0
  #rbf_mu = collect(range(mu_start, stop=mu_stop, length=Nb))
  rbf_mu = exp.(collect(range(log(mu_start), stop = log(mu_stop), length=Nb)))
  rbf_sigma = repeat([mu_start/2], Nb)
  basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
  for i = 1:Nb
    basis[i] = GaussianBasisFunction(rbf_mu[i], rbf_sigma[i])
  end

  # Precompute the various matrices
  A = get_rbf_inner_products(basis)
  Source = get_kernel_rbf_source(basis, rbf_mu, kernel_func)
  Sink = get_kernel_rbf_sink(basis, rbf_mu, kernel_func)
  mass_cons = get_mass_cons_term(basis)
  (c0, mass) = get_IC_vec(dist_init, basis, A, mass_cons)
  println(c0)

  println("Initial mass: ", mass, ";  initial collocation mass: ", mass_cons'*c0)
  println("Percent error: ", (mass_cons'*c0 - mass)/mass*100, "%")

  # set up the explicit time stepper
  tspan = (0.0, 1.0)
  dt = 1e-2
  tsteps = range(tspan[1], stop=tspan[2], step=dt)
  nj = dist_init.(rbf_mu)
  dndt = ni->collision_coalescence_QP(ni, A, Source, Sink, mass_cons, mass)

  # track the moments
  basis_mom = vcat(get_moment(basis, 0.0)', get_moment(basis, 1.0)', get_moment(basis, 2.0)')
  mom_coll = zeros(FT, length(tsteps)+1, 3)
  mom_coll[1,:] = (basis_mom*c0)'
  println(basis_mom*c0)
  
  for (i,t) in enumerate(tsteps)
    nj += dndt(nj)*dt
    cj = get_constants_vec2(nj, A, mass_cons, mass)
    mom_coll[i+1,:] = (basis_mom*cj)'
  end

  t_coll = collect(tsteps)

  ################## MOMENT-BASED APPROACH ###################
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

  t_mom = sol.t
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = vcat(sol.u'...)[:, 3]

  ##################        PLOTTING        ###################
  pyplot()
  plot(t_mom,
      moment_0,
      linewidth=3,
      title="\$C(m, m') = k\$ gamma_MOM (CLIMA) vs. GP-Collocation with 3 RBF",
      xaxis="time",
      yaxis="M\$_k\$(time)",
      xlims=tspan,
      ylims=(0, 450), 
      label="M\$_0\$ CLIMA"
  )
  plot!(t_mom,
      t-> (1 / moments_init[1] + 0.5 * coalescence_coeff * t)^(-1),
      lw=3,
      ls=:dash,
      label="M\$_0\$ Exact"
  )
  plot!(t_mom,
      moment_1,
      linewidth=3,
      label="M\$_1\$ CLIMA"
  )
  plot!(t_mom,
      t-> moments_init[2],
      lw=3,
      ls=:dash,
      label="M\$_1\$ Exact"
  )
  plot!(t_mom,
      moment_2,
      linewidth=3,
      label="M\$_2\$ CLIMA"
  )
  plot!(t_mom,
      t-> moments_init[3] + moments_init[2]^2 * coalescence_coeff * t,
      lw=3,
      ls=:dash,
      label="M\$_2\$ Exact"
  )
  plot!(t_coll, mom_coll[:,1], lw=3, label="M\$_0\$ RBF")
  plot!(t_coll, mom_coll[:,2], lw=3, label="M\$_1\$ RBF")
  plot!(t_coll, mom_coll[:,3], lw=3, label="M\$_2\$ RBF")
  savefig("rbf/constant_kernel_comparison_expCollX.png")
end

@time main()