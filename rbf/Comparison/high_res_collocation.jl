"Constant coalescence kernel example"

using Plots
using Cloudy.BasisFunctions
using Cloudy.Collocatio
using NonNegLeastSquares
using QuadGK

function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-8

  # Physicsal parameters
  coalescence_coeff = 8e-7
  kernel_func = x -> coalescence_coeff*(x[1]+x[2])

  # Initial condition
  N0 = 1e5
  mu = 15.0
  sigma = 5.0
  dist_init = x-> N0/sigma/sqrt(2*pi)*exp(-(x-mu)^2/2/sigma^2)

  # Choose the basis functions: exponentially space, match IC exactly
  Nb = 40
  xmax = 300.0
  rbf_mu = exp.(range(log(mu), stop=log(180.0), length=Nb) |>collect)
  rbf_sigma = append!([5.0], (rbf_mu[2:end]-rbf_mu[1:end-1])/1.5)
  basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
  for i = 1:Nb
    basis[i] = GaussianBasisFunction(rbf_mu[i], rbf_sigma[i])
  end

  #println("Basis:  ", basis)

  # Precompute the various matrices
  A = get_rbf_inner_products(basis)
  Source = get_kernel_rbf_source(basis, rbf_mu, kernel_func)
  Sink = get_kernel_rbf_sink(basis, rbf_mu, kernel_func)
  mass_cons = get_mass_cons_term(basis)
  (c0, mass) = get_IC_vec(dist_init, basis, A, mass_cons)

  #println(c0)
  #println("Mass: ", mass, ";  initial collocation mass: ", mass_cons'*c0)
  #println("Percent error: ", (mass_cons'*c0 - mass)/mass*100, "%")

  # set up the explicit time stepper
  tspan = (0.0, 1.0)
  dt = 1e-3
  tsteps = range(tspan[1]+dt, stop=tspan[2], step=dt)
  nj = dist_init.(rbf_mu)
  dndt = ni->collision_coalescence(ni, A, Source, Sink, mass_cons, mass)

  basis_mom = vcat(get_moment(basis, 0.0)', get_moment(basis, 1.0)', get_moment(basis, 2.0)')
  mom_coll = zeros(FT, length(tsteps)+1, 3)
  mom_coll[1,:] = (basis_mom*c0)'

  c05 = c0
  for (i,t) in enumerate(tsteps)
    nj += dndt(nj)*dt
    cj = get_constants_vec(nj, A, mass_cons, mass)

    # save intermediate time step
    if t==0.5
      c05 = cj
    end

    mom_coll[i+1,:] = (basis_mom*cj)'
  end

  moments_init = mom_coll[1,:]

  c_final = get_constants_vec(nj, A, mass_cons, mass)


  #################### PLOTTING  ####################
  pyplot()
  t_coll = append!([0.0], tsteps)
  time = t_coll

  # Plot the moments and compare to analytical solution
  plot(t_coll, 
      mom_coll[:,1]/mom_coll[1,1], 
      lw=3, 
      title="\$C(m, m') = k\$ (Smolu. 1916) vs. Climate Machine",
      xaxis="time",
      yaxis="M\$_k\$(time)/M\$_k\$(0)",
      xlims=tspan,
      ylims=(0,10),
      label="M\$_0\$ RBF",
      )
  plot!(time,
      t-> ((1 / moments_init[1] + 0.5 * coalescence_coeff * t)^(-1))/moments_init[1],
      lw=3,
      ls=:dash,
      label="M\$_0\$ Exact"
  )
  plot!(t_coll, mom_coll[:,2]/mom_coll[1,2], lw=3, label="M\$_1\$ RBF")
  plot!(time,
    t-> 1,
    lw=3,
    ls=:dash,
    label="M\$_1\$ Exact"
  )
  plot!(t_coll, mom_coll[:,3]/mom_coll[1,3], lw=3, label="M\$_2\$ RBF")
  plot!(time,
      t-> (moments_init[3] + moments_init[2]^2 * coalescence_coeff * t)/moments_init[3],
      lw=3,
      ls=:dash,
      label="M\$_2\$ Exact"
  )
  savefig("rbf/Comparison/COL_constant_kernel_gaussian_lowres.png")


  # Plot the distributions
  x = collect(range(eps(), stop=250.0, length=1000))
  # initial
  plot(x, 
    evaluate_rbf(basis, c0, x)/N0, 
    lw=2,
    title="\$C(m, m') = k\$ (Smolu. 1916) vs. Climate Machine",
    xaxis="mass",
    yaxis="Probability distribution (normalized)",
    label="t = 0")

  # intermediate time
  plot!(x, 
    evaluate_rbf(basis, c05, x)/sum(c05), 
    label="t = 0.5",
    lw=2)

  # final distribution
  plot!(x, 
    evaluate_rbf(basis, c_final, x)/sum(c_final), 
    label="t = 1.0",
    lw=2)

  # initial exact and collocation pts
  plot!(x, 
    dist_init.(x)/N0,
    lw=2,
    ls=:dash,
    label="Exact I.C."
  )
  scatter!(rbf_mu, 
    zeros(FT, Nb), 
    markershape=:circle,
    label="Collocation points")

  # RBF backgrounds
  for i=1:Nb
    ci = zeros(FT, Nb)
    ci[i] = 1.0
    plot!(x, 
    evaluate_rbf(basis, ci, x),
      lw=1,
      ls=:dot,
      color=:gray,
      label="Basis fn")
  end

  savefig("rbf/Comparison/COL_constant_kernel_gaussiandist_lowres.png")
end

@time main()