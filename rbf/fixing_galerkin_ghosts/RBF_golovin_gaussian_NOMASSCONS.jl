"Golovin coalescence kernel example"

using Plots
using Cloudy.BasisFunctions
using Cloudy.Collocation
using QuadGK

function main()
  # Numerical parameters
  FT = Float64

  # Physical parameters: Kernel
  b = 8e-7
  kernel_func = x -> b*(x[1]+x[2])


  ################## COLLOCATION APPROACH ###################
  # Initial condition
  N = 1e5
  mu = 15.0
  sigma = 5.0
  dist_init = x-> N/sigma/sqrt(2*pi)*exp(-(x-mu)^2/2/sigma^2)

  # Choose the basis functions
  Nb = 10
  xmax = 300.0
  rbf_mu = exp.(range(log(mu), stop=log(xmax), length=Nb) |>collect)
  rbf_sigma = append!([5.0], (rbf_mu[2:end]-rbf_mu[1:end-1])/1.5)
  rbf_sigma[1] = sigma
  basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
  for i = 1:Nb
    basis[i] = GaussianBasisFunction(rbf_mu[i], rbf_sigma[i])
  end
  println("mu", rbf_mu)
  println("sigma", rbf_sigma)

  # Precompute the various matrices
  A = get_rbf_inner_products(basis)
  Source = get_kernel_rbf_source(basis, rbf_mu, kernel_func)
  Sink = get_kernel_rbf_sink(basis, rbf_mu, kernel_func)
  c0 = get_IC_vec(dist_init, basis, A)

  # set up the explicit time stepper
  tspan = (0.0, 1.0)
  dt = 1e-2
  tsteps = range(tspan[1], stop=tspan[2], step=dt)
  nj = dist_init.(rbf_mu)
  dndt = ni->collision_coalescence(ni, A, Source, Sink)

  # track the moments
  basis_mom = vcat(get_moment(basis, 0.0)', get_moment(basis, 1.0)', get_moment(basis, 2.0)')
  mom_coll = zeros(FT, length(tsteps)+1, 3)
  mom_coll[1,:] = (basis_mom*c0)'
  moments_init = mom_coll[1,:]

  cj = c0


  for (i,t) in enumerate(tsteps)
    nj += dndt(nj)*dt
    cj = get_constants_vec(nj, A)
    mom_coll[i+1,:] = (basis_mom*cj)'
  end

  t_coll = collect(tsteps)

  ############################### PLOTTING ####################################
    # plot the actual distribution
    x = collect(range(eps(), stop=250.0, length=1000))
    plot(x, 
      evaluate_rbf(basis, c0, x)/sum(c0),
      linewidth=2,
      title="\$C(m, m') = k\$ Collocation Method",
      xaxis="mass",
      yaxis="Probability distribution (normalized)",
      label="t = 0"
    )
    
    plot!(x,
      evaluate_rbf(basis, cj, x)/sum(cj),
      linewidth=2,
      label="t = 1"
    )

    for i=1:Nb
      c_basis = zeros(FT,Nb)
      c_basis[i] = 1
      plot!(x,
        evaluate_rbf(basis, c_basis, x),
        ls=:dash,
        linecolor=:gray,
        label="basis_fn")
    end
  
    savefig("rbf/fixing_galerkin_ghosts/NOMASSCONS_golovin_gaussiandist.png")
  
    # plot the moments
  pyplot()
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

  plot!(t_coll, mom_coll[1:end-1,1]/moments_init[1], lw=3, label="M\$_0\$ RBF")
  plot!(t_coll, mom_coll[1:end-1,2]/moments_init[2], lw=3, label="M\$_1\$ RBF")
  plot!(t_coll, mom_coll[1:end-1,3]/moments_init[3], lw=3, label="M\$_2\$ RBF")
  savefig("rbf/fixing_galerkin_ghosts/NOMASSCONS_golovin_gaussian3.png")

  # print out the final moment and the initial and final distribution parameters
  #println("Initial moments: ", mom_coll[1,:])
  #println("Final moments: ", mom_coll[end,:])
  #println("Initial distribution constants: ", c0)
  #println("Normalized: ", c0/sum(c0))
  #println("Final distribution constants: ", cj)
  #println("Normalized: ", cj/sum(cj))
  
end

@time main()