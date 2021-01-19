using Plots
using Cloudy.BasisFunctions
using Cloudy.Collocation
using QuadGK
using SpecialFunctions: gamma

function main()
  # Numerical parameters
  FT = Float64

  # Physical parameters: Kernel
  b = 8e-7
  kernel_func = x -> b*(x[1]+x[2])

  ################## COLLOCATION APPROACH ###################
  # Initial condition: gamma
  N = 1e5
  k=2
  theta=5
  dist_init = x-> N*x^(k-1)*exp(-x/theta)/theta^k/gamma(k)

  # Choose the basis functions
  Nb = 5
  xmin_loc = 10.0
  xmax_loc = 150.0
  rbf_loc = select_rbf_locs(xmin_loc, xmax_loc, Nb)
  rbf_stddev = select_rbf_shapes(rbf_loc)
  basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
  for i = 1:Nb
    basis[i] = GaussianBasisFunction(rbf_loc[i], rbf_stddev[i])
  end
  #basis[Nb] = GaussianBasisFunction(rbf_loc[Nb]*0.9, 5.0)
  println(basis)

  # Precompute the various matrices
  # integration limits:
  x_min = 0.0
  x_max = xmax_loc
  
  # computation
  A = get_rbf_inner_products(basis, rbf_loc)
  first_moments = get_mass_cons_term(basis, xstart=x_min, xstop=x_max)
  # TODO 
  Source = get_kernel_rbf_source(basis, rbf_loc, kernel_func, xstart=x_min)
  Sink = get_kernel_rbf_sink(basis, rbf_loc, kernel_func, xstart=x_min, xstop=x_max)

  # INITIAL CONDITION
  # mass conserving:
  (c0, mass) = get_IC_vec(dist_init, basis, rbf_loc, A, first_moments, xstart=x_min, xstop=x_max)

  # non mass conserving
  #c0 = get_IC_vec(dist_init, basis, rbf_loc, A)

  println("precomputation complete")

  # set up the explicit time stepper
  tspan = (0.0, 10.0)
  dt = 1e-2
  tsteps = range(tspan[1], stop=tspan[2], step=dt)
  nj = dist_init.(rbf_loc)
  # mass conserving:
  dndt = ni->collision_coalescence(ni, A, Source, Sink, first_moments, mass)
  # non mass conserving
  # dndt = ni->collision_coalescence(ni, A, Source, Sink)

  # track the moments
  basis_mom = vcat(get_moment(basis, 0.0, xstart=x_min, xstop=x_max)', get_moment(basis, 1.0, xstart=x_min, xstop=x_max)', get_moment(basis, 2.0, xstart=x_min, xstop=x_max)')
  mom_coll = zeros(FT, length(tsteps)+1, 3)
  mom_coll[1,:] = (basis_mom*c0)'
  moments_init = mom_coll[1,:]

  cj = c0

  for (i,t) in enumerate(tsteps)
    nj += dndt(nj)*dt
    # mass conserving
    cj = get_constants_vec(nj, A, first_moments, mass)
    mom_coll[i+1,:] = (basis_mom*cj)'
  end

  t_coll = collect(tsteps)

  ############################### PLOTTING ####################################
    # plot the actual distribution
    x = collect(range(eps(), stop=250.0, length=1000))
    plot(x, 
      evaluate_rbf(basis, c0, x)/sum(c0),
      linewidth=2,
      title="Golovin; Collocation truncated integral",
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
  
    savefig("rbf/golovin_mass_conserving.png")
  
    # plot the moments
  pyplot()
  plot(t_coll,
  t-> (moments_init[1]*exp(-b*moments_init[2]*t))/moments_init[1],
      linewidth=3,
      title="Golovin; Collocation truncated integral",
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
  savefig("rbf/golovin_cons_moments.png")

  # print out the final moment and the initial and final distribution parameters
  println("Initial moments: ", mom_coll[1,:])
  println("Final moments: ", mom_coll[end,:])
  println("Initial distribution constants: ", c0)
  println("Normalized: ", c0/sum(c0))
  println("Final distribution constants: ", cj)
  println("Normalized: ", cj/sum(cj))
  
end

@time main()