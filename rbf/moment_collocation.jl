using Plots
using Cloudy.BasisFunctions
using Cloudy.MomentCollocation
using QuadGK
using SpecialFunctions: gamma
using DifferentialEquations

function main()
  # Numerical parameters
  FT = Float64

  # Physical parameters: Kernel
  b = 1e-4
  kernel_func = x -> b*(x[1]+x[2])
  tracked_moments = [0.0, 1.0]

  ################## COLLOCATION APPROACH ###################
  # Initial condition: gamma
  N = 300
  k=2
  theta=5
  dist_init = x-> N*x^(k-1)*exp(-x/theta)/theta^k/gamma(k)

  # Choose the basis functions
  Nb = 5
  xmin_loc = 10.0
  xmax_loc = 300.0
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
  A = get_rbf_inner_products(basis, rbf_loc, tracked_moments)
  Source = get_kernel_rbf_source(basis, rbf_loc, tracked_moments, kernel_func, xstart=x_min)
  Sink = get_kernel_rbf_sink(basis, rbf_loc, tracked_moments, kernel_func, xstart=x_min, xstop=x_max)

  # INITIAL CONDITION
  (c0, nj_init) = get_IC_vecs(dist_init, basis, rbf_loc, A, tracked_moments)
  println(c0, nj_init)
  println("precomputation complete")

  # Implicit Time stepping
  tspan = (0.0, 10.0)
  
  function dndt(ni,t,p)
    return collision_coalescence(ni, A, Source, Sink)
  end

  prob = ODEProblem(dndt, nj_init, tspan)
  sol = solve(prob)
  #println(sol)

  t_coll = sol.t

  # track the moments
  basis_mom = vcat(get_moment(basis, 0.0, xstart=x_min, xstop=x_max)', get_moment(basis, 1.0, xstart=x_min, xstop=x_max)', get_moment(basis, 2.0, xstart=x_min, xstop=x_max)')
  c_coll = zeros(FT, length(t_coll)+1, Nb)
  c_coll[1,:] = c0
  for (i,t) in enumerate(t_coll)
    nj_t = sol(t)
    c_coll[i+1,:] = get_constants_vec(nj_t, A)
  end
  
  mom_coll = (basis_mom*c_coll')'
  #println(mom_coll)
  moments_init = mom_coll[1,:]

  ############################### PLOTTING ####################################
    # plot the actual distribution
    x = collect(range(eps(), stop=30.0, length=1000))
    plot(x, 
      evaluate_rbf(basis, c0, x)/sum(c0),
      linewidth=2,
      title="Golovin; Collocation truncated integral",
      xaxis="mass",
      yaxis="Probability distribution (normalized)",
      label="t = 0"
    )
    
    plot!(x,
      evaluate_rbf(basis, c_coll[end,:], x)/sum(c_coll[end,:]),
      linewidth=2,
      label="t = 10.0"
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
  
    savefig("rbf/moment_hybrid.png")
  
    # plot the moments
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
  savefig("rbf/moment_hybrid_moments.png")

  # print out the final moment and the initial and final distribution parameters
  println("Initial moments: ", mom_coll[1,:])
  println("Final moments: ", mom_coll[end,:])
  println("Initial distribution constants: ", c0)
  println("Normalized: ", c0/sum(c0))
  println("Final distribution constants: ", c_coll[end,:])
  println("Normalized: ", c_coll[end,:]/sum(c_coll[end,:]))
  
end

@time main()