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
  a = 0.05
  b = 0.0
  c = 1.0
  xmin_loc = 1.0
  xmax_loc = 20.0
  kernel_func = x -> a + b*(x[1]+x[2]) + c*abs(x[1]^(2/3)-x[2]^(2/3))/xmax_loc^(2/3)*(x[1]^(1/3)+x[2]^(1/3))^2
  tracked_moments = [1.0]

  # Initial condition
  N = 0.0
  #N2 = 100
  #k=2
  theta=1
  #dist_init = x-> N*x^(k-1)*exp(-x/theta)/theta^k/gamma(k)
  dist_init = x-> N*basis_func(CompactBasisFunction1(1.0, 1.0))(x)
  #dist_init = x-> N*basis_func(CompactBasisFunction1(1.0, 1.0))(x)+N2*basis_func(CompactBasisFunction1(5.0, 1.0))(x)
  
  ################## COLLOCATION APPROACH ###################

  # Choose the basis functions: log spacing
  Nb = 10
  
  # Log-spaced compact basis functions
  """basis = Array{CompactBasisFunc}(undef, Nb)
  rbf_loc = select_rbf_locs(xmin_loc, xmax_loc, Nb)
  rbf_shapes = zeros(Nb)
  rbf_shapes[3:end] = (rbf_loc[3:end] - rbf_loc[1:end-2])
  rbf_shapes[1:2] = rbf_loc[1:2]
  for i = 1:Nb
    basis[i] = CompactBasisFunction1(rbf_loc[i], rbf_shapes[i])
  end"""

  # log-spaced uneven compact rbf
  """basis = Array{CompactBasisFunc}(undef, Nb)
  rbf_loc = select_rbf_locs(xmin_loc, xmax_loc, Nb)
  rbf_shapes = zeros(Nb)
  rbf_shapes[3:end] = (rbf_loc[3:end] - rbf_loc[1:end-2])
  rbf_shapes[1:2] = rbf_loc[1:2]
  rbf_shapes_R = zeros(Nb)
  rbf_shapes_R[1:end-2] = (rbf_loc[3:end] - rbf_loc[1:end-2])
  rbf_shapes_R[end-1:end] = xmax_loc .- rbf_loc[end-1:end]
  rbf_shapes_R[end] = max(rbf_shapes_R[end], 1.0)
  for i = 1:Nb
    basis[i] = CompactBasisFunctionUneven(rbf_loc[i], rbf_shapes[i], rbf_shapes_R[i])
  end"""

  # log-spaced gamma rbf
  """basis = Array{GlobalBasisFunc}(undef, Nb)
  rbf_loc = select_rbf_locs(xmin_loc, xmax_loc, Nb)
  rbf_shapes = zeros(Nb)
  rbf_shapes[3:end] = (rbf_loc[3:end] - rbf_loc[1:end-2])
  rbf_shapes[1:2] = rbf_loc[1:2]
  rbf_k = rbf_loc.^2 ./ rbf_shapes.^2
  rbf_theta = rbf_shapes.^2 ./ rbf_loc
  for i = 1:Nb
    basis[i] = GammaBasisFunction(rbf_k[i], rbf_theta[i])
  end"""

  # lin-spaced log compact rbf
  basis = Array{CompactBasisFunc}(undef, Nb)
  rbf_loc = collect(range(0.0, stop=log(xmax_loc), length=Nb))
  rbf_shapes = zeros(Nb)
  rbf_shapes[3:end] = (rbf_loc[3:end] - rbf_loc[1:end-2])
  rbf_shapes[2] = rbf_loc[2]
  rbf_shapes[1] = rbf_loc[2]
  for i=1:Nb
    basis[i] = CompactBasisFunctionLog(rbf_loc[i], rbf_shapes[i])
  end
  println(basis)
  rbf_loc = exp.(rbf_loc)

  # Injection rate
  inject_rate = [100]#, 200, 100, 50, 10]
  function inject_rate_fn(x)
    f = 0
    for (i, r) in enumerate(inject_rate)
      f = f + r*basis_func(CompactBasisFunction1(i+0.0, 1.0))(x)
    end
    #println(f)
    return f
  end

  # Precompute the various matrices
  # integration limits:
  x_min = 0.0
  x_max = xmax_loc
  
  # computation
  A = get_rbf_inner_products(basis, rbf_loc, tracked_moments)
  Source = get_kernel_rbf_source(basis, rbf_loc, tracked_moments, kernel_func, xstart=x_min)
  Sink = get_kernel_rbf_sink_precip(basis, rbf_loc, tracked_moments, kernel_func, xstart=x_min, xstop=x_max)
  #Inject = get_injection_source(rbf_loc, tracked_moments, inject_rate_fn)
  (c_inject, Inject) = get_basis_projection(basis, rbf_loc, A, tracked_moments, inject_rate_fn, x_max)
  J = get_mass_cons_term(basis, xstart = x_min, xstop = x_max)
  m_inject = sum(c_inject .* J)
  println(c_inject)
  println(Inject)

  # INITIAL CONDITION
  #(c0, nj_init) = get_IC_vecs(dist_init, basis, rbf_loc, A, tracked_moments)
  (c0, nj_init) = get_basis_projection(basis, rbf_loc, A, tracked_moments, dist_init, x_max)
  m_init = sum(c0 .* J)
  println("precomputation complete")

  # Implicit Time stepping
  tspan = (0.0, 10.0)
  
  function dndt(ni,t,p)
    return collision_coalescence(ni, A, Source, Sink, Inject)
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
    x = collect(range(eps(), stop=xmax_loc, length=1000))
    plot(x,
      dist_init.(x),#/sum(c0),
      linewidth=2,
      ls=:dash,
      label="Exact I.C.",
      title="Constant kernel with source and sink",
      xaxis="mass",
      yaxis="Probability distribution (normalized)")

    for (i,t) in enumerate(t_coll)
        if mod(i,5) == 0
            plot!(x,
                evaluate_rbf(basis, c_coll[i,:], x),#/sum(c_coll[i,:]),
                linewidth=2,
                label=string("t=",t)
                )
        end
    end
  
    savefig("rbf/moment_hybrid.png")

    # plot final distribution
    plot(x, evaluate_rbf(basis, c_coll[end,:], x),#/sum(c_coll[i,:]),
                linewidth=2,
                label=string("t=",10)
                )
    for i=1:Nb
      c_basis = zeros(FT,Nb)
      c_basis[i] = 0.5
      plot!(x,
        evaluate_rbf(basis, c_basis, x)*sum(c_coll[end,:]),
        ls=:dash,
        linecolor=:gray,
        label="basis_fn")
    end
    savefig("rbf/moment_hybrid_end.png")
  
    # plot the moments
  """plot(t_coll,
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
  )"""

  #plot(t_coll, mom_coll[1:end-1,1]/moments_init[1], lw=3, label="M\$_0\$ RBF")
  plot(t_coll, mom_coll[1:end-1,2], lw=3, label="M\$_1\$ RBF")
  #plot!(t_coll, mom_coll[1:end-1,3]/moments_init[3], lw=3, label="M\$_2\$ RBF")
  savefig("rbf/moment_hybrid_mass.png")

  # print out the final moment and the initial and final distribution parameters
  println("Initial moments: ", mom_coll[1,:])
  println("Final moments: ", mom_coll[end,:])
  println("Initial distribution constants: ", c0)
  #println("Normalized: ", c0/sum(c0))
  println("Final distribution constants: ", c_coll[end,:])
  #println("Normalized: ", c_coll[end,:]/sum(c_coll[end,:]))
  
end

@time main()