using Plots
using Cloudy.BasisFunctions
using Cloudy.LOGCollocation
using QuadGK

function main()
  ############################ SETUP ###################################

    # Numerical parameters
    FT = Float64

    # Physical parameters
    K     = x-> 1e-4      # kernel function in cm3 per sec 
    
    # Initial distribution
    N0    = 300           # initial droplet density: number per cm^3
    θ_v   = 1000          # volume scale factor: µm^3
    k     = 3             # shape factor for volume size distribution 
    ρ_w   = 1.0e-12       # density of droplets: 1 g/µm^3

    # initial distribution in volume: gamma distribution, number per cm^3
    n_v_init = v-> N0*v^(k-1)/θ_v^k * exp(-v/θ_v) / gamma(k)

    # basis setup
    vmin = 500
    vmax = 4e6
    Nb = 4
    ζ = vmax/vmin
    rbf_mu = select_rbf_locs(Nb)
    rbf_sigma = select_rbf_shapes(rbf_mu, smoothing_factor=1.5)
    basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
    for i = 1:Nb
      basis[i] = GaussianBasisFunction(rbf_mu[i], rbf_sigma[i])
      println(basis[i])
    end


    ################## COLLOCATION APPROACH ###################

    # Precompute the various matrices
    A = get_rbf_inner_products(basis)
    Source = get_kernel_rbf_source(basis, rbf_mu_z, kernel_func, zstart=zstart)
    Sink = get_kernel_rbf_sink(basis, rbf_mu_z, kernel_func, zstart=zstart, zstop=zstop)
    mass_cons = get_moment_log(basis, 1.0, zstart=zstart, zstop=zstop)
    (c0, mass) = get_IC_vec(dist_init, basis, A, mass_cons, xstart=exp(zstart), xstop=exp(zstop))
    println("precomputation complete")

    # set up the explicit time stepper
    tspan = (0.0, 0.05)
    dt = 1e-2
    tsteps = range(tspan[1], stop=tspan[2], step=dt)
    nj = dist_init.(rbf_mu_x)
    dndt = ni->collision_coalescence(ni, A, Source, Sink, mass_cons, mass)

    # track the moments
    basis_mom = vcat(get_moment_log(basis, 0.0, zstart=zstart, zstop=zstop)', get_moment_log(basis, 1.0, zstart=zstart, zstop=zstop)', get_moment_log(basis, 2.0, zstart=zstart, zstop=zstop)')
    mom_coll = zeros(FT, length(tsteps)+1, 3)
    mom_coll[1,:] = (basis_mom*c0)'
    moments_init = mom_coll[1,:]

    cj = c0

    for (i,t) in enumerate(tsteps)
      println(cj)
      nj += dndt(nj)*dt
      cj = get_constants_vec(nj, A, mass_cons, mass)
      mom_coll[i+1,:] = (basis_mom*cj)'
    end

    t_coll = collect(tsteps)

  ############################### PLOTTING ####################################
    # plot the actual distribution
    z = collect(range(zstart/5, stop=zstop/5, length=1000))
    x = exp.(z)
    plot(x, 
      dist_init.(x)/N,
      linewidth=2,
      title="\$C(m, m') = k\$ Collocation Method: Lognormal Basis",
      xaxis="mass",
      yaxis="Probability distribution (normalized)",
      label="t = 0"
    )

    plot!(x, 
      evaluate_rbf(basis, c0, z)/sum(c0).*exp.(-z),
      linewidth=2,
      label="t = 0"
    )
    
    plot!(x,
      evaluate_rbf(basis, cj, z)/sum(cj).*exp.(-z),
      linewidth=2,
      label="t = 1"
    )

    for i=1:Nb
      c_basis = zeros(FT,Nb)
      c_basis[i] = 1
      plot!(x,
        evaluate_rbf(basis, c_basis, z).*exp.(-z),
        ls=:dash,
        linecolor=:gray,
        label="basis_fn")
    end
  
    savefig("rbf/LOGRBF_dist.png")
  
    # plot the moments
  pyplot()
  plot(t_coll,
  t-> (moments_init[1]*exp(-b*moments_init[2]*t))/moments_init[1],
      linewidth=3,
      title="\$C(m, m') = k\$ (Smolu. 1916) vs. GP-Collocation with 3 Lognormal RBF",
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
  savefig("rbf/LOGRBF.png")

  # print out the final moment and the initial and final distribution parameters
  println("Initial moments: ", mom_coll[1,:])
  println("Final moments: ", mom_coll[end,:])
  println("Initial distribution constants: ", c0)
  println("Normalized: ", c0/sum(c0))
  println("Final distribution constants: ", cj)
  println("Normalized: ", cj/sum(cj))
  
end

@time main()