using Plots
using Cloudy.BasisFunctions
using Cloudy.MomentCollocation
using QuadGK
using SpecialFunctions: gamma
using DifferentialEquations

function main()
  # Numerical parameters
  FT = Float64

  # Physical parameters
  K     = x-> 1e-4      # kernel function in cm3 per sec 
  N0    = 300           # initial droplet density: number per cm^3
  θ_r   = 10.0          # radius scale factor: µm
  θ_v   = 4/3*pi*θ_r^3  # volume scale factor: µm^3
  k     = 2             # shape factor for volume size distribution 
  ρ_w   = 1.0e-12       # density of droplets: 1 g/µm^3

  # initial distribution in volume: gamma distribution in radius, number per cm^3
  n_v_init = v -> N0*v^(k-1)/θ_v^k * exp(-v / θ_v) / gamma(k)

  # basis setup 
  Nb = 10
  rmax  = 20.0
  rmin  = 10.0
  vmin = 4/3*pi*rmin^3
  vmax = 4/3*pi*rmax^3
  #rbf_loc = select_rbf_locs(vmin, vmax, Nb)
  #rbf_sigma = select_rbf_shapes(rbf_mu, smoothing_factor=1.2)
  rbf_loc = collect(range(vmin, stop=vmax, length=Nb))
  rbf_stddev = (rbf_loc[end] - rbf_loc[end-1])/1.5*ones(Nb)
  rbf_stddev[1] = min(rbf_stddev[1], rbf_loc[1]/1.5)

  basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
  for i = 1:Nb
    #basis[i] = GaussianBasisFunctionCubeRoot(rbf_mu[i], rbf_sigma[i])
    basis[i] = GaussianBasisFunction(rbf_loc[i], rbf_stddev[i])
    #basis[i] = GammaBasisFunction(rbf_k[i], rbf_θ[i])
    println(basis[i])
  end
  tracked_moments = [1.0]

  # Precompute the various matrices
  # integration limits:
  x_min = eps()
  x_max = vmax
  
  # computation
  A = get_rbf_inner_products(basis, rbf_loc, tracked_moments)
  Source = get_kernel_rbf_source(basis, rbf_loc, tracked_moments, kernel_func, xstart=x_min)
  Sink = get_kernel_rbf_sink(basis, rbf_loc, tracked_moments, kernel_func, xstart=x_min, xstop=x_max)

  # INITIAL CONDITION
  (c0, nj_init) = get_IC_vecs(n_v_init, basis, rbf_loc, A, tracked_moments)

  println("precomputation complete")

  # Implicit Time stepping
  tspan = (0.0, 1.0)
  println(nj_init)
  function dndt(ni,t,p)
    return collision_coalescence(ni, A, Source, Sink)
  end

  prob = ODEProblem(dndt, nj_init, tspan)
  sol = solve(prob)
  println(sol)

  t_coll = sol.t

  ############################### PLOTTING ####################################
    # track the moments and constants
    v_start = eps()
    v_stop = vmax

    basis_mom = vcat(get_moment(basis, 0.0, xstart=v_start, xstop=v_stop)', get_moment(basis, 1.0, xstart=v_start, xstop=v_stop)', get_moment(basis, 2.0, xstart=v_start, xstop=v_stop)')
    c_coll = zeros(FT, length(t_coll), Nb)
    for (i,t) in enumerate(t_coll)
      nj_t = sol(t)
      #c_coll[i,:] = get_constants_vec(nj_t, Φ, first_moments, mass)
      c_coll[i,:] = get_constants_vec(nj_t, A)
    end
    mom_coll = (c_coll*basis_mom')
    moments_init = mom_coll[1,:]

    plot_nv_result(vmin*0.1, vmax*5, basis, c0, c_coll[end,:], plot_exact=true, n_v_init=n_v_init)  
    #plot_nr_result(rmin*0.1, rmax*1.2, basis, c0, c_coll[end,:], plot_exact=true, n_v_init=n_v_init) 

    M_0_exact = zeros(FT, 3)
    for p=1:3
        M_0_exact[p] = quadgk(x->n_v_init(x)*x^(p-1), v_start, v_stop)[1]
    end
    plot_moments(collect(t_coll), mom_coll, plot_exact=true, moments_init = M_0_exact, constant_a = 1e-4)
end

function plot_init()
    # often plotted g(ln r) = 3x^2*n(x,t); mass per m^3 per unit log r
    g_lnr_init = r-> 3*(4*pi/3*r^3)^2*n_v_init(4*pi/3*r^3)*ρ_w
  
    # PLOT INITIAL MASS DISTRIBUTION: should look similar to Fig 10 from Long 1974
    r_plot = collect(range(0, stop=50.0, length=100))
    plot(r_plot, 
        g_lnr_init.(r_plot),
        linewidth=2,
        title="Initial distribution",
        ylabel="mass [gram /m^3 / unit log(r)",
        xaxis="r (µm)",
        xlim=[6, 25]
      )
    savefig("rbf_paper/initial_dist.png")
  
    # PLOT INITIAL DISTRIBUTION: should look similar to Tzivion 1987 fig 1
    r_plot = collect(range(0, stop=100.0, length=100))
    plot(r_plot, 
        n_v_init.(r_plot.^3*4*pi/3),
        linewidth=2,
        title="Initial distribution",
        ylabel="number /m^3 ",
        xlabel="r (µm)",
        xlim=[1, 100],
        ylim=[1e-2, 1e4],
        xaxis=:log,
        yaxis=:log
      )
    savefig("rbf_paper/initial_dist.png")
  end
  
  function plot_nr_result(rmin::FT, rmax::FT, basis::Array{PrimitiveUnivariateBasisFunc, 1}, c::Array{FT, 1}...;
                          plot_exact::Bool=false, n_v_init::Function = x-> 0.0) where {FT <: Real}
    r_plot = exp.(collect(range(log(rmin), stop=log(rmax), length=1000)))
    v_plot = 4/3*pi*r_plot.^3
    if plot_exact
      plot(r_plot,
            n_v_init.(v_plot),
            lw=2,
            label="Exact")
    end
    for cvec in c
      n_plot = evaluate_rbf(basis, cvec, v_plot)
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number",
            #xaxis=:log,
            #yaxis=:log,
            ylim=[1e-4, 1e5])
    end
    savefig("rbf_paper/nr.png")
  end
  
  function plot_nv_result(vmin::FT, vmax::FT, basis::Array{PrimitiveUnivariateBasisFunc, 1}, 
                          c::Array{FT, 1}...; plot_exact::Bool=false, n_v_init::Function = x-> 0.0, plot_basis::Bool=false) where {FT <: Real}
    v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
    if plot_exact
      plot(v_plot,
          n_v_init.(v_plot),
          lw=2,
          label="Exact I.C.")
    end
    for (i,cvec) in enumerate(c)
      n_plot = evaluate_rbf(basis, cvec, v_plot)
      plot!(v_plot,
          n_plot,
          lw=2,
          #ylim=[1e-4, 1],
          xlabel="volume, µm^3",
          ylabel="number",
          #xaxis=:log,
          #yaxis=:log,
          label=string("time ", i-1)
      )
    end
    if plot_basis
      Nb = length(basis)
      for i = 1:Nb
          c_basis = zeros(FT,Nb)
          c_basis[i] = maximum[c[1]]
          plot!(v_plot,
            evaluate_rbf(basis, c_basis, v_plot),
            ls=:dash,
            linecolor=:gray,
            label="basis_fn")
      end
    end
    savefig("rbf_paper/nv.png")
  end
  
  function plot_nr_result(rmin::FT, rmax::FT, basis::Array{PrimitiveUnivariateBasisFunc, 1}, c::Array{FT, 1}...;
                          plot_exact::Bool=false, n_v_init::Function = x-> 0.0) where {FT <: Real}
    r_plot = exp.(collect(range(log(rmin), stop=log(rmax), length=1000)))
    v_plot = 4/3*pi*r_plot.^3
    if plot_exact
      plot(r_plot,
            n_v_init.(v_plot),
            lw=2,
            label="Exact")
    end
    for cvec in c
      n_plot = evaluate_rbf(basis, cvec, v_plot)
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number / cm^3",
            #xaxis=:log,
            #yaxis=:log,
            #ylim=[1e-4, 1e5]
      )
    end
    savefig("rbf_paper/nr.png")
  end
  
  function plot_moments(tsteps::Array{FT}, moments::Array{FT, 2}; plot_exact::Bool=false, moments_init::Array{FT, 1}=[0.0], constant_a::FT = 0.0) where {FT <: Real}
    for i=1:length(moments[1,:])
      plot(tsteps,
            moments[:,i],
            lw=2,
            xlabel="time, sec",
            ylabel=string("M_",i-1))
      if (plot_exact && i==1)
        plot!(tsteps,
              t-> ((1 / moments_init[1] + 0.5 * constant_a * t)^(-1)),
              lw=2,
              label="exact"
              )
      elseif (plot_exact && i==2)
        plot!(tsteps,
              t-> moments_init[2],
              lw=2,
              label="exact",
              ylim=[0.5*moments_init[2], 1.5*moments_init[2]])
      elseif (plot_exact && i==3)
        plot!(tsteps,
          t-> (moments_init[3] + moments_init[2]^2 * constant_a * t),
          lw=2,
          label="exact")
      end
  
      savefig(string("rbf_paper/moments",i-1,".png"))
    end
  end
  
  main()
  