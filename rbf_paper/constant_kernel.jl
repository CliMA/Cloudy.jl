""" collision-coalescence only with constant kernel """

using Plots
using Cloudy.BasisFunctions
using Cloudy.Collocation
using QuadGK
using SpecialFunctions: gamma
using DifferentialEquations

function main()
    ############################ SETUP ###################################

    # Numerical parameters
    FT = Float64

    # Physical parameters
    K     = x-> 1e-4      # kernel function in cm3 per sec 
    N0    = 300           # initial droplet density: number per cm^3
    θ_r   = 20.0          # radius scale factor: µm
    θ_v   = 4/3*pi*θ_r^3  # volume scale factor: µm^3
    k     = 2             # shape factor for volume size distribution 
    ρ_w   = 1.0e-12       # density of droplets: 1 g/µm^3

    μ_r   = k*θ_r
    σ_r   = sqrt(k)*θ_r

    # initial distribution in volume: gamma distribution in radius, number per cm^3
    #r = v->(3/4/pi*v)^(1/3)
    #drdv = v-> 1/4/pi/(r(v)^2)
    #n_r_init = R -> N0*(R^(k-1))/θ_r^k * exp(-R/θ_r) / gamma(k)
    #n_v_init = v -> n_r_init(r(v))*drdv(v)
    #n_v_init = v -> N0/σ_r/2/pi*exp(-(r(v) - μ_r)^2/2/σ_r^2)*drdv(v)
    n_v_init = v -> N0*v^(k-1)/θ_v^k * exp(-v / θ_v) / gamma(k)

    # basis setup 
    Nb = 5
    rmax  = 200.0
    rmin  = 20.0
    vmin = 4/3*pi*rmin^3
    vmax = 4/3*pi*rmax^3
    rbf_mu = select_rbf_locs(vmin, vmax, Nb)
    rbf_sigma = select_rbf_shapes(rbf_mu, smoothing_factor=1.2)
    #rbf_mu = select_rbf_locs(rmin, rmax, Nb)
    #rbf_sigma = select_rbf_shapes(rbf_mu, smoothing_factor=1.2)
    #rbf_k = rbf_mu.^2 ./ rbf_sigma.^2
    #rbf_θ = rbf_mu.^2 ./ rbf_sigma
    basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
    for i = 1:Nb
      #basis[i] = GaussianBasisFunctionCubeRoot(rbf_mu[i], rbf_sigma[i])
      basis[i] = GaussianBasisFunction(rbf_mu[i], rbf_sigma[i])
      #basis[i] = GammaBasisFunction(rbf_k[i], rbf_θ[i])
      println(basis[i])
    end

    ########################### PRECOMPUTATION ################################
    v_start = eps()
    v_stop = vmax

    # Precomputation
    Φ = get_rbf_inner_products(basis, rbf_mu)
    Source = get_kernel_rbf_source(basis, rbf_mu, K, xstart = v_start)
    Sink = get_kernel_rbf_sink(basis, rbf_mu, K, xstart = v_start, xstop=v_stop)
    first_moments = get_mass_cons_term(basis, xstart = v_start, xstop=v_stop)
    #c0 = get_IC_vec(n_v_init, basis, rbf_mu, Φ)
    (c0, mass) = get_IC_vec(n_v_init, basis, rbf_mu, Φ, first_moments, xstart = v_start, xstop=v_stop)
    
    ########################### DYNAMICS ################################
    nj_init = n_v_init.(rbf_mu)
    tspan = (0.0, 3600.0)


    # Implicit time step
    function dndt(ni,t,p)
      return collision_coalescence(ni, Φ, Source, Sink, first_moments, mass)
      #return collision_coalescence(ni, Φ, Source, Sink)
    end
    prob = ODEProblem(dndt, nj_init, tspan)
    sol = solve(prob)
    println(sol)

    t_coll = range(tspan[1], stop=tspan[2], length=10)

    # track the moments and constants
    basis_mom = vcat(get_moment(basis, 0.0, xstart=v_start, xstop=v_stop)', get_moment(basis, 1.0, xstart=v_start, xstop=v_stop)', get_moment(basis, 2.0, xstart=v_start, xstop=v_stop)')
    c_coll = zeros(FT, length(t_coll), Nb)
    for (i,t) in enumerate(t_coll)
      nj_t = sol(t)
      c_coll[i,:] = get_constants_vec(nj_t, Φ, first_moments, mass)
      #c_coll[i,:] = get_constants_vec(nj_t, Φ)
    end
    mom_coll = (c_coll*basis_mom')
    println("Initial constants: ", c0)
    println("Final constants: ", c_coll[end,:])

    plot_nv_result(vmin*0.1, vmax*0.05, basis, c0, c_coll[Int(end/2),:], c_coll[end,:], plot_exact=true, n_v_init=n_v_init)  
    #plot_nr_result(rmin*0.1, rmax*1.2, basis, c0, c_coll[end,:], plot_exact=true, n_v_init=n_v_init) 
    plot_moments(collect(t_coll), mom_coll, plot_exact=true, constant_a = 1e-4)
end





function plot_init()
  # often plotted g(ln r) = 3x^2*n(x,t); mass per m^3 per unit log r
  g_lnr_init = r-> 3*(4*pi/3*r^3)^2*n_v_init(4*pi/3*r^3)*ρ_w

  # PLOT INITIAL MASS DISTRIBUTION: should look similar to Fig 10 from Long 1974
  pyplot()
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
  pyplot()
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
  pyplot()
  if plot_exact
    plot(r_plot,
          n_v_init.(v_plot),
          lw=2,
          label="Exact")
  end
  for cvec in c
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    pyplot()
    plot!(r_plot,
          n_plot,
          lw=2,
          xlabel="radius, µm",
          ylabel="number",
          #xaxis=:log,
          #yaxis=:log,
          ylim=[1e-4, 1e5])
  end
  savefig("rbf_paper/temp.png")
end

function plot_nv_result(vmin::FT, vmax::FT, basis::Array{PrimitiveUnivariateBasisFunc, 1}, 
                        c::Array{FT, 1}...; plot_exact::Bool=false, n_v_init::Function = x-> 0.0, plot_basis::Bool=false) where {FT <: Real}
  v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
  pyplot()
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
        c_basis[i] = 1.0 / evaluate_rbf(basis[i], get_moment(basis[i], 1.0))
        plot!(v_plot,
          evaluate_rbf(basis, c_basis, v_plot),
          ls=:dash,
          linecolor=:gray,
          label="basis_fn")
    end
  end
  savefig("rbf_paper/temp.png")
  savefig("rbf_paper/temp.png")
end

function plot_nr_result(rmin::FT, rmax::FT, basis::Array{PrimitiveUnivariateBasisFunc, 1}, c::Array{FT, 1}...;
                        plot_exact::Bool=false, n_v_init::Function = x-> 0.0) where {FT <: Real}
  r_plot = exp.(collect(range(log(rmin), stop=log(rmax), length=1000)))
  v_plot = 4/3*pi*r_plot.^3
  pyplot()
  if plot_exact
    plot(r_plot,
          n_v_init.(v_plot),
          lw=2,
          label="Exact")
  end
  for cvec in c
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    pyplot()
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
  savefig("rbf_paper/temp.png")
end

function plot_moments(tsteps::Array{FT}, moments::Array{FT, 2}; plot_exact::Bool=false, constant_a::FT = 0.0) where {FT <: Real}
  M0_0 = moments[1,1]
  M1_0 = moments[1,2]
  M2_0 = moments[1,3]
  moments_init = [M0_0, M1_0, M2_0]
  pyplot()
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
            label="exact")
    elseif (plot_exact && i==2)
      plot!(tsteps,
            t-> moments_init[2],
            lw=2,
            label="exact")
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
