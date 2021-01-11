""" collision-coalescence only with constant kernel """

using Plots
using Cloudy.BasisFunctions
using Cloudy.Collocation
using SpecialFunctions: gamma

function main()
    ############################ SETUP ###################################

    # Numerical parameters
    FT = Float64

    # Physical parameters
    K     = x-> 1e-4      # kernel function in cm3 per sec 
    
    N0    = 300           # initial droplet density: number per cm^3
    N     = N0            # total number density of droplets initially
    θ_r   = 10            # radius scale factor: µm
    #θ_v   = 4/3*pi*θ_r^3  # volume scale factor: µm^3
    #θ_v   = 10
    k     = 3             # shape factor for volume size distribution 
    ρ_w   = 1.0e-12       # density of droplets: 1 g/µm^3

    # initial distribution in volume: gamma distribution in radius, number per cm^3
    r = v->(3/4/pi*v)^(1/3)
    n_v_init = v -> N*(r(v))^(k-1)/θ_r^k * exp(-r(v)/θ_r) / gamma(k)

    # basis setup 
    Nb = 4
    rmax  = 100.0
    rmin  = 10.0
    vmin = rmin^3
    vmax = rmax^3
    rbf_mu = select_rbf_locs(vmin, vmax, Nb)
    rbf_sigma = select_rbf_shapes(rbf_mu, smoothing_factor=1.5)
    rbf_k = rbf_mu.^2 ./ rbf_sigma.^2
    rbf_θ = rbf_mu.^2 ./ rbf_sigma
    basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
    for i = 1:Nb
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
    mass_cons = get_mass_cons_term(basis, xstart = v_start, xstop=v_stop)
    (c0, mass) = get_IC_vec(n_v_init, basis, Φ, mass_cons, xstart = v_start, xstop=v_stop)

    ########################### DYNAMICS ################################
    tspan = (0.0, 60.0)
    dt = 1.0
    tsteps = range(tspan[1]+dt, stop=tspan[2], step=dt)
    nj = n_v_init.(rbf_mu)
    dndt = ni->collision_coalescence(ni, Φ, Source, Sink, mass_cons, mass)

    basis_mom = vcat(get_moment(basis, 0.0)', get_moment(basis, 1.0)', get_moment(basis, 2.0)')
    mom_coll = zeros(FT, length(tsteps)+1, 3)
    mom_coll[1,:] = (basis_mom*c0)'

    c05 = c0
    for (i,t) in enumerate(tsteps)
      nj += dndt(nj)*dt
      cj = get_constants_vec(nj, Φ, mass_cons, mass)
      println(cj)

      # save intermediate time step
      if t/tspan[2]==0.5
        c05 = cj
      end

      mom_coll[i+1,:] = (basis_mom*cj)'
    end

    moments_init = mom_coll[1,:]

    c_final = get_constants_vec(nj, Φ, mass_cons, mass)
    plot_nv_result(vmin*0.1, vmax*1.2, basis, c0, c05, c_final, plot_exact=true, n_v_init=n_v_init)
    #plot_nr_result(rmin*0.1, rmax*1.2, basis, c0, c_final, plot_exact=true, n_v_init=n_v_init)
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

function plot_nv_result(vmin::FT, vmax::FT, basis::Array{PrimitiveUnivariateBasisFunc, 1}, 
                        c::Array{FT, 1}...; plot_exact::Bool=false, n_v_init::Function = x-> 0.0) where {FT <: Real}
  v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
  pyplot()
  if plot_exact
    plot(v_plot,
        n_v_init.(v_plot),
        lw=2,
        label="Exact I.C.")
  end
  for (i,cvec) in enumerate(c)
    println(cvec)
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    plot!(v_plot,
        n_plot,
        lw=2,
        ylim=[1e-4, 1],
        xlabel="volume, µm^3",
        ylabel="number",
        xaxis=:log,
        yaxis=:log,
        label=string("time ", i)
    )
  end

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
          ylabel="number",
          xaxis=:log,
          yaxis=:log,
          ylim=[1e-4, 1e5])
  end
  savefig("rbf_paper/temp.png")
end

function plot_nv_result(vmin::FT, vmax::FT, basis::Array{PrimitiveUnivariateBasisFunc, 1}, 
                        c::Array{FT, 1}...; plot_exact::Bool=false, n_v_init::Function = x-> 0.0) where {FT <: Real}
  v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
  pyplot()
  if plot_exact
    plot(v_plot,
        n_v_init.(v_plot),
        lw=2,
        label="Exact I.C.")
  end
  for (i,cvec) in enumerate(c)
    println(cvec)
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    plot!(v_plot,
        n_plot,
        lw=2,
        #ylim=[1e-4, 1],
        xlabel="volume, µm^3",
        ylabel="number",
        #xaxis=:log,
        #yaxis=:log,
        label=string("time ", i)
    )
  end

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
          xaxis=:log,
          yaxis=:log,
          ylim=[1e-4, 1e5])
  end
  savefig("rbf_paper/temp.png")
end

function plot_moments(tsteps::Array{FT}, moments::Array{FT, 2}) where {FT <: Real}
  pyplot()
  plot(tsteps,
        moments[:,1],
        lw=2,
        xlabel="time, sec",
        ylabel="number / cm^3",
        label="M_0")
  for i=1:2
    plot!(tstepts, 
          moments[:,i],
          label=string("M_",i))
  end
end

main()
