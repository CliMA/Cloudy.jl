""" collision-coalescence only with constant kernel """

using Plots
using Cloudy.BasisFunctions
using Cloudy.MomentCollocation
using QuadGK
using SpecialFunctions: gamma
using DifferentialEquations

function main()
    ############################ SETUP ###################################
    casename = "low_res/hydro_3-1"

    # Numerical parameters
    FT = Float64
    tspan = (0.0, 60.0)

    # basis setup 
    Nb = 3
    rmax  = 50.0
    rmin  = 6.3
    vmin = 4/3*pi*rmin^3
    vmax = 4/3*pi*rmax^3

    # Physical parameters: Kernel
    a = 0.0
    b = 0.0
    c = 1.15e-14 * 1e9
    kernel_func = x -> a + b*(x[1]+x[2]) + c*abs(x[1]^(2/3)-x[2]^(2/3))/vmax^(2/3)*(x[1]^(1/3)+x[2]^(1/3))^2
    tracked_moments = [1.0]
    inject_rate = 0
    N     = 100           # initial droplet density: number per cm^3
    θ_v   = 1000            # volume scale factor: µm          
    θ_r   = 10             # radius scale factor: µm
    k     = 3             # shape factor for particle size distribution 
    ρ_w   = 1.0e-12       # density of droplets: 1 g/µm^3

    # initial/injection distribution in volume: gamma distribution in radius, number per cm^3
    r = v->(3/4/pi*v)^(1/3)
    #n_v_init = v -> N*(r(v))^(k-1)/θ_r^k * exp(-r(v)/θ_r) / gamma(k)
    n_v_init = v -> N*v^(k-1)/θ_v^k * exp(-v/θ_v) / gamma(k)
    #n_v_init = v -> N / σ_v / v / sqrt(2*pi) * exp(-(log(v)/σ_v)^2/2)
    n_v_inject = v -> (r(v))^(k-1)/θ_r^k * exp(-r(v)/θ_r) / gamma(k)
    
    # TODO: Smarter choice of RBF setup for low-res case
    basis = Array{CompactBasisFunc}(undef, Nb)
    rbf_loc = 2.3*[3.0, 4.0, 5.0]
    rbf_shapes = 2.3*[1.0, 1.0, 1.0]
    for i=1:Nb
      basis[i] = CompactBasisFunctionLog(rbf_loc[i], rbf_shapes[i])
    end
    println("means = ", rbf_loc)
    println("stddevs = ", rbf_shapes)
    #println(basis)
    plot_basis(basis, xstart=vmin*0.001, xstop=vmax)
    rbf_loc = exp.(rbf_loc)

    # Injection rate
    function inject_rate_fn(v)
      f = inject_rate*n_v_inject(v)
      return f
    end
    ########################### PRECOMPUTATION ################################
    v_start = 0.0
    v_stop = vmax

    start = time_ns()

    # Precomputation
    A = get_rbf_inner_products(basis, rbf_loc, tracked_moments)
    Source = get_kernel_rbf_source(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin)
    Sink = get_kernel_rbf_sink_precip(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin, xstop=vmax)
    #Sink = get_kernel_rbf_sink(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin, xstop=vmax)
    #Inject = get_injection_source(rbf_loc, tracked_moments, inject_rate_fn)
    (c_inject, Inject) = get_basis_projection(basis, rbf_loc, A, tracked_moments, inject_rate_fn, vmax)
    J = get_mass_cons_term(basis, xstart = vmin, xstop = vmax)
    m_inject = sum(c_inject .* J)

    # INITIAL CONDITION
    #(c0, nj_init) = get_IC_vecs(dist_init, basis, rbf_loc, A, tracked_moments)
    (c0, nj_init) = get_basis_projection(basis, rbf_loc, A, tracked_moments, n_v_init, vmax)
    m_init = sum(c0 .* J)
    t_elapsed = time_ns() - start
    println("precomputation complete:  ", t_elapsed*1e-9, " sec")

    ########################### DYNAMICS ################################
    start = time_ns()
    # Implicit Time stepping    
    function dndt(ni,t,p)
      return collision_coalescence(ni, A, Source, Sink, Inject)
    end

    prob = ODEProblem(dndt, nj_init, tspan)
    sol = solve(prob)
    #println(sol)
    t_elapsed = time_ns() - start
    println("time stepping complete:  ", t_elapsed*1e-9, " sec")

    t_coll = sol.t

    # track the moments
    basis_mom = vcat(get_moment(basis, 0.0, xstart=vmin, xstop=vmax)', get_moment(basis, 1.0, xstart=vmin, xstop=vmax)', get_moment(basis, 2.0, xstart=vmin, xstop=vmax)')
    c_coll = zeros(FT, length(t_coll), Nb)
    for (i,t) in enumerate(t_coll)
      nj_t = sol(t)
      c_coll[i,:] = get_constants_vec(nj_t, A)
    end
    
    mom_coll = c_coll*basis_mom'
    moments_init = mom_coll[1,:]
    println("times = ", t_coll)
    println("M_0 = ", mom_coll[:,1])
    println("M_1 = ", mom_coll[:,2])
    println("M_2 = ", mom_coll[:,3])
    println("c_init = ", c_coll[1,:])
    println("c_final = ", c_coll[end,:])

    #plot_nv_result(vmin*0.1, 1000.0, basis, c_coll[1,:], plot_exact=true, n_v_init=n_v_init, casename=casename)
    plot_nv_result(vmin*0.001, vmax, basis, c_coll[1,:], c_coll[end,:], plot_exact=true, n_v_init=n_v_init, log_scale=true, casename = casename)
    plot_nr_result(rmin*0.1, rmax, basis, c_coll[1,:], c_coll[end,:], plot_exact=true, n_v_init=n_v_init, log_scale=true, casename = casename)
    plot_moments(t_coll, mom_coll, casename = casename)
end


""" Plot Initial distribution only """
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

""" Plot the n(v) result, with option to show exact I.C. and log or linear scale """
function plot_nv_result(vmin::FT, vmax::FT, basis::Array{CompactBasisFunc, 1}, 
                        c::Array{FT, 1}...; plot_exact::Bool=false, n_v_init::Function = x-> 0.0, 
                        log_scale::Bool=false, casename::String="") where {FT <: Real}
  v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
  if plot_exact
    plot(v_plot,
        n_v_init.(v_plot),
        lw=2,
        label="Exact I.C.")
  else
    plot()
  end
  for (i,cvec) in enumerate(c)
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    if log_scale
      plot!(v_plot,
          n_plot,
          lw=2,
          ylim=[1e-6, 1e3],
          xlabel="volume, µm^3",
          ylabel="number",
          xaxis=:log,
          yaxis=:log,
          label=string("time ", i))
    else
      plot!(v_plot,
          n_plot,
          lw=2,
          ylim=[1e-6, 1e3],
          xlabel="volume, µm^3",
          ylabel="number",
          label=string("time ", i))
    end
  end
  Nb = length(basis)
  for i=1:Nb
    c_basis = zeros(FT,Nb)
    c_basis[i] = 1
    plot!(v_plot,
      evaluate_rbf(basis, c_basis, v_plot),
      ls=:dash,
      linecolor=:gray,
      label="basis_fn",
      legend=:topright)
  end

  savefig(string("rbf_paper/",casename,"nv.png"))
end

""" Plot the n(v) result, with option to show exact I.C. and log or linear scale """
function plot_nv_result(vmin::FT, vmax::FT, basis::Array{CompactBasisFunc, 1}, t::Array{FT,1},
                        c::Array{FT, 2}; plot_exact::Bool=false, n_v_init::Function = x-> 0.0, 
                        log_scale::Bool=false, casename::String="") where {FT <: Real}
  v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
  if plot_exact
    plot(v_plot,
        n_v_init.(v_plot),
        lw=2,
        label="Exact I.C.")
  else
    plot()
  end
  for (i,tsim) in enumerate(t)
    cvec = c[i,:]
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    if log_scale
      plot!(v_plot,
          n_plot,
          lw=2,
          ylim=[1e-6, 1e1],
          xlabel="volume, µm^3",
          ylabel="number",
          xaxis=:log,
          yaxis=:log,
          label=string("time ", tsim), legend=:bottomleft)
    else
      plot!(v_plot,
          n_plot,
          lw=2,
          ylim=[1e-6, 1e1],
          xlabel="volume, µm^3",
          ylabel="number",
          label=string("time ", tsim), legend=:bottomleft)
    end
  end
  Nb = length(basis)
  for i=1:Nb
    c_basis = zeros(FT,Nb)
    c_basis[i] = 1
    plot!(v_plot,
      evaluate_rbf(basis, c_basis, v_plot),
      ls=:dash,
      linecolor=:gray,
      label="basis_fn",
      legend=:topright)
  end

  savefig(string("rbf_paper/",casename,"nv.png"))
end

""" Plot the n(r) result, with option to show exact I.C. and log or linear scale """
function plot_nr_result(rmin::FT, rmax::FT, basis::Array{CompactBasisFunc, 1}, c::Array{FT, 1}...;
                        plot_exact::Bool=false, n_v_init::Function = x-> 0.0, 
                        log_scale::Bool=false, casename::String="") where {FT <: Real}
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
    if log_scale
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number",
            xaxis=:log,
            yaxis=:log,
            ylim=[1e-6, 1e1], legend=:bottomleft)
    else
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number",
            ylim=[1e-6, 1e1], legend=:bottomleft)
    end
  end
  savefig(string("rbf_paper/",casename,"nr.png"))
end

""" Plot the n(r) result, with option to show exact I.C. and log or linear scale """
function plot_nr_result(rmin::FT, rmax::FT, basis::Array{CompactBasisFunc, 1}, t::Array{FT,1}, c::Array{FT, 2};
                        plot_exact::Bool=false, n_v_init::Function = x-> 0.0, 
                        log_scale::Bool=false, casename::String="") where {FT <: Real}
  r_plot = exp.(collect(range(log(rmin), stop=log(rmax), length=1000)))
  v_plot = 4/3*pi*r_plot.^3
  if plot_exact
    plot(r_plot,
          n_v_init.(v_plot),
          lw=2,
          label="Exact")
  end
  for (i, tsim) in enumerate(t)
    cvec = c[i,:]
    n_plot = evaluate_rbf(basis, cvec, v_plot)
    if log_scale
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number",
            xaxis=:log,
            yaxis=:log,
            ylim=[1e-2, 1e1], legend=:bottomleft)
    else
      plot!(r_plot,
            n_plot,
            lw=2,
            xlabel="radius, µm",
            ylabel="number",
            ylim=[1e-2, 1e1], legend=:bottomleft)
    end
  end
  savefig(string("rbf_paper/",casename,"nr.png"))
end

""" Plot the moments supplied over time """
function plot_moments(tsteps::Array{FT}, moments::Array{FT, 2}; casename::String="") where {FT <: Real}
  plot(tsteps,
        moments[:,1],
        lw=2,
        xlabel="time, sec",
        ylabel="number / cm^3",
        label="M_0")
  for i=1:length(moments[1,:])
    plot(tsteps, 
          moments[:,i],
          lw=2,
          xlabel="time, sec",
          ylabel=string("M_",i-1),
          label=string("M_",i-1))
    savefig(string("rbf_paper/",casename,"M_",i-1,".png"))
  end
end

"""function plot_basis(basis::Array{CompactBasisFunc,1}, vmin::FT, vmax::FT,casename::String="") where {FT <: Real}
  Nb = length(basis)
  v_plot = exp.(collect(range(log(vmin), stop=log(vmax), length=1000)))
  for i=1:Nb
    c_basis = zeros(FT,Nb)
    c_basis[i] = 1
    plot!(v_plot,
      evaluate_rbf(basis, c_basis, v_plot),
      ls=:dash,
      linecolor=:gray,
      label="basis_fn",
      legend=:topright,
      xaxis=:log,
      yaxis=:log)
  end
  savefig(string("rbf_paper/",casename,"basis.png"))
end"""

@time main()