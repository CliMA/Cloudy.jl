""" collision-coalescence only with constant kernel """

using Plots
using Cloudy.BasisFunctions
using Cloudy.MomentCollocation
using QuadGK
using SpecialFunctions: gamma
using DifferentialEquations

function main()
    ############################ SETUP ###################################

    # Numerical parameters
    FT = Float64

    # basis setup 
    Nb = 15
    rmax  = 100.0
    rmin  = 2.0
    vmin = 4*rmin^3
    vmax = 4*rmax^3

    # Physical parameters: Kernel
    a = 1e-6
    b = 0.0
    c = 0.0
    kernel_func = x -> a + b*(x[1]+x[2]) + c*abs(x[1]^(2/3)-x[2]^(2/3))/vmax^(2/3)*(x[1]^(1/3)+x[2]^(1/3))^2
    tracked_moments = [1.0]
    inject_rate = 0
    N     = 100           # initial droplet density: number per cm^3
    θ_r   = 6.42          # radius scale factor: µm
    k     = 3             # shape factor for particle size distribution 
    ρ_w   = 1.0e-12       # density of droplets: 1 g/µm^3

    # initial/injection distribution in volume: gamma distribution in radius, number per cm^3
    r = v->(3/4/pi*v)^(1/3)
    n_v_init = v -> N*(r(v))^(k-1)/θ_r^k * exp(-r(v)/θ_r) / gamma(k)
    n_v_inject = v -> (r(v))^(k-1)/θ_r^k * exp(-r(v)/θ_r) / gamma(k)
    
    # lin-spaced log compact rbf
    basis = Array{CompactBasisFunc}(undef, Nb)
    rbf_loc = collect(range(log(vmin), stop=log(vmax), length=Nb))
    rbf_shapes = zeros(Nb)
    rbf_shapes[3:end] = (rbf_loc[3:end] - rbf_loc[1:end-2])
    rbf_shapes[2] = rbf_loc[2]
    rbf_shapes[1] = rbf_loc[2]
    for i=1:Nb
      basis[i] = CompactBasisFunctionLog(rbf_loc[i], rbf_shapes[i])
    end
    #println(basis)
    rbf_loc = exp.(rbf_loc)

    # Injection rate
    function inject_rate_fn(v)
      f = inject_rate*n_v_inject(v)
      return f
    end
    ########################### PRECOMPUTATION ################################
    v_start = 0.0
    v_stop = vmax

    # Precomputation
    A = get_rbf_inner_products(basis, rbf_loc, tracked_moments)
    Source = get_kernel_rbf_source(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin)
    Sink = get_kernel_rbf_sink_precip(basis, rbf_loc, tracked_moments, kernel_func, xstart=vmin, xstop=vmax)
    #Inject = get_injection_source(rbf_loc, tracked_moments, inject_rate_fn)
    (c_inject, Inject) = get_basis_projection(basis, rbf_loc, A, tracked_moments, inject_rate_fn, vmax)
    J = get_mass_cons_term(basis, xstart = vmin, xstop = vmax)
    m_inject = sum(c_inject .* J)

    # INITIAL CONDITION
    #(c0, nj_init) = get_IC_vecs(dist_init, basis, rbf_loc, A, tracked_moments)
    (c0, nj_init) = get_basis_projection(basis, rbf_loc, A, tracked_moments, n_v_init, θ_v*12.0)
    m_init = sum(c0 .* J)
    println("precomputation complete")

    ########################### DYNAMICS ################################
    # Implicit Time stepping
    tspan = (0.0, 60.0)
    
    function dndt(ni,t,p)
      return collision_coalescence(ni, A, Source, Sink, Inject)
    end

    prob = ODEProblem(dndt, nj_init, tspan)
    sol = solve(prob)
    #println(sol)

    t_coll = sol.t

    # track the moments
    basis_mom = vcat(get_moment(basis, 0.0, xstart=vmin, xstop=vmax)', get_moment(basis, 1.0, xstart=vmin, xstop=vmax)', get_moment(basis, 2.0, xstart=vmin, xstop=vmax)')
    println(basis_mom)
    c_coll = zeros(FT, length(t_coll), Nb)
    for (i,t) in enumerate(t_coll)
      nj_t = sol(t)
      c_coll[i,:] = get_constants_vec(nj_t, A)
      println(t, c_coll[i,:])
    end
    
    mom_coll = c_coll*basis_mom'
    moments_init = mom_coll[1,:]

    #plot_nv_result(vmin*0.1, vmax, basis, c0, c_coll[end,:], plot_exact=true, n_v_init=n_v_init)
    plot_nr_result(rmin/2, rmax*2, basis, c0, c_coll[end,:], plot_exact=true, n_v_init=n_v_init)
    plot_moments(t_coll, mom_coll)
end

function plot_nv_result(vmin::FT, vmax::FT, basis::Array{CompactBasisFunc, 1}, 
                        c::Array{FT, 1}...; plot_exact::Bool=false, n_v_init::Function = x-> 0.0) where {FT <: Real}
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
    plot!(v_plot,
        n_plot,
        lw=2,
        ylim=[1e-2, 1e2],
        xlabel="volume, µm^3",
        ylabel="number",
        xaxis=:log,
        yaxis=:log,
        label=string("time ", i)
    )
  end

  savefig("rbf_paper/nv.png")
end

function plot_nr_result(rmin::FT, rmax::FT, basis::Array{CompactBasisFunc, 1}, c::Array{FT, 1}...;
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
          xaxis=:log,
          yaxis=:log,
          ylim=[1e-3, 1e2])
  end
  savefig("rbf_paper/nr.png")
end

function plot_moments(tsteps::Array{FT}, moments::Array{FT, 2}) where {FT <: Real}
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
    savefig(string("rbf_paper/M_",i-1,".png"))
  end
end

main()