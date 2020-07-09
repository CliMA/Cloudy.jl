"Single aerosol species MOM evolution"

using DifferentialEquations
using Plots

using Cloudy.ParticleDistributions
using Cloudy.Activation

function main()
    # Numerical parameters
    FT = Float64
    tol = FT(1e-8)
  
    # Initial condition:
    S_init = FT(0.02)
    dist_init = GammaPrimitiveParticleDistribution(FT(100), FT(28.0704505), FT(3.8564))
    v_up = FT(1)
    tspan = (FT(0), FT(1))
  
    moments_S_init = FT[0.0, 0.0, 0.0, S_init]
    println("Initializing with moments:")
    for k in 0:2
      moments_S_init[k+1] = moment(dist_init, FT(k))
      println(moments_S_init[k+1])
    end
    println("Supersaturation: ", S_init)
    println()
  
    ODE_parameters = Dict(:dist => dist_init)
  
    # implement callbacks
    function out_of_bounds(m, t, integrator)
  
      for i in 1:3
        if integrator.m[i] < 0
          return true
        else
          return false
        end
      end
    end
  
    #function affect!(integrator)
    #  ddt_try = get_proposed_dt(integrator)
    #  set_proposed_dt!(integrator, dt_try/2)
    #end
  
    #function affect!(integrator)
    #  for i in 1:3
    #    if integrator.
  
    #condition(m, t, integrator) = out_of_bounds(m, t, integrator)
    #cb=DiscreteCallback(condition, affect!)
  
  
    # set up ODE
    rhs(m, par, t) = get_aerosol_growth_3mom(m, par, t, v_up)
  
    # solve the ODE
    println("Solving ODE...")
    prob = ODEProblem(rhs, moments_S_init, tspan, ODE_parameters)
    #sol = solve(prob, reltol = tol, abstol = tol, callback=cb)
    sol = solve(prob, reltol = tol, abstol = tol, isoutofdomain = (m,par,t) -> any(x->x<0, m))
  
    # Plot the solution for the 0th moment
    pyplot()
    gr()
    time = sol.t
    mom = vcat(sol.u'...)
    moment_0 = mom[:, 1]
    moment_1 = mom[:, 2]
    moment_2 = mom[:, 3]
    S = vcat(sol.u'...)[:,4]
  
    plot(time,
        moment_0,
        linewidth=3,
        xaxis="time",
        yaxis="M\$_k\$(time)",
        xlims=(0, 1.0),
        ylims=(0, 600.0),
        label="M\$_0\$ CLIMA"
    )
    plot!(time,
        moment_1,
        linewidth=3,
        label="M\$_1\$ CLIMA"
    )
    plot!(time,
        moment_2,
        linewidth=3,
        label="M\$_2\$ CLIMA"
    )
    savefig("aerosol_growth.png")
  
    pyplot()
    gr()
    plot(time,
        S,
        linewidth=3,
        label="S CLIMA")
    savefig("aerosol_growth_S.png")
  end

  main()