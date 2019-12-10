"Constant coalescence kernel example"

using DifferentialEquations
using Plots

using Cloudy.KernelTensors
using Cloudy.Distributions
using Cloudy.Sources


function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-8

  # Physicsal parameters
  coalescence_coeff = 1/3.14/4
  kernel_func = x -> coalescence_coeff
  kernel = CoalescenceTensor(kernel_func, 0, 100.0)

  # Initial condition
  moments_init = [150.0, 30.0, 200.0]
  distribution = Gamma(150.0, 6.466666667, 0.03092815)

  # Set up the right hand side of ODE
  rhs(m,p,t) = get_src_coalescence(m, distribution, kernel)

  # Solve the ODE
  tspan = (0.0, 1.0)
  prob = ODEProblem(rhs, moments_init, tspan)
  sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

  # Plot the solution for the 0th moment and compare to analytical solution
  pyplot()
  gr()
  time = sol.t
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = vcat(sol.u'...)[:, 3]
  p1 = plot(time,
      moment_0,
      linewidth=3,
      title="\$C(m, m') = k\$ (Smolu. 1916) vs. Climate Machine",
      xaxis="time",
      yaxis="M\$_k\$(time)",
      xlims=(0, 1.0),
      ylims=(0, 600.0),
      label="M\$_0\$ CLIMA"
  )
  plot!(time,
      t-> (1 / moments_init[1] + 0.5 * coalescence_coeff * t)^(-1),
      lw=3,
      ls=:dash,
      label="M\$_0\$ Exact"
  )
  plot!(time,
      moment_1,
      linewidth=3,
      label="M\$_1\$ CLIMA"
  )
  plot!(time,
      t-> moments_init[2],
      lw=3,
      ls=:dash,
      label="M\$_1\$ Exact"
  )
  plot!(time,
      moment_2,
      linewidth=3,
      label="M\$_2\$ CLIMA"
  )
  plot!(time,
      t-> moments_init[3] + moments_init[2]^2 * coalescence_coeff * t,
      lw=3,
      ls=:dash,
      label="M\$_2\$ Exact"
  )
  savefig("constant_kernel_example.png")
end

main()
