"Product coalescence kernel example"

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
  coalescence_coeff = Array{FT}([[0.0, 0.0] [0.0, 1/3.14/25]])
  kernel = LinearCoalescenceTensor(coalescence_coeff)

  # Initial condition
  moments_init = [100.0, 90.0, 100.0]
  distribution = Gamma(100.0, 0.21111111111111114, 4.263157894736842)

  # Set up the right hand side of ODE
  rhs(m,p,t) = get_src_coalescence(m, distribution, kernel)

  # Solve the ODE
  tspan = (0.0, 0.75)
  prob = ODEProblem(rhs, moments_init, tspan)
  sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

  # Plot the solution for the 0th moment and compare to analytical solution
  pyplot()
  gr()
  time = sol.t
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  moment_2 = vcat(sol.u'...)[:, 3]

  plot(time,
      moment_0,
      linewidth=3,
      title="\$C(m, m') = k mm'\$ vs. Climate Machine",
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
  savefig("product_kernel_example.png")
end

main()
