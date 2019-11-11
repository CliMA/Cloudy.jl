"Smoluchowski 1916 constant coalescence kernel example"

using DifferentialEquations
using Plots

using Cloudy.KernelTensors
using Cloudy.MassDistributions
using Cloudy.Sources


function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-8

  # Physicsal parameters
  coalescence_coeff = 1/3.14
  kernel = ConstantCoalescenceTensor(coalescence_coeff)

  # Initial condition
  moments_init = Array{FT}([67.14, 123.325, 524.23])
  distribution = Gamma(4.56, 1.24, 6.23)
  update_params!(distribution, moments_init)

  # Set up the right hand side of ODE
  rhs(m,p,t) = get_src_coalescence(m, distribution, kernel)

  # Solve the ODE
  tspan = (0.0, 1.0)
  prob = ODEProblem(rhs, moments_init, tspan)
  sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

  # Plot the solution for the 0th moment and compare to analytical solution
  time = sol.t
  moment_0 = vcat(sol.u'...)[:, 1]
  moment_1 = vcat(sol.u'...)[:, 2]
  plot(time,
      moment_0,
      linewidth=3,
      title="Smoluchowski 1916 (K(m,m') = const.) vs. Climate Machine",
      xaxis="Time (t)",
      yaxis="M_0(t)",
      xlims=(0, 1.0),
      label="CliMA"
  )
  plot!(time,
      t-> (1 / moments_init[1] + 0.5 * coalescence_coeff * t)^(-1),
      lw=3,
      ls=:dash,
      label="SM1916"
  )
end

main()
