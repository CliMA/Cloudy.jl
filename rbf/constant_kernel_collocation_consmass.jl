"Constant coalescence kernel example"

using Plots
using Cloudy.BasisFunctions
using Cloudy.Collocation
using QuadGK

function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-8

  # Physicsal parameters
  coalescence_coeff = 1/3.14/4
  kernel_func = x -> coalescence_coeff

  # Initial condition: any function; here we choose a lognormal distribution
  N = 150.0
  mu = 1.5
  sigma = 0.5
  dist_init = x-> N/x/sigma/sqrt(2*pi)*exp(-(log.(x)-mu)^2/(2*sigma^2))

  # Choose the basis functions
  Nb = 10
  mu_start = 5.0
  mu_stop = 25.0
  rbf_mu = collect(range(mu_start, stop=mu_stop, length=Nb))
  rbf_sigma = repeat([mu_start/2], Nb)
  basis = Array{PrimitiveUnivariateBasisFunc}(undef, Nb)
  for i = 1:Nb
    basis[i] = GaussianBasisFunction(rbf_mu[i], rbf_sigma[i])
  end

  println("Basis:  ", basis)

  # Precompute the various matrices
  A = get_rbf_inner_products(basis)
  Source = get_kernel_rbf_source(basis, rbf_mu, kernel_func)
  Sink = get_kernel_rbf_sink(basis, rbf_mu, kernel_func)
  mass_cons = get_mass_cons_term(basis)
  (c0, mass) = get_IC_vec(dist_init, basis, A, mass_cons)

  # set up the explicit time stepper
  tspan = (0.0, 1.0)
  dt = 1e-3
  tsteps = range(tspan[1], stop=tspan[2], step=dt)
  nj = dist_init.(rbf_mu)
  dndt = ni->collision_coalescence(ni, A, Source, Sink, mass_cons, mass)

  for t in tsteps
    nj += dndt(nj)*dt
  end

  A2 = vcat(A, mass_cons')
  nj2 = vcat(nj, mass)

  c_final = nonneg_lsq(A2, nj2)[:,1]
  #c_final = A\nj

  ##PLOTTING
  # plot the initial distribution
  x = range(1.0, stop=50.0, step=0.1) |> collect
  dist_exact = dist_init.(x)
  dist_galerkin = evaluate(basis, c0, x)
  pyplot()
  plot(x, dist_exact, label="Exact", title="Constant Kernel")
  plot!(x, dist_galerkin, label="Collocation approximation")

  # plot the final distribution
  dist_galerkin = evaluate(basis, c_final, x)
  plot!(x, dist_galerkin, label="Collocation approximation: final state")

  mass_dist0 = x->evaluate(basis,c0,x)*x
  mass_distf = x->evaluate(basis,c_final,x)*x

  xstart = eps()
  xstop = 1000.0
  m0 = quadgk(mass_dist0, xstart, xstop)[1]
  mf = quadgk(mass_distf, xstart, xstop)[1]
  println("Starting mass: ", m0)
  println("Ending mass: ", mf)
  annotate!([(15.0, 20.0, Plots.text(string("Starting mass: ", m0), 12))])
  annotate!([(15.0, 15.0, Plots.text(string("Ending mass: ", mf), 12))])

  savefig("rbf/initial_final_collocation_masscons.png")
end

@time main()