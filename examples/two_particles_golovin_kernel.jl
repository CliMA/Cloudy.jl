"Linear (Golovin) coalescence kernel example"

using DifferentialEquations
using LinearAlgebra
using Plots
using Random: seed!

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.MultiParticleSources

seed!(123)


function main()
  # Numerical parameters
  FT = Float64
  tol = 1e-4
  n_samples = 25
  n_inducing = 5

  # Physicsal parameters
  # Mass has been rescaled below by a factor of 1e3 so that 1 gram = 1e3 milligram 
  # Time has been rescaled below by a factor of 1e1 so that 1 sec = 10 deciseconds
  mass_scale = 1e3
  time_scale = 1e1
  
  T_end = 3 * time_scale #3 s
  coalescence_coeff = 5.78e3 / mass_scale / time_scale #5.78e3 cm^3 g^-1 s-1  
  kernel_func = LinearKernelFunction(coalescence_coeff)
  
  # Parameter transform used to transform native distribution
  # parameters to the real axis
  trafo = native_state -> native_state > 10.0 ? native_state : log.(exp.(native_state) - 1.0)
  inv_trafo = state -> state > 10.0 ? state : log.(exp.(state) + 1.0)
  inv_trafo_der = state -> 1.0 ./ (1.0 + exp.(-state))

  # Initial condition
  # We carrry transformed parameters in our time stepper for
  # stability purposes
  particle_number = [1e4, 1e1]
  mean_particles_mass = [1e-8 * mass_scale, 1e-6 * mass_scale] #1e-7 g
  particle_mass_std = [0.5e-8 * mass_scale, 0.25e-6 * mass_scale] #0.5e-7 g
  pars_init = reduce(vcat, transpose.([particle_number, (mean_particles_mass ./ particle_mass_std).^2, particle_mass_std.^2 ./ mean_particles_mass]))
  state_init = trafo.(pars_init) 

  # Set up the ODE problem
  # Step 1) Define termination criterion: stop integration when one of the 
  #         distribution parameters leaves its allowed domain (which can 
  #         happen before the end of the time period defined below by tspan)
  nothing

  # Step 2) Set up the right hand side of ODE
  function rhs!(dstate, state, p, t)
    # Transform state to native distribution parameters
    native_state = inv_trafo.(state)

    # Evaluate processes at inducing points using a closure distribution
    pdists = Array{GammaParticleDistribution{FT}}(undef, length(particle_number))
    inducing_points = Array{FT}(undef,length(particle_number),n_inducing)
    for i in 1:length(particle_number)
        pdists[i] = GammaParticleDistribution(native_state[1, i], native_state[2,i], native_state[3,i])
        inducing_points[i,:] = sample(pdists[i], n_inducing)
    end

    (coal_int_plus, coal_int_minus) = get_coalescence_integrals(inducing_points, kernel_func, pdists, n_samples)

    for i in 1:length(pdists)
        # Obtain time derivatve of the transformed distribution parameters
        jacobian = density_gradient(pdists[i], inducing_points[i,:]) * diagm(inv_trafo_der.(state[:,i]))
        transformed_int_plus = inv(jacobian'*jacobian)*jacobian'*coal_int_plus[:,i]
        transformed_int_minus = inv(jacobian'*jacobian)*jacobian'*coal_int_minus[:,i]

        # Projection to enforce mass conservation in transformed space
        normal = normal_mass_constraint(pdists[i])
        transformed_normal = diagm(inv_trafo_der.(state[:,i])) * normal
        unit_normal = transformed_normal / norm(transformed_normal)
        transformed_int_plus = (I - unit_normal * unit_normal') * transformed_int_plus
        transformed_int_minus = (I - unit_normal * unit_normal') * transformed_int_minus

        # Assign time derivative
        for j in 1:length(dstate[:,i])
            dstate[j,i] = transformed_int_plus[j] - transformed_int_minus[j]
        end
    end
  end

  # Step 3) Solve the ODE
  tspan = (0.0, T_end)
  prob = ODEProblem(rhs!, state_init, tspan)
  sol = solve(prob, Tsit5(), reltol=tol, abstol=tol)

  # Step 4) Plot the results
  time = sol.t / time_scale

  # Get the native distribution parameters
  n = inv_trafo.(vcat(sol.u'...)[:, 1])
  k = inv_trafo.(vcat(sol.u'...)[:, 2])
  θ = inv_trafo.(vcat(sol.u'...)[:, 3])

  # Calculate moments for plotting
  moment_0 = n
  moment_1 = n.*k.*θ
  moment_2 = n.*k.*(k.+1.0).*θ.^2

  p1 = plot(time,
      moment_0,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M0 [1/cm^3]",
      xlims=(0, maximum(time)),
      ylims=(0, 1.5*maximum(moment_0)),
      label="M0 CLIMA"
  )
  plot!(p1, time,
      t-> (moment_0[1] * exp(-moment_1[1] * coalescence_coeff * t * time_scale)),
      lw=3,
      ls=:dash,
      label="M0 Exact"
  )

  p2 = plot(time,
      moment_1,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M1 [milligrams/cm^3]",
      ylims=(0, 1.5*maximum(moment_1)),
      label="M1 CLIMA"
  )
  plot!(p2, time,
      t-> moment_1[1],
      lw=3,
      ls=:dash,
      label="M1 Exact"
  )
  p3 = plot(time,
      moment_2,
      linewidth=3,
      xaxis="time [s]",
      yaxis="M2 [milligrams^2/cm^3]",
      ylims=(0, 1.5*maximum(moment_2)),
      label="M2 CLIMA"
  )
  plot!(p3, time,
  t-> (moment_2[1] * exp(2 * moment_1[1] * coalescence_coeff * t * time_scale)),
  lw=3,
      ls=:dash,
      label="M2 Exact"
  )
  plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
  savefig("golovin_kernel_test.png")
end

main()
