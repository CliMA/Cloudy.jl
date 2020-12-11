"Evaluation of MC integration"

using DifferentialEquations
using LinearAlgebra
using Plots
using Statistics
using Random: seed!

using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.Sources

seed!(123)


function main()
  # Numerical parameters
  n_mc_samples = 1000
  n_samples = [10, 50, 100, 500, 1000, 10000]

  # Kernel
  coalescence_coeff = 1.0e-6
  kernel_func = ConstantKernelFunction(coalescence_coeff)
  coalescence_coeff = 5.78e3 
  kernel_func = LinearKernelFunction(coalescence_coeff)

  # Distribution
  particle_number = 1e4 # 1e4 g^-1
  mean_particles_mass = 1e-8 # 1e-8 g
  particle_mass_std = 0.5e-8 # 0.5e-8 g
  pars = [particle_number; (mean_particles_mass/particle_mass_std)^2; particle_mass_std^2/mean_particles_mass]
  pdist = GammaParticleDistribution(pars[1], pars[2], pars[3])
  k = 4 # moment of interest

  # Performance computation for MCMC
  mc_mean = zeros(length(n_samples))
  mc_std = zeros(length(n_samples))
  mc_samples = zeros(n_mc_samples)
  for j in 1:length(n_samples)
    for i = 1:n_mc_samples
      mc_samples[i] = get_coalescence_integral_moment(k, kernel_func, pdist, n_samples[j])
    end
    mc_mean[j] = mean(mc_samples)
    mc_std[j] = std(mc_samples)
  end 

  p1 = plot(n_samples,
      2*mc_std/(mc_mean[end]),
      lw=3,
      ls=:dash,
      xlabel="number of MC samples",
      ylabel="relative coalescence_integral error",
      xlims=(10, maximum(n_samples)),
      xaxis=:log10,
      #yaxis=:log10,
      )
#   plot(p1, p2, p3, layout=(1, 3), size=(1000, 375), margin=5Plots.mm)
   savefig("mc_evaluation.png")
end

main()
