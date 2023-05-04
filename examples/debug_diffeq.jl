using Cloudy.KernelFunctions
using Cloudy.ParticleDistributions
using Cloudy.MultiParticleSources
using QuadGK
using Distributions: Distribution, Gamma

particle_number = 2
dist_params = Array([9085.299759283542, 0.9198205475401283, 3.4621243630711337e-10, 125.18796848904218, 0.15482085294932268, 4.448742237221811e-7], 2, 3)
dist_params = transpose(dist_params)
pdists = map(1:length(particle_number)) do i
    GammaParticleDistribution(dist_params[1, i], dist_params[2,i], dist_params[3,i])
end
kernel_func = LinearKernelFunction{Float64}(200.0)
moment_order = 0

function weighting_fn(x, pdist1, pdist2)
    denom = pdist1(x) + pdist2(x)
    if denom == 0.0
      return 0.0
    else
      return pdist1(x) / (pdist1(x) + pdist2(x))
    end
  end
  
  function q_integrand_inner(x, y, j, k, kernel, pdists)
    integrand = 0.5 * kernel(x - y, y) * pdists[j](x-y) * pdists[k](y)
    return integrand
  end
  
  function q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    outer = x.^moment_order * quadgk(yy -> q_integrand_inner(x, yy, j, k, kernel, pdists), 0.0, x; rtol=1e-4)[1]
    return outer
  end
  
  function r_integrand(x, y, j, k, kernel, pdists, moment_order)
    integrand = x.^moment_order * kernel(x, y) * pdists[j](x) * pdists[k](y)
    return integrand
  end

  function s_integrand1(x, j, k, kernel, pdists, moment_order)
    integrandj = weighting_fn(x, pdists[j], pdists[k]) * q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    return integrandj
  end
  
  function s_integrand2(x, j, k, kernel, pdists, moment_order)
    integrandk = (1 - weighting_fn(x, pdists[j], pdists[k])) * q_integrand_outer(x, j, k, kernel, pdists, moment_order)
    return integrandk
  end

x_issue = 1.15e-9
@show q_integrand_outer(x_issue, 1, 1, kernel_func, pdists, moment_order)
