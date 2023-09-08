using Test

FT = Float64


@testset "Cloudy" begin
  @testset "Correctness" begin
    # @testset "Kernels" begin
    #   include("test_KernelTensors_correctness.jl")
    #   include("test_KernelFunctions_correctness.jl")
	  # end

	  @testset "ParticleDistributions" begin
	    include("test_ParticleDistributions_correctness.jl")
#      include("test_SuperParticleDistributions_correctness.jl")
    end

	  # @testset "Sources" begin
	  #   include("test_Sources_correctness.jl")
    #   include("test_MultiParticleSources_correctness.jl")
    # end
  end

  # @testset "Type stability" begin
  #   @testset "Kernels" begin
  #     include("test_KernelFunctions_opt.jl")
  #   end

  #   @testset "ParticleDistributions" begin
  #     include("test_SuperParticleDistributions_correctness.jl")
  #   end

  #   @testset "Sources" begin
  #     include("test_MultiParticleSources_opt.jl")
  #   end
  # end
end
