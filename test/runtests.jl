using Test

FT = Float64


@testset "Cloudy" begin
  @testset "Correctness" begin
    @testset "Kernels" begin
      include("test_KernelTensors_correctness.jl")
      include("test_KernelFunctions_correctness.jl")
	  end

	  @testset "Distributions" begin
	    include("test_Distributions_correctness.jl")
    end

	  @testset "Sources" begin
	    include("test_Sources_correctness.jl")
    end
  end

  #@testset "Type stability" begin
  #  #TO BE DONE.
  #end
end
