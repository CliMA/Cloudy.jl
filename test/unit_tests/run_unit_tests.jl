using Test

FT = Float64


@testset "Cloudy" begin
    @testset "Correctness" begin
        @testset "Helper functions" begin
            include("test_helper_functions.jl")
        end

        @testset "Kernels" begin
            include("test_KernelTensors_correctness.jl")
            include("test_KernelFunctions_correctness.jl")
        end

        @testset "ParticleDistributions" begin
            include("test_ParticleDistributions_correctness.jl")
        end

        @testset "Sources" begin
            include("test_Sources_correctness.jl")
        end
    end

    @testset "Type stability" begin
        @testset "Kernels" begin
            include("test_KernelFunctions_opt.jl")
            include("test_KernelTensors_opt.jl")
        end

        # TODO
        # @testset "Sources" begin
        #     include("test_Sources_opt.jl")
        # end

        include("performance_tests.jl")

    end
end
