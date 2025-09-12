# This file will contain tests for general MultivariateBasis logic, not specific to Hermite
using TransportMaps
using Test

@testset "MultivariateBasis General" begin
    @testset "Multi-index generation" begin
        idx = multivariate_indices(2, 2)
        @test length(idx) > 0
        @test all(length(i) == 2 for i in idx)
    end
    # Add more general tests as needed
end
