using Test
using Statistics
using TransportMaps

# Quick tests for RadialBasis
@testset "RadialBasis basic" begin
    # construct from number of rbfs
    rb = RadialBasis(3)
    @test rb isa RadialBasis
    @test length(rb.centers) == 3
    @test length(rb.scales) == 3

    # construct from samples
    samples = randn(1000)
    rb2 = RadialBasis(samples, 4)
    @test length(rb2.centers) == 4
    @test length(rb2.scales) == 4

    # evaluate a basisfunction and derivative
    val = basisfunction(rb2, 0.0, samples[1])
    der = basisfunction_derivative(rb2, 0.0, samples[1])
    @test isfinite(val)
    @test isfinite(der)

    # out of bounds index should throw
    @test_throws ArgumentError basisfunction(rb2, 10.0, samples[1])
end
