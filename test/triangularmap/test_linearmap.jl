using TransportMaps
using Test
using LinearAlgebra
using Random

@testset "LinearMap" begin
    rng = MersenneTwister(1234)

    @testset "Constructor and basic operations" begin
        samples = randn(rng, 100, 3)
        L = LinearMap(samples)

        @test numberdimensions(L) == 3

        x = randn(rng, 3)
        y = evaluate(L, x)
        x_rec = inverse(L, y)
        @test isapprox(x_rec, x; atol=1e-10, rtol=1e-10)

        # Matrix form
        X = randn(rng, 7, 3)
        Y = evaluate(L, X)
        X_rec = inverse(L, Y)
        @test isapprox(X_rec, X; atol=1e-10, rtol=1e-10)
    end

    @testset "Identity LinearMap" begin
        L0 = LinearMap()
        @test numberdimensions(L0) == 0

        x = Float64[]
        @test evaluate(L0, x) == x
        @test inverse(L0, x) == x

        X = Array{Float64}(undef, 2, 0)
        @test size(evaluate(L0, X)) == size(X)
        @test size(inverse(L0, X)) == size(X)
    end

    @testset "Show" begin
        L = LinearMap()
        @test_nowarn sprint(show, L)
        @test_nowarn sprint(print, L)
        @test_nowarn display(L)
    end
end
