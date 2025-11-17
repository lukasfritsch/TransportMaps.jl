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

        # Test constructor with explicit μ and σ
        μ_test = [1.0, 2.0, 3.0]
        σ_test = [0.5, 1.5, 2.5]
        L2 = LinearMap(μ_test, σ_test)
        
        @test L2.μ == μ_test
        @test L2.σ == σ_test
        @test numberdimensions(L2) == 3
        
        # Verify it works correctly with specified parameters
        x_test = [2.0, 5.0, 8.0]
        y_test = evaluate(L2, x_test)
        x_recovered = inverse(L2, y_test)
        @test isapprox(x_recovered, x_test; atol=1e-10, rtol=1e-10)
        # Jacobian
        jac = jacobian(L)
        @test isapprox(jac, prod(L.σ); atol=1e-10, rtol=1e-10)
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

        L2 = LinearMap(2)
        @test numberdimensions(L2) == 2
    end

    @testset "Set parameters" begin
        L = LinearMap(2)
        μ_new = [1.0, -1.0]
        σ_new = [2.0, 0.5]
        setparameters!(L, μ_new, σ_new)

        @test L.μ == μ_new
        @test L.σ == σ_new

        x = [3.0, 0.0]
        y = evaluate(L, x)
        x_rec = inverse(L, y)
        @test isapprox(x_rec, x; atol=1e-10, rtol=1e-10)
    end

    @testset "Show" begin
        L = LinearMap()
        @test_nowarn sprint(show, L)
        @test_nowarn sprint(print, L)
        @test_nowarn display(L)
    end
end
