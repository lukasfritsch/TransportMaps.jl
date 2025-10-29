using TransportMaps
using Test
using LinearAlgebra
using Random

@testset "LaplaceMap" begin
    # Create samples from a Laplace distribution
    rng = MersenneTwister(123)
    n_samples = 10
    samples = randn(rng, n_samples, 2)

    # Build LaplaceMap from samples
    L_map = LaplaceMap(samples)

    @test numberdimensions(L_map) == 2

    # Test evaluate/inverse round-trip
    x = [0.5, 0.2]
    y = evaluate(L_map, x)
    x_rec = inverse(L_map, y)
    @test isapprox(x_rec, x; atol=1e-8)

    # Matrix forms
    X = randn(rng, 5, 2)
    Y = evaluate(L_map, X)
    X_rec = inverse(L_map, Y)
    @test isapprox(X_rec, X; atol=1e-8)

    # Jacobian
    jac = jacobian(L_map)
    @test isapprox(jac, abs(det(L_map.Chol)); atol=1e-10)

    @testset "Show" begin
        @test_nowarn sprint(show, L_map)
        @test_nowarn sprint(print, L_map)
        @test_nowarn display(L_map)
    end
end
