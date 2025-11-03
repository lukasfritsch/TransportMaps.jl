using TransportMaps
using Test
using LinearAlgebra
using Random

@testset "LaplaceMap from Samples" begin
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
    @test isapprox(jac, abs(det(L_map.chol)); atol=1e-10)

    # Test mean and covariance
    μ = mean(samples, dims=1) |> vec
    @test isapprox(mean(L_map), μ; atol=1e-8)
    @test isapprox(cov(L_map), cov(samples); atol=1e-8)
    @test isapprox(mode(L_map), μ; atol=1e-8)
    @test isapprox(cov(L_map), L_map.chol * L_map.chol'; atol=1e-8)

    @testset "Show" begin
        @test_nowarn sprint(show, L_map)
        @test_nowarn sprint(print, L_map)
        @test_nowarn display(L_map)
    end
end


@testset "LaplaceMap from Density" begin

    density(x) = pdf(LogNormal(0.0, 0.5), x[1]) * pdf(LogNormal(1, 0.5), x[2] - x[1])
    target = MapTargetDensity(density, :ad)

    x0 = [0.5, 1.0]
    L_map = LaplaceMap(target, x0)

    @test numberdimensions(L_map) == 2
end
