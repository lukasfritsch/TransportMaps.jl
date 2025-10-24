using TransportMaps
using Test
using Random

@testset "ComposedMap" begin
    rng = MersenneTwister(42)

    # Build a simple PolynomialMap that acts approximately as identity
    pm = PolynomialMap(2, 1, :normal, IdentityRectifier())
    # For degree 1 with our basis choices, setcoefficients to produce identity-like mapping
    # Component 1 (1D linear): coefficients [0, 1] -> f(x)=x
    setcoefficients!(pm.components[1], [0.0, 1.0])
    # Component 2 (2D linear): choose coefficients so M^2(z1,z2) = z2
    # Number of coefficients varies; set a vector where linear term for z2 is 1 and others 0
    coeffs2 = zeros(length(pm.components[2].coefficients))
    # Attempt to place 1.0 at the last coefficient which typically corresponds to z2 linear term
    coeffs2[end] = 1.0
    setcoefficients!(pm.components[2], coeffs2)

    # Build LinearMap from simple samples (shift-scale)
    samples = randn(rng, 50, 2) .* [2.0 1.0] .+ [0.5 - 0.3]
    L = LinearMap(samples)

    C = ComposedMap(L, pm)

    @test numberdimensions(C) == 2

    # Test evaluate/inverse round-trip
    x = [0.1, -0.2]
    y = evaluate(C, x)
    x_rec = inverse(C, y)
    @test isapprox(x_rec, x; atol=1e-8)

    # Matrix forms
    X = randn(rng, 5, 2)
    Y = evaluate(C, X)
    X_rec = inverse(C, Y)
    @test isapprox(X_rec, X; atol=1e-8)

    # Test pullback scaling: pullback(Composed) == pullback(Polynomial) / prod(σ)
    # Compute pullbacks at a point (use X row)
    x0 = X[1, :]
    pb_C = pullback(C, x0)
    pb_pm = pullback(pm, evaluate(L, x0))
    scale = prod(L.σ)
    @test isapprox(pb_C * scale, pb_pm; atol=1e-8, rtol=1e-8)

    # Test pullback scaling: pullback(Composed) == pullback(Polynomial) / prod(σ) for matrix input
    pb_C = pullback(C, X)
    pb_pm = pullback(pm, evaluate(L, X))
    scale = prod(L.σ)
    @test isapprox(pb_C * scale, pb_pm; atol=1e-8, rtol=1e-8)

    @testset "Show" begin
        @test_nowarn sprint(show, C)
        @test_nowarn sprint(print, C)
        @test_nowarn display(C)
    end
end
