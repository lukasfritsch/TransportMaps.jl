using TransportMaps
using Test

@testset "GaussianWeightedHermiteBasis" begin
    gb = GaussianWeightedHermiteBasis()
    # Basic basisfunction behavior compared to hermite * gaussian weight
    @test basisfunction(gb, 2, 2.0) ≈ hermite_polynomial(2, 2.0) * exp(-0.25 * 2.0^2)
    @test basisfunction(gb, 1, 2.0) ≈ hermite_polynomial(1, 2.0)

    # Constructor edge cases
    @test_throws MethodError GaussianWeightedHermiteBasis(3)

    # Derivative evaluation
    d = basisfunction_derivative(gb, 2, 1.0)
    @test isfinite(d)

    @testset "Show" begin
        gb_show = GaussianWeightedHermiteBasis()
        @test_nowarn sprint(show, gb_show)
        @test_nowarn sprint(print, gb_show)
    end
end
