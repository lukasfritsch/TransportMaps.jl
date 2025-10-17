using TransportMaps
using Test

@testset "GaussianWeightedHermiteBasis" begin
    gb = GaussianWeightedHermiteBasis()
    # Basic basisfunction behavior compared to hermite * gaussian weight
    @test basisfunction(gb, 2, 2.0) ≈ hermite_polynomial(2, 2.0) * exp(-0.25 * 2.0^2)
    @test basisfunction(gb, 1, 2.0) ≈ hermite_polynomial(1, 2.0)

    # show method
    s = sprint(show, gb)
    @test occursin("GaussianWeightedHermiteBasis", s)

    # Constructor edge cases
    @test_throws MethodError GaussianWeightedHermiteBasis(3)

    # Derivative evaluation
    d = basisfunction_derivative(gb, 2, 1.0)
    @test isfinite(d)
end
