using TransportMaps
using Test
using Distributions

@testset "CubicSplineHermiteBasis" begin
    # Default constructor
    cs_default = CubicSplineHermiteBasis()
    @test isa(cs_default.radius, Float64)
    @test isfinite(basisfunction(cs_default, 2, 0.5))
    @test isfinite(basisfunction_derivative(cs_default, 2, 0.5))

    # Float radius constructor
    cs = CubicSplineHermiteBasis(3.0)
    @test cs.radius == 3.0
    @test basisfunction(cs, 2, 1.0) ≈ hermite_polynomial(2, 1.0) * (2 * (min(1.0, abs(1.0) / cs.radius))^3 - 3 * (min(1.0, abs(1.0) / cs.radius))^2 + 1)
    @test basisfunction(cs, 1, 2.0) ≈ hermite_polynomial(1, 2.0)

    # Vector-of-samples constructor
    samples = randn(200)
    cs_samples = CubicSplineHermiteBasis(samples)
    @test isa(cs_samples.radius, Float64)
    @test cs_samples.radius > 0.0
    @test isfinite(basisfunction(cs_samples, 3, median(samples)))

    # Density-based constructor

    cs_density = CubicSplineHermiteBasis(Normal(0.0, 1.0))
    @test isa(cs_density.radius, Float64)
    @test cs_density.radius > 0.0

    # Constructor negative/invalid arg should throw
    @test_throws MethodError CubicSplineHermiteBasis(-2) # expects Float64 or samples

    # Derivative checks
    dd = basisfunction_derivative(cs, 2, 1.0)
    @test isfinite(dd)

    @testset "Show" begin
        cs_show = CubicSplineHermiteBasis()
        @test_nowarn sprint(show, cs_show)
        @test_nowarn sprint(print, cs_show)
        @test_nowarn display(cs_show)
    end
end
