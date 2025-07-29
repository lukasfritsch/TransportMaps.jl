using TransportMaps
using Test

@testset "Gauss Quadrature" begin
    @testset "Basic Integration Tests" begin
        # Test integration of constant function
        f_const(x) = 1.0
        result = gaussquadrature(f_const, 5, 0.0, 1.0)
        @test result ≈ 1.0 atol=1e-10

        result = gaussquadrature(f_const, 10, -1.0, 1.0)
        @test result ≈ 2.0 atol=1e-10

        # Test integration of linear function
        f_linear(x) = x
        result = gaussquadrature(f_linear, 5, 0.0, 2.0)
        @test result ≈ 2.0 atol=1e-10  # ∫₀² x dx = [x²/2]₀² = 2

        result = gaussquadrature(f_linear, 10, -1.0, 1.0)
        @test result ≈ 0.0 atol=1e-10  # ∫₋₁¹ x dx = 0 (odd function)

        # Test integration of quadratic function
        f_quad(x) = x^2
        result = gaussquadrature(f_quad, 5, 0.0, 1.0)
        @test result ≈ 1/3 atol=1e-10  # ∫₀¹ x² dx = [x³/3]₀¹ = 1/3

        result = gaussquadrature(f_quad, 10, -2.0, 2.0)
        @test result ≈ 16/3 atol=1e-10  # ∫₋₂² x² dx = [x³/3]₋₂² = 8/3 - (-8/3) = 16/3
    end

    @testset "Polynomial Integration" begin
        # Test exact integration of polynomials up to degree 2n-1
        # where n is the number of quadrature points

        # Cubic polynomial with 2 points (should be exact up to degree 3)
        f_cubic(x) = x^3 + 2*x^2 + x + 1
        result = gaussquadrature(f_cubic, 2, 0.0, 1.0)
        # Analytical: ∫₀¹ (x³ + 2x² + x + 1) dx = [x⁴/4 + 2x³/3 + x²/2 + x]₀¹ = 1/4 + 2/3 + 1/2 + 1 = 29/12
        expected = 1/4 + 2/3 + 1/2 + 1
        @test result ≈ expected atol=1e-12

        # Higher degree polynomial with more points
        f_poly5(x) = x^5 + x^4 + x^3 + x^2 + x + 1
        result = gaussquadrature(f_poly5, 3, 0.0, 1.0)
        # Analytical: ∫₀¹ (x⁵ + x⁴ + x³ + x² + x + 1) dx = [x⁶/6 + x⁵/5 + x⁴/4 + x³/3 + x²/2 + x]₀¹
        expected = 1/6 + 1/5 + 1/4 + 1/3 + 1/2 + 1
        @test result ≈ expected atol=1e-12
    end

    @testset "Transcendental Functions" begin
        # Test exponential function
        f_exp(x) = exp(x)
        result = gaussquadrature(f_exp, 10, 0.0, 1.0)
        expected = exp(1.0) - exp(0.0)  # e - 1
        @test result ≈ expected atol=1e-10

        # Test trigonometric function
        f_sin(x) = sin(x)
        result = gaussquadrature(f_sin, 10, 0.0, Float64(π))
        expected = 2.0  # ∫₀^π sin(x) dx = [-cos(x)]₀^π = -cos(π) + cos(0) = 1 + 1 = 2
        @test result ≈ expected atol=1e-10

        # Test Gaussian function
        f_gauss(x) = exp(-x^2)
        result = gaussquadrature(f_gauss, 20, -2.0, 2.0)
        # Compare with high-precision reference (approximately sqrt(π) ≈ 1.772)
        # Over [-2,2] it should be close to the full integral value
        @test result > 1.7 && result < 1.78
    end

    @testset "Edge Cases" begin
        # Test with zero interval
        f(x) = x^2
        result = gaussquadrature(f, 5, 1.0, 1.0)
        @test result ≈ 0.0 atol=1e-15

        # Test with reversed interval (should give negative result)
        result_forward = gaussquadrature(f, 5, 0.0, 1.0)
        result_backward = gaussquadrature(f, 5, 1.0, 0.0)
        @test result_forward ≈ -result_backward atol=1e-15

        # Test with single point
        result = gaussquadrature(f, 1, 0.0, 2.0)
        @test abs(result - 8/3) < 1.0  # Should be approximate but not exact
    end

end
