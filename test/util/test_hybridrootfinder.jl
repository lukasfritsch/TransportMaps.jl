using TransportMaps
using Test

@testset "Hybrid Root Finder" begin
    @testset "_inverse_bound Function" begin
        # Test with simple linear function
        f_linear(x) = x - 1.0  # Root at x = 1
        lower, upper = TransportMaps._inverse_bound(f_linear)
        @test f_linear(lower) * f_linear(upper) <= 0.0  # Root should be bracketed
        @test lower < upper

        # Test with quadratic function
        f_quad(x) = x^2 - 4.0  # Roots at x = ±2
        lower, upper = TransportMaps._inverse_bound(f_quad)
        @test f_quad(lower) * f_quad(upper) <= 0.0  # Root should be bracketed
        @test lower < upper

        # Test with function that has root at 0
        f_zero(x) = x  # Root at x = 0
        lower, upper = TransportMaps._inverse_bound(f_zero)
        @test f_zero(lower) * f_zero(upper) <= 0.0  # Root should be bracketed
        @test lower <= 0.0 <= upper
    end

    @testset "hybridrootfinder Function - Basic Cases" begin
        # Test with simple linear function
        f_linear(x) = x - 2.0  # Root at x = 2
        ∂f_linear(x) = 1.0    # Derivative is constant 1

        root, fval, dfval = hybridrootfinder(f_linear, ∂f_linear, 0.0, 5.0)
        @test root ≈ 2.0 atol=1e-6
        @test abs(fval) < 1e-6
        @test dfval ≈ 1.0

        # Test with quadratic function (positive root)
        f_quad(x) = x^2 - 9.0  # Roots at x = ±3
        ∂f_quad(x) = 2*x

        # Find positive root
        root, fval, dfval = hybridrootfinder(f_quad, ∂f_quad, 0.0, 5.0)
        @test root ≈ 3.0 atol=1e-6
        @test abs(fval) < 1e-6
        @test dfval ≈ 6.0 atol=1e-6

        # Test with cubic function
        f_cubic(x) = x^3 - 8.0  # Root at x = 2
        ∂f_cubic(x) = 3*x^2

        root, fval, dfval = hybridrootfinder(f_cubic, ∂f_cubic, 1.0, 3.0)
        @test root ≈ 2.0 atol=1e-6
        @test abs(fval) < 1e-6
        @test dfval ≈ 12.0 atol=1e-6
    end

    @testset "Robust Root Finding" begin
        # Test with exponential function: e^x - 2 = 0, root at x = ln(2)
        f_exp(x) = exp(x) - 2.0
        ∂f_exp(x) = exp(x)

        root, fval, dfval = hybridrootfinder(f_exp, ∂f_exp, 0.0, 1.0)
        @test root ≈ log(2.0) atol=1e-6
        @test abs(fval) < 1e-6

        # Test with function where root is at boundary
        f_boundary(x) = x  # Root at x = 0
        ∂f_boundary(x) = 1.0

        root, fval, dfval = hybridrootfinder(f_boundary, ∂f_boundary, -1.0, 1.0)
        @test root ≈ 0.0 atol=1e-6
        @test abs(fval) < 1e-6

        # Test with tight tolerance
        f_test(x) = x^2 - 4.0
        ∂f_test(x) = 2*x

        root, fval, dfval = hybridrootfinder(f_test, ∂f_test, 1.0, 3.0, ftol=1e-12, xtol=1e-12)
        @test root ≈ 2.0 atol=1e-10
        @test abs(fval) < 1e-12
    end

    @testset "Convergence Properties" begin
        # Test convergence with different initial bounds
        f_convergence(x) = x^3 - x - 1.0  # Has root around x ≈ 1.324717957...
        ∂f_convergence(x) = 3*x^2 - 1.0

        # Test with wide bounds
        root1, _, _ = hybridrootfinder(f_convergence, ∂f_convergence, 0.0, 10.0)
        # Test with narrow bounds
        root2, _, _ = hybridrootfinder(f_convergence, ∂f_convergence, 1.0, 2.0)

        @test abs(root1 - root2) < 1e-6  # Should converge to same root
        @test abs(f_convergence(root1)) < 1e-6
        @test abs(f_convergence(root2)) < 1e-6

        # Test function returning proper number of values
        root, fval, dfval = hybridrootfinder(f_convergence, ∂f_convergence, 1.0, 2.0)
        @test isa(root, Real)
        @test isa(fval, Real)
        @test isa(dfval, Real)
    end

    @testset "Warning" begin
        # Test that a warning is issued when derivative is zero at initial guess
        f_flat(x) = x^3 - 3x + 2  # Has root at x = 1
        ∂f_flat(x) = 3x^2 - 3

        @test_warn "Maximum iterations reached in hybridrootfinder" hybridrootfinder(f_flat, ∂f_flat, 1.0, 3.0; maxiter=5)
    end
end
