using TransportMaps
using Test

@testset "TransportMaps.jl" begin
    @testset "Hermite Polynomials" begin
        # Test univariate Hermite polynomials
        @testset "Univariate Hermite" begin
            # Test first few Hermite polynomials at x = 0
            @test hermite_polynomial(0, 0.0) ≈ 1.0
            @test hermite_polynomial(1, 0.0) ≈ 0.0
            @test hermite_polynomial(2, 0.0) ≈ -1.0
            @test hermite_polynomial(3, 0.0) ≈ 0.0
            
            # Test at x = 1
            @test hermite_polynomial(0, 1.0) ≈ 1.0
            @test hermite_polynomial(1, 1.0) ≈ 1.0
            @test hermite_polynomial(2, 1.0) ≈ 0.0  # x^2 - 1 at x=1
            @test hermite_polynomial(3, 1.0) ≈ -2.0  # x^3 - 3x at x=1
            
            # Test Psi function interface
            @test Psi(0.0, 1.0) ≈ 1.0
            @test Psi(1.0, 1.0) ≈ 1.0
            @test Psi(2.0, 0.0) ≈ -1.0
        end
        
        @testset "Multivariate Hermite" begin
            # Test multivariate Psi
            alpha = [0.0, 1.0]
            x = [1.0, 2.0]
            result = Psi(alpha, x)
            expected = Psi(0.0, 1.0) * Psi(1.0, 2.0)
            @test result ≈ expected
            
            # Test with different dimensions
            alpha3 = [1.0, 0.0, 2.0]
            x3 = [0.5, 1.0, -0.5]
            result3 = Psi(alpha3, x3)
            expected3 = Psi(1.0, 0.5) * Psi(0.0, 1.0) * Psi(2.0, -0.5)
            @test result3 ≈ expected3
        end
        
        @testset "MVBasis Structure" begin
            # Test MVBasis creation
            mvb = MVBasis([1, 2, 0])
            @test mvb.multi_index == [1, 2, 0]
            @test mvb.basis_type isa HermiteBasis
            
            # Test evaluation
            x = [1.0, 0.0, 2.0]
            result = evaluate(mvb, x)
            expected = hermite_polynomial(1, 1.0) * hermite_polynomial(2, 0.0) * hermite_polynomial(0, 2.0)
            @test result ≈ expected
        end
        
        @testset "Multivariate Function f" begin
            # Create basis functions
            mvb1 = MVBasis([0, 0])  # constant
            mvb2 = MVBasis([1, 0])  # x1
            mvb3 = MVBasis([0, 1])  # x2
            
            Psi_vec = [mvb1, mvb2, mvb3]
            coefficients = [1.0, 2.0, 3.0]  # f = 1 + 2*x1 + 3*x2
            
            x = [1.0, 2.0]
            result = f(Psi_vec, coefficients, x)
            expected = 1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 2.0  # 1 + 2 + 6 = 9
            @test result ≈ expected
            
            # Test functional interface
            func = f(Psi_vec, coefficients)
            @test func(x) ≈ expected
        end
        
        @testset "Derivatives" begin
            # Test Hermite polynomial derivatives
            @test hermite_derivative(0, 1.0) ≈ 0.0
            @test hermite_derivative(1, 1.0) ≈ 1.0
            @test hermite_derivative(2, 0.0) ≈ 0.0  # H_2'(x) = 2*H_1(x) = 2*x, at x=0: 2*0 = 0
            @test hermite_derivative(3, 1.0) ≈ 0.0  # H_3'(x) = 3*H_2(x) = 3*(x^2-1), at x=1: 3*(1-1) = 0
            
            # Verify the derivative formula: H_n'(x) = n*H_{n-1}(x)
            @test hermite_derivative(3, 1.0) ≈ 3.0 * hermite_polynomial(2, 1.0)
            
            # Test partial derivatives of MVBasis
            mvb = MVBasis([1, 2])
            x = [1.0, 0.5]
            
            # ∂/∂x1 of (x1 * (x2^2 - 1)) = (x2^2 - 1)
            pd1 = partial_derivative_x(mvb, x, 1)
            expected1 = hermite_derivative(1, 1.0) * hermite_polynomial(2, 0.5)
            @test pd1 ≈ expected1
            
            # ∂/∂x2 of (x1 * (x2^2 - 1)) = x1 * 2*x2
            pd2 = partial_derivative_x(mvb, x, 2)
            expected2 = hermite_polynomial(1, 1.0) * hermite_derivative(2, 0.5)
            @test pd2 ≈ expected2
            
            # Test gradient
            grad = gradient_x(mvb, x)
            @test length(grad) == 2
            @test grad[1] ≈ pd1
            @test grad[2] ≈ pd2
        end
        
        @testset "Function Derivatives" begin
            # Create a simple function f = 2*x1 + 3*x2
            mvb1 = MVBasis([1, 0])  # x1
            mvb2 = MVBasis([0, 1])  # x2
            
            Psi_vec = [mvb1, mvb2]
            coefficients = [2.0, 3.0]
            x = [1.0, 2.0]
            
            # Gradient w.r.t. x should be [2, 3]
            grad_x = gradient_x(Psi_vec, coefficients, x)
            @test grad_x ≈ [2.0, 3.0]
            
            # Gradient w.r.t. coefficients should be the basis function values
            grad_c = gradient_coefficients(Psi_vec, x)
            expected_grad_c = [evaluate(mvb1, x), evaluate(mvb2, x)]
            @test grad_c ≈ expected_grad_c
        end
    end
    
    @testset "Abstract Types" begin
        @test HermiteBasis() isa AbstractPolynomialBasis
        @test HermiteBasis() isa AbstractBasisFunction
        @test AbstractPolynomialBasis <: AbstractBasisFunction
    end
end
