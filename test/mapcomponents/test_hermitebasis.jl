using TransportMaps
using Test
using Statistics

@testset "Hermite Polynomials and Multivariate Basis" begin
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

            # Test basisfunction interface (HermiteBasis default)
            @test basisfunction(HermiteBasis(:none), 0.0, 1.0) ≈ 1.0
            @test basisfunction(HermiteBasis(:none), 1.0, 1.0) ≈ 1.0
            @test basisfunction(HermiteBasis(:none), 2.0, 0.0) ≈ -1.0

            # Test edge-controlled Hermite polynomials
            @test basisfunction(HermiteBasis(:gaussian), 1.0, 2.0) ≈ hermite_polynomial(1, 2.0) * exp(-.25 * 2.0^2)
            @test basisfunction(HermiteBasis(:cubic), 2.0, 1.0) ≈ hermite_polynomial(2, 1.0) * (2 * (min(1.0, abs(1.0)/4.0))^3 - 3 * (min(1.0, abs(1.0)/4.0))^2 + 1)
        end

        @testset "Multivariate Hermite" begin
            # Test multivariate Psi (using HermiteBasis)
            alpha = [0.0, 1.0]
            x = [1.0, 2.0]
            result = Psi(alpha, x, HermiteBasis(:none))
            expected = basisfunction(HermiteBasis(:none), 0.0, 1.0) * basisfunction(HermiteBasis(:none), 1.0, 2.0)
            @test result ≈ expected

            # Test with different dimensions
            alpha3 = [1.0, 0.0, 2.0]
            x3 = [0.5, 1.0, -0.5]
            result3 = Psi(alpha3, x3, HermiteBasis(:none))
            expected3 = basisfunction(HermiteBasis(:none), 1.0, 0.5) * basisfunction(HermiteBasis(:none), 0.0, 1.0) * basisfunction(HermiteBasis(:none), 2.0, -0.5)
            @test result3 ≈ expected3
        end

        @testset "MultivariateBasis Structure" begin
            # Test MultivariateBasis creation
            mvb = MultivariateBasis([1, 2, 0], HermiteBasis())
            @test mvb.multi_index == [1, 2, 0]
            @test mvb.basis_type isa HermiteBasis

            # Test evaluation (default Hermite)
            x = [1.0, 0.0, 2.0]
            result = evaluate(mvb, x)
            expected = hermite_polynomial(1, 1.0) * hermite_polynomial(2, 0.0) * hermite_polynomial(0, 2.0)
            @test result ≈ expected

            # Test evaluation with edge control
            mvb_gauss = MultivariateBasis([1, 2, 0], HermiteBasis(:gaussian))
            result_gauss = evaluate(mvb_gauss, x)
            expected_gauss = edge_controlled_hermite_polynomial(1, 1.0, :gaussian) * edge_controlled_hermite_polynomial(2, 0.0, :gaussian) * edge_controlled_hermite_polynomial(0, 2.0, :gaussian)
            @test result_gauss ≈ expected_gauss
        end

        @testset "Multivariate Function f" begin
            # Create basis functions
            mvb1 = MultivariateBasis([0, 0], HermiteBasis())  # constant
            mvb2 = MultivariateBasis([1, 0], HermiteBasis())  # x1
            mvb3 = MultivariateBasis([0, 1], HermiteBasis())  # x2

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

            # Test partial derivatives of MultivariateBasis
            mvb = MultivariateBasis([1, 2], HermiteBasis())
            x = [1.0, 0.5]

            # ∂/∂x1 of (x1 * (x2^2 - 1)) = (x2^2 - 1)
            pd1 = partial_derivative_z(mvb, x, 1)
            expected1 = edge_controlled_hermite_derivative(1, 1.0, mvb.basis_type.edge_control) * edge_controlled_hermite_polynomial(2, 0.5, mvb.basis_type.edge_control)
            @test pd1 ≈ expected1

            # ∂/∂x2 of (x1 * (x2^2 - 1)) = x1 * 2*x2
            pd2 = partial_derivative_z(mvb, x, 2)
            expected2 = edge_controlled_hermite_polynomial(1, 1.0, mvb.basis_type.edge_control) * edge_controlled_hermite_derivative(2, 0.5, mvb.basis_type.edge_control)
            @test pd2 ≈ expected2

            # Test gradient
            grad = gradient_z(mvb, x)
            @test length(grad) == 2
            @test grad[1] ≈ pd1
            @test grad[2] ≈ pd2
        end

        @testset "Function Derivatives" begin
            # Create a simple function f = 2*x1 + 3*x2
            mvb1 = MultivariateBasis([1, 0], HermiteBasis())  # x1
            mvb2 = MultivariateBasis([0, 1], HermiteBasis())  # x2

            Psi_vec = [mvb1, mvb2]
            coefficients = [2.0, 3.0]
            x = [1.0, 2.0]

            # Gradient w.r.t. x should be [2, 3]
            grad_z = gradient_z(Psi_vec, coefficients, x)
            @test grad_z ≈ [2.0, 3.0]

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

@testset "Linearized HermiteBasis" begin
    # Create samples from a normal distribution
    samples = randn(1000)
    max_degree = 4
    k = 2
    basis = LinearizedHermiteBasis(samples, max_degree, k)

    # Test bounds are set to quantiles
    lower, upper = basis.bounds_linearization
    @test isapprox(lower, quantile(samples, 0.01); atol=1e-8)
    @test isapprox(upper, quantile(samples, 0.99); atol=1e-8)

    # Test normalization for k and not k
    for n in 0:max_degree
        if n == k
            @test basis.normalization[n+1] == factorial(n+1)
        else
            @test basis.normalization[n+1] == factorial(n)
        end
    end

    # Test piecewise polynomial and derivative
    n = 3
    z_a = lower - 1.0
    z_b = upper + 1.0
    z_mid = (lower + upper) / 2
    # Left linear region
    ψ_left = basisfunction(basis, n, z_a)
    ψ_left_expected = hermite_polynomial(n, lower) + hermite_derivative(n, lower) * (z_a - lower)
    ψ_left_expected /= sqrt(basis.normalization[n])
    @test isapprox(ψ_left, ψ_left_expected; atol=1e-10)
    # Right linear region
    ψ_right = basisfunction(basis, n, z_b)
    ψ_right_expected = hermite_polynomial(n, upper) + hermite_derivative(n, upper) * (z_b - upper)
    ψ_right_expected /= sqrt(basis.normalization[n])
    @test isapprox(ψ_right, ψ_right_expected; atol=1e-10)
    # Middle region
    ψ_mid = basisfunction(basis, n, z_mid)
    ψ_mid_expected = hermite_polynomial(n, z_mid) / sqrt(basis.normalization[n])
    @test isapprox(ψ_mid, ψ_mid_expected; atol=1e-10)

    # Derivative left
    dψ_left = basisfunction_derivative(basis, n, z_a)
    dψ_left_expected = hermite_derivative(n, lower) / sqrt(basis.normalization[n])
    @test isapprox(dψ_left, dψ_left_expected; atol=1e-10)
    # Derivative right
    dψ_right = basisfunction_derivative(basis, n, z_b)
    dψ_right_expected = hermite_derivative(n, upper) / sqrt(basis.normalization[n])
    @test isapprox(dψ_right, dψ_right_expected; atol=1e-10)
    # Derivative mid
    dψ_mid = basisfunction_derivative(basis, n, z_mid)
    dψ_mid_expected = hermite_derivative(n, z_mid) / sqrt(basis.normalization[n])
    @test isapprox(dψ_mid, dψ_mid_expected; atol=1e-10)
end
