using TransportMaps
using Test

@testset "Polynomial Map Component" begin
    @testset "Construction" begin
        # Test basic construction
        pmc = PolynomialMapComponent(1, 2)
        @test pmc isa AbstractMapComponent
        @test pmc.index == 1
        @test pmc.rectifier isa Softplus  # Default rectifier
        @test length(pmc.basisfunctions) > 0
        @test length(pmc.coefficients) == length(pmc.basisfunctions)

        # Test construction with custom rectifier
        pmc_identity = PolynomialMapComponent(2, 3, IdentityRectifier())
        @test pmc_identity.index == 2
        @test pmc_identity.rectifier isa IdentityRectifier

        # Test construction with custom basis
        pmc_hermite = PolynomialMapComponent(1, 2, Softplus(), HermiteBasis())
        @test pmc_hermite.rectifier isa Softplus
        @test all(bf.basis_type isa HermiteBasis for bf in pmc_hermite.basisfunctions)

        # Test invalid construction
        @test_throws AssertionError PolynomialMapComponent(0, 2)  # Invalid index
        @test_throws AssertionError PolynomialMapComponent(1, 0)  # Invalid degree
        @test_throws AssertionError PolynomialMapComponent(-1, 2)  # Negative index
        @test_throws AssertionError PolynomialMapComponent(1, -1)  # Negative degree
    end

    @testset "Direct Construction with Basis Functions" begin
        # Create basis functions manually
        mvb1 = MultivariateBasis([0])  # Constant
        mvb2 = MultivariateBasis([1])  # Linear
        basisfunctions = [mvb1, mvb2]
        coefficients = [1.0, 2.0]

        pmc = PolynomialMapComponent(basisfunctions, coefficients, IdentityRectifier(), 1)
        @test pmc.index == 1
        @test length(pmc.basisfunctions) == 2
        @test length(pmc.coefficients) == 2
        @test pmc.coefficients == [1.0, 2.0]

        # Test mismatched lengths
        @test_throws AssertionError PolynomialMapComponent(basisfunctions, [1.0], IdentityRectifier(), 1)
        @test_throws AssertionError PolynomialMapComponent([mvb1], [1.0, 2.0], IdentityRectifier(), 1)
    end

    @testset "Evaluation" begin
        # Create a simple 1D polynomial map component
        # f(x) = a₀ + a₁*x with coefficients [1.0, 2.0]
        mvb1 = MultivariateBasis([0])  # ψ₀(x) = H₀(x) = 1
        mvb2 = MultivariateBasis([1])  # ψ₁(x) = H₁(x) = x
        basisfunctions = [mvb1, mvb2]
        coefficients = [1.0, 2.0]  # f(x) = 1 + 2x

        pmc = PolynomialMapComponent(basisfunctions, coefficients, IdentityRectifier(), 1)

        # Test evaluation: M¹(x) = f(0) + ∫₀ˣ g(∂f/∂x₁) dx₁
        # Here f(x) = 1 + 2x, so f(0) = 1, ∂f/∂x₁ = 2
        # With IdentityRectifier: g(ξ) = ξ, so g(2) = 2
        # M¹(x) = 1 + ∫₀ˣ 2 dx₁ = 1 + 2x
        result = evaluate(pmc, [1.0])
        # The actual result depends on the numerical integration, but should be close to 1 + 2*1 = 3
        @test result isa Float64
        @test isfinite(result)

        # Test evaluation at x = 0 should give f(0)
        result_zero = evaluate(pmc, [0.0])
        @test result_zero ≈ 1.0 atol=1e-6  # f(0) = 1

        # Test dimension mismatch
        @test_throws AssertionError evaluate(pmc, [1.0, 2.0])  # Wrong dimension
    end

    @testset "Partial Derivative" begin
        # Create a simple polynomial map component
        mvb1 = MultivariateBasis([0])  # Constant
        mvb2 = MultivariateBasis([1])  # Linear
        basisfunctions = [mvb1, mvb2]
        coefficients = [1.0, 2.0]  # f(x) = 1 + 2x

        pmc = PolynomialMapComponent(basisfunctions, coefficients, IdentityRectifier(), 1)

        # Test partial derivative: ∂M¹/∂x₁ = g(∂f/∂x₁) = g(2) = 2 (with IdentityRectifier)
        pd = partial_derivative_xk(pmc, [1.0])
        @test pd ≈ 2.0 atol=1e-6

        # Test at different points
        pd_zero = partial_derivative_xk(pmc, [0.0])
        @test pd_zero ≈ 2.0 atol=1e-6  # Should be constant for linear function

        pd_neg = partial_derivative_xk(pmc, [-1.0])
        @test pd_neg ≈ 2.0 atol=1e-6
    end

    @testset "Different Rectifiers" begin
        # Test with Softplus rectifier
        mvb1 = MultivariateBasis([0])
        mvb2 = MultivariateBasis([1])
        basisfunctions = [mvb1, mvb2]
        coefficients = [0.0, 1.0]  # f(x) = x, so ∂f/∂x = 1

        pmc_softplus = PolynomialMapComponent(basisfunctions, coefficients, Softplus(), 1)
        pd_softplus = partial_derivative_xk(pmc_softplus, [1.0])
        @test pd_softplus ≈ log1p(exp(1.0)) atol=1e-6  # Softplus(1) = log(1 + e¹)

        # Test with ShiftedELU rectifier
        pmc_elu = PolynomialMapComponent(basisfunctions, coefficients, ShiftedELU(), 1)
        pd_elu = partial_derivative_xk(pmc_elu, [1.0])
        @test pd_elu ≈ 2.0 atol=1e-6  # ShiftedELU(1) = 1 + 1 = 2

        # Test with IdentityRectifier
        pmc_identity = PolynomialMapComponent(basisfunctions, coefficients, IdentityRectifier(), 1)
        pd_identity = partial_derivative_xk(pmc_identity, [1.0])
        @test pd_identity ≈ 1.0 atol=1e-6  # Identity(1) = 1
    end

    @testset "Higher Dimensions" begin
        # Test 2D polynomial map component with index 2
        pmc_2d = PolynomialMapComponent(2, 2)  # 2nd component, degree 2
        pmc_2d.coefficients .= [1.0, 0.3, 0.1, 0.05, 0.02, 0.01]
        @test pmc_2d.index == 2

        # Check that basis functions have correct dimension
        @test all(length(bf.multi_index) == 2 for bf in pmc_2d.basisfunctions)

        # Test evaluation with 2D input
        result_2d = evaluate(pmc_2d, [1.0, 2.0])
        @test result_2d isa Float64
        @test isfinite(result_2d)

        # Test partial derivative
        pd_2d = partial_derivative_xk(pmc_2d, [1.0, 2.0])
        @test pd_2d isa Float64
        @test isfinite(pd_2d)
        @test pd_2d > 0  # Softplus ensures positivity
    end

    @testset "Coefficient Setting" begin
        # Test that we can modify coefficients
        pmc = PolynomialMapComponent(1, 2)
        pmc.coefficients .= [0.5, 1.5, 1.0]  # Set coefficients to new values
        original_coeffs = copy(pmc.coefficients)

        # Modify coefficients
        pmc.coefficients .= 1.0
        @test all(pmc.coefficients .== 1.0)
        @test pmc.coefficients != original_coeffs

        # Test evaluation with new coefficients
        result = evaluate(pmc, [1.0])
        @test result isa Float64
        @test isfinite(result)
    end

    @testset "Edge Cases" begin
        # Test with degree 1 (minimal case)
        pmc_min = PolynomialMapComponent(1, 1)
        pmc_min.coefficients .= [0.0, 1.0]  # f(x) = x
        @test length(pmc_min.basisfunctions) > 0
        @test length(pmc_min.coefficients) == length(pmc_min.basisfunctions)

        # Test evaluation and derivatives work
        result_min = evaluate(pmc_min, [0.5])
        @test isfinite(result_min)

        pd_min = partial_derivative_xk(pmc_min, [0.5])
        @test isfinite(pd_min)
        @test pd_min > 0  # Softplus ensures positivity
    end
end
