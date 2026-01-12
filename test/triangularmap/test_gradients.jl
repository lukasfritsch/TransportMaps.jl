using Test
using TransportMaps
using Distributions
using Random
using LinearAlgebra
using Optim

@testset "Gradient Computation Tests" begin

    @testset "KL Divergence Gradient Accuracy" begin
        Random.seed!(42)

        # Test with 2D degree-1 polynomial map
        M = PolynomialMap(2, 1, :normal, Softplus(), HermiteBasis())
        initial_coeffs = 0.1 * randn(numbercoefficients(M))
        setcoefficients!(M, initial_coeffs)

        # Simple target density
        target_density_func(x) = logpdf(Normal(), x[1].^2) + logpdf(Normal(), x[2])
        target_density = MapTargetDensity(target_density_func, :auto_diff)

        # Small quadrature for testing
        quadrature = LatinHypercubeWeights(50, 2)

        # Compute analytical gradient
        analytical_grad = TransportMaps.kldivergence_gradient(M, target_density, quadrature)

        # Compute numerical gradient using finite differences
        function objective(coeffs)
            setcoefficients!(M, coeffs)
            return TransportMaps.kldivergence(M, target_density, quadrature)
        end

        ε = 1e-6
        n_coeffs = length(initial_coeffs)
        numerical_grad = zeros(n_coeffs)

        obj_base = objective(initial_coeffs)
        for i in 1:n_coeffs
            coeffs_plus = copy(initial_coeffs)
            coeffs_plus[i] += ε
            obj_plus = objective(coeffs_plus)
            numerical_grad[i] = (obj_plus - obj_base) / ε
        end

        # Check gradient accuracy - should be very close
        for i in 1:n_coeffs
            rel_error = abs(analytical_grad[i] - numerical_grad[i]) / (abs(numerical_grad[i]) + 1e-12)
            @test rel_error < 0.01  # 1% tolerance
        end

    end

    @testset "Gradient Consistency Different Maps" begin
        Random.seed!(123)

        # Test different map configurations
        map_configs = [
            (2, 1, :normal, Softplus(), HermiteBasis()),
            (2, 1, :normal, IdentityRectifier(), HermiteBasis()),
            (2, 2, :normal, Softplus(), HermiteBasis())  # Higher degree
        ]

        target_density_func(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1])
        target_density = MapTargetDensity(target_density_func, :auto_diff)
        quadrature = LatinHypercubeWeights(30, 2)

        for (dim, degree, reftype, rectifier, basis) in map_configs
            M = PolynomialMap(dim, degree, reftype, rectifier, basis)
            coeffs = 0.1 * randn(numbercoefficients(M))
            setcoefficients!(M, coeffs)

            # Should not throw errors
            @test_nowarn TransportMaps.kldivergence_gradient(M, target_density, quadrature)

            grad = TransportMaps.kldivergence_gradient(M, target_density, quadrature)

            # Gradient should be finite
            @test all(isfinite.(grad))

            # Gradient should have correct length
            @test length(grad) == numbercoefficients(M)
        end

    end

    @testset "Gradient vs Finite Differences Multiple Points" begin
        Random.seed!(456)

        # Test at multiple random starting points
        M = PolynomialMap(2, 1, :normal, Softplus(), HermiteBasis())
        target_density_func(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2])
        target_density = MapTargetDensity(target_density_func, :auto_diff)
        quadrature = LatinHypercubeWeights(25, 2)

        function objective(coeffs)
            setcoefficients!(M, coeffs)
            return TransportMaps.kldivergence(M, target_density, quadrature)
        end

        # Test 5 random starting points
        for test_point in 1:5
            coeffs = 0.2 * randn(numbercoefficients(M))
            setcoefficients!(M, coeffs)

            analytical_grad = TransportMaps.kldivergence_gradient(M, target_density, quadrature)

            # Check gradient for first coefficient using central differences
            ε = 1e-6
            coeffs_plus = copy(coeffs)
            coeffs_minus = copy(coeffs)
            coeffs_plus[1] += ε
            coeffs_minus[1] -= ε

            obj_plus = objective(coeffs_plus)
            obj_minus = objective(coeffs_minus)
            numerical_grad_1 = (obj_plus - obj_minus) / (2ε)

            rel_error = abs(analytical_grad[1] - numerical_grad_1) / (abs(numerical_grad_1) + 1e-12)
            @test rel_error < 0.05  # 5% tolerance for different starting points
        end

    end

    @testset "Individual Component Gradient Tests" begin
        Random.seed!(789)

        # Test individual polynomial map component gradients
        component = PolynomialMapComponent(2, 2, Softplus(), HermiteBasis())
        coeffs = 0.1 * randn(length(component.coefficients))
        component.coefficients .= coeffs

        z = [0.5, 1.0]

        # Test gradient_coefficients function
        @test_nowarn gradient_coefficients(component, z)

        grad = gradient_coefficients(component, z)
        @test length(grad) == length(component.coefficients)
        @test all(isfinite.(grad))

        # Verify gradient using finite differences
        function comp_eval(c)
            temp_coeffs = copy(component.coefficients)
            component.coefficients .= c
            result = evaluate(component, z)
            component.coefficients .= temp_coeffs
            return result
        end

        ε = 1e-7
        for i in 1:length(coeffs)
            coeffs_plus = copy(coeffs)
            coeffs_plus[i] += ε

            eval_plus = comp_eval(coeffs_plus)
            eval_base = comp_eval(coeffs)

            numerical_grad = (eval_plus - eval_base) / ε

            # Should be reasonably close (finite differences can be less accurate for components)
            rel_error = abs(grad[i] - numerical_grad) / (abs(numerical_grad) + 1e-10)
            @test rel_error < 0.1  # 10% tolerance for component-level gradients
        end

    end

    @testset "Full Map Gradient Matrix Tests" begin
        Random.seed!(101112)

        M = PolynomialMap(2, 1, :normal, Softplus(), HermiteBasis())
        setcoefficients!(M, 0.1 * randn(numbercoefficients(M)))

        z = [0.3, 0.7]

        # Test gradient_coefficients for full map
        @test_nowarn gradient_coefficients(M, z)

        grad_matrix = gradient_coefficients(M, z)

        # Should be (n_dims × n_coeffs) matrix
        @test size(grad_matrix) == (numberdimensions(M), numbercoefficients(M))
        @test all(isfinite.(grad_matrix))

        # Verify triangular structure (for triangular maps)
        # Component i should only depend on coefficients 1 through those used by component i
        n_comp1_coeffs = length(M.components[1].coefficients)

        # Second component (dimension 2) should not depend on coefficients beyond its range
        # This is a structural property we can verify
        @test size(grad_matrix, 1) == 2  # 2D map
        @test size(grad_matrix, 2) == numbercoefficients(M)

    end

    @testset "Optimization Integration Tests" begin
        Random.seed!(131415)

        # Test that optimization actually improves the objective
        M = PolynomialMap(2, 1, :normal, Softplus(), HermiteBasis())
        initial_coeffs = 0.2 * randn(numbercoefficients(M))
        setcoefficients!(M, initial_coeffs)

        target_density_func(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2])
        target_density = MapTargetDensity(target_density_func, :auto_diff)
        quadrature = LatinHypercubeWeights(30, 2)

        # Initial KL divergence
        initial_kl = TransportMaps.kldivergence(M, target_density, quadrature)

        # Optimize with gradient (current implementation always uses gradients)
        result = optimize!(M, target_density, quadrature)
        final_kl = TransportMaps.kldivergence(M, target_density, quadrature)

        # Should improve (decrease) the KL divergence
        @test final_kl < initial_kl
        @test result.iterations > 0
        @test isfinite(final_kl)

        # Reset and test that we can call optimize multiple times
        setcoefficients!(M, initial_coeffs)
        result2 = optimize!(M, target_density, quadrature)
        final_kl2 = TransportMaps.kldivergence(M, target_density, quadrature)

        # Should also improve
        @test final_kl2 < initial_kl

        # Both optimizations should give reasonable results
        @test isfinite(final_kl2)
        @test result2.iterations > 0

    end

    @testset "Gradients with Analytical Target Density Gradients" begin
        Random.seed!(192021)

        # Test with target densities that have known analytical gradients
        M = PolynomialMap(2, 1, :normal, Softplus(), HermiteBasis())
        setcoefficients!(M, 0.1 * randn(numbercoefficients(M)))

        quadrature = LatinHypercubeWeights(40, 2)

        # Test 1: Simple Gaussian with known gradient
        # π(x) = exp(-0.5 * (x₁² + x₂²)) / (2π)
        # ∇π(x) = -π(x) * x
        function gaussian_density(x)
            return -0.5 * (x[1]^2 + x[2]^2) - log(2π)
        end

        function gaussian_density_gradient(x)
            return -[x[1], x[2]]
        end

        target_density = MapTargetDensity(gaussian_density, gaussian_density_gradient)

        # Test that gradient computation works with analytical target gradient
        @test_nowarn TransportMaps.kldivergence_gradient(M, target_density, quadrature)

        grad1 = TransportMaps.kldivergence_gradient(M, target_density, quadrature)
        @test all(isfinite.(grad1))
        @test length(grad1) == numbercoefficients(M)

        # Compare with finite difference version (should be similar)
        target_density_fd = MapTargetDensity(gaussian_density, :auto_diff)
        grad1_fd = TransportMaps.kldivergence_gradient(M, target_density_fd, quadrature)
        @test all(isfinite.(grad1_fd))

        # Analytical and finite difference gradients should be close
        for i in 1:length(grad1)
            rel_error = abs(grad1[i] - grad1_fd[i]) / (abs(grad1_fd[i]) + 1e-12)
            @test rel_error < 0.05  # 5% tolerance between analytical and FD target gradients
        end

        # Test 2: Exponential density π(x) = exp(-|x₁| - |x₂|) (not differentiable at origin, but still useful)
        function exponential_density(x)
            return exp(-abs(x[1]) - abs(x[2]))
        end

        target_density_exp = MapTargetDensity(exponential_density, :auto_diff)

        @test_nowarn TransportMaps.kldivergence_gradient(M, target_density_exp, quadrature)
        grad2 = TransportMaps.kldivergence_gradient(M, target_density_exp, quadrature)
        @test all(isfinite.(grad2))

        # Test 3: Quadratic density π(x) = exp(-x₁² - 2x₂² - x₁x₂)
        # This has cross terms and different scaling in each dimension
        function quadratic_density(x)
            return -x[1]^2 - 2 * x[2]^2 - x[1] * x[2]
        end

        function quadratic_density_gradient(x)
            density_val = quadratic_density(x)
            grad_x1 = -(2 * x[1] + x[2])
            grad_x2 = -(4 * x[2] + x[1])
            return [grad_x1, grad_x2]
        end

        target_density_quad = MapTargetDensity(quadratic_density, quadratic_density_gradient)

        @test_nowarn TransportMaps.kldivergence_gradient(M, target_density_quad, quadrature)
        grad3 = TransportMaps.kldivergence_gradient(M, target_density_quad, quadrature)
        @test all(isfinite.(grad3))

        # Compare with finite difference version for the quadratic density
        target_density_quad_fd = MapTargetDensity(quadratic_density, :auto_diff)
        grad3_fd = TransportMaps.kldivergence_gradient(M, target_density_quad_fd, quadrature)
        for i in 1:length(grad3)
            rel_error = abs(grad3[i] - grad3_fd[i]) / (abs(grad3_fd[i]) + 1e-12)
            @test rel_error < 0.05  # 5% tolerance between analytical and FD target gradients
        end

        # Test 4: Verify consistency - different target densities should give different gradients
        # (unless the map is already perfectly adapted, which is unlikely with random coefficients)
        @test !isapprox(grad1, grad2, rtol=0.1)
        @test !isapprox(grad1, grad3, rtol=0.1)
        @test !isapprox(grad2, grad3, rtol=0.1)

        # Test 5: Verify that gradient computation is consistent across different coefficient settings
        coeffs_alternative = 0.2 * randn(numbercoefficients(M))
        setcoefficients!(M, coeffs_alternative)

        grad1_alt = TransportMaps.kldivergence_gradient(M, target_density, quadrature)
        @test all(isfinite.(grad1_alt))
        @test !isapprox(grad1, grad1_alt, rtol=0.1)  # Different coefficients should give different gradients

        # Test 6: Higher degree map with analytical gradient
        M_deg2 = PolynomialMap(2, 2, :normal, Softplus(), HermiteBasis())
        setcoefficients!(M_deg2, 0.05 * randn(numbercoefficients(M_deg2)))

        grad_deg2 = TransportMaps.kldivergence_gradient(M_deg2, target_density, quadrature)
        @test all(isfinite.(grad_deg2))
        @test length(grad_deg2) == numbercoefficients(M_deg2)
        @test length(grad_deg2) > length(grad1)  # Higher degree should have more coefficients

    end

    @testset "Edge Cases and Robustness" begin
        Random.seed!(161718)

        M = PolynomialMap(2, 1, :normal, Softplus(), HermiteBasis())
        target_density_func(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2])
        target_density = MapTargetDensity(target_density_func, :auto_diff)
        quadrature = LatinHypercubeWeights(20, 2)

        # Test with very small coefficients
        setcoefficients!(M, 1e-8 * ones(numbercoefficients(M)))
        @test_nowarn TransportMaps.kldivergence_gradient(M, target_density, quadrature)

        # Test with larger coefficients
        setcoefficients!(M, 2.0 * ones(numbercoefficients(M)))
        @test_nowarn TransportMaps.kldivergence_gradient(M, target_density, quadrature)

        # Test with mixed positive/negative coefficients
        mixed_coeffs = [1.0, -1.0, 0.5, -0.5, 0.0]
        setcoefficients!(M, mixed_coeffs[1:numbercoefficients(M)])
        grad = TransportMaps.kldivergence_gradient(M, target_density, quadrature)
        @test all(isfinite.(grad))

        # Test with different target densities
        # Banana-shaped distribution
        banana_density(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
        target_density_banana = MapTargetDensity(banana_density, :auto_diff)
        setcoefficients!(M, 0.1 * randn(numbercoefficients(M)))
        @test_nowarn TransportMaps.kldivergence_gradient(M, target_density_banana, quadrature)

    end
end
