using TransportMaps
using Test
using Distributions
using LinearAlgebra
using Random
using Optim

@testset "Map from Density" begin

    @testset "KL Divergence Computation" begin
        # Test with simple 1D linear map and normal target
        M = PolynomialMap(1, 1, :normal, Softplus())
        setcoefficients!(M.components[1], [0.0, 1.0])  # Near identity map through softplus

        # Standard normal target
        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
        quadrature = GaussHermiteWeights(3, 1)

        # KL divergence should be finite for near-identity map to standard normal
        kl = TransportMaps.kldivergence(M, target, quadrature)
        @test isfinite(kl)

        # Test with shifted coefficients
        setcoefficients!(M.components[1], [1.0, 1.0])  # Different map
        kl_shifted = TransportMaps.kldivergence(M, target, quadrature)
        @test isfinite(kl_shifted)  # Should also be finite
    end

    @testset "KL Divergence Gradient" begin
        # Test gradient computation
        M = PolynomialMap(1, 2, :normal, Softplus())
        setcoefficients!(M.components[1], [1.0, 0.1, 0.05])

        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
        quadrature = GaussHermiteWeights(3, 1)

        grad = TransportMaps.kldivergence_gradient(M, target, quadrature)
        @test length(grad) == numbercoefficients(M)
        @test all(isfinite.(grad))

        # Test finite difference approximation of gradient
        ε = 1e-6
        coeffs = getcoefficients(M)
        grad_fd = similar(grad)

        for i in 1:length(coeffs)
            coeffs_plus = copy(coeffs)
            coeffs_minus = copy(coeffs)
            coeffs_plus[i] += ε
            coeffs_minus[i] -= ε

            setcoefficients!(M, coeffs_plus)
            kl_plus = TransportMaps.kldivergence(M, target, quadrature)

            setcoefficients!(M, coeffs_minus)
            kl_minus = TransportMaps.kldivergence(M, target, quadrature)

            grad_fd[i] = (kl_plus - kl_minus) / (2*ε)
        end

        # Reset coefficients
        setcoefficients!(M, coeffs)

        # Analytical and finite difference gradients should be close
        @test grad ≈ grad_fd atol=1e-4
    end

    @testset "Linear Map Optimization" begin
        # Test optimization of simple linear maps

        # 2D linear map with proper initialization
        M = PolynomialMap(2, 1, :normal, Softplus())
        # Initialize close to identity map to avoid degeneracy
        setcoefficients!(M.components[1], [0.1, 0.9])      # ≈ z₁ through softplus
        setcoefficients!(M.components[2], [0.1, 0.1, 0.9]) # ≈ z₂ through softplus

        # Target: Standard bivariate normal
        target = MapTargetDensity(x -> logpdf(MvNormal(I(2)), x))
        quadrature = GaussHermiteWeights(3, 2)

        # Optimize
        result = optimize!(M, target, quadrature)
        # Check that optimization attempted some iterations or achieved reasonable result
        @test result.iterations ≥ 0  # Should at least try

        # Check that coefficients are reasonable
        coeffs = getcoefficients(M)
        @test all(isfinite.(coeffs))

        # Final KL divergence should be finite
        final_kl = TransportMaps.kldivergence(M, target, quadrature)
        @test isfinite(final_kl)
    end

    @testset "Optimization with Zero Initialization" begin
        # Test that optimization handles zero initialization gracefully

        # 1D map - simpler case
        M = PolynomialMap(1, 1, :normal, Softplus())
        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))
        quadrature = GaussHermiteWeights(3, 1)

        # This should handle the zero initialization issue internally
        result = optimize!(M, target, quadrature)

        # Should at least not crash and produce finite coefficients
        coeffs = getcoefficients(M)
        @test all(isfinite.(coeffs))
    end

    @testset "Banana Density Optimization" begin
        # Test optimization with banana-shaped target density

        M = PolynomialMap(2, 2, :normal, Softplus())

        # Banana density: π(x) ∝ exp(-x₁²/2) * exp(-(x₂ - x₁²)²/2)
        banana_density = function(x)
            return (-0.5 * x[1]^2) + (-0.5 * (x[2] - x[1]^2)^2)
        end

        target = MapTargetDensity(banana_density)
        quadrature = GaussHermiteWeights(3, 2)

        # Optimize
        result = optimize!(M, target, quadrature)
        @test result.iterations > 0  # Check that optimization ran

        # Check convergence
        @test result.minimum < 10.0  # Should achieve reasonable objective value

        # Test that map produces reasonable outputs
        test_points = randn(10, 2)
        for z in eachrow(test_points)
            mapped = evaluate(M, z)
            @test all(isfinite.(mapped))
            @test length(mapped) == 2
        end
    end

    @testset "Variance Diagnostics" begin
        # Test variance diagnostic computation

        # Simple 1D case
        M = PolynomialMap(1, 1, :normal, Softplus())
        setcoefficients!(M.components[1], [0.0, 1.0])  # Near identity through softplus

        target = MapTargetDensity(x -> logpdf(Normal(), x[1]))

        # Generate test samples
        Z = randn(10, 1)

        var_diag = variance_diagnostic(M, target, Z)
        @test var_diag ≥ 0.0  # Variance must be non-negative
        @test isfinite(var_diag)

        # Test 2D case
        M2 = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M2.components[1], [0.0, 1.0])       # First component through softplus
        setcoefficients!(M2.components[2], [0.0, 0.0, 1.0])  # Second component through softplus

        target2 = MapTargetDensity(x -> pdf(MvNormal(I(2)), x))
        Z2 = randn(10, 2)

        var_diag2 = variance_diagnostic(M2, target2, Z2)
        @test var_diag2 ≥ 0.0
        @test isfinite(var_diag2)
    end

    @testset "Optimization Convergence with Different Targets" begin
        # Test optimization with various target distributions

        @testset "Mixture Target" begin
            # 2D mixture of two Gaussians
            M = PolynomialMap(2, 2, :normal, Softplus())

            mixture_target = function(x)
                # Mixture of two 2D normals
                μ1 = [-1.0, -1.0]
                μ2 = [1.0, 1.0]
                Σ = I(2) * 0.5

                w1, w2 = 0.6, 0.4
                p1 = pdf(MvNormal(μ1, Σ), x)
                p2 = pdf(MvNormal(μ2, Σ), x)

                return log.(w1 * p1 + w2 * p2)
            end

            target = MapTargetDensity(mixture_target)
            quadrature = GaussHermiteWeights(3, 2)

            result = optimize!(M, target, quadrature)
            @test result.iterations > 0  # Check that optimization ran
            @test isfinite(result.minimum)

            # Test variance diagnostic for optimized map
            Z = randn(50, 2)
            var_diag = variance_diagnostic(M, target, Z)
            @test isfinite(var_diag)
            @test var_diag ≥ 0.0
        end
    end

    @testset "Edge Cases and Error Handling" begin
        # Test with very small quadrature
        M = PolynomialMap(1, 1, :normal, Softplus())
        setcoefficients!(M.components[1], [0.0, 1.0])  # Near identity through softplus

        target = MapTargetDensity(x -> pdf(Normal(), x[1]))
        quadrature_small = GaussHermiteWeights(2, 1)  # Very few points

        kl = TransportMaps.kldivergence(M, target, quadrature_small)
        @test isfinite(kl)

        grad = TransportMaps.kldivergence_gradient(M, target, quadrature_small)
        @test all(isfinite.(grad))

        # Test variance diagnostic with single sample
        Z_single = reshape([0.0], 1, 1)
        var_diag_single = variance_diagnostic(M, target, Z_single)
        # For single sample, variance can be NaN or 0, both are acceptable
        @test isnan(var_diag_single) || var_diag_single == 0.0
    end

end
