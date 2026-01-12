using TransportMaps
using Test
using Random
using LinearAlgebra

@testset "Multithreading Support" begin
    Random.seed!(42)

    @testset "Forward Evaluation" begin
        # Create a simple 2D map
        M = PolynomialMap(2, 1, :normal, IdentityRectifier())
        setcoefficients!(M, [0.0, 1.0, 0.0, 0.0, 1.0])  # Identity-like map

        # Test single vector
        z_single = [0.5, 1.0]
        result_single = evaluate(M, z_single)
        @test length(result_single) == 2
        @test all(isfinite.(result_single))

        # Test matrix input
        n_points = 10
        Z_matrix = randn(n_points, 2)
        results_matrix = evaluate(M, Z_matrix)
        @test size(results_matrix) == (n_points, 2)
        @test all(isfinite.(results_matrix))

        # Test consistency between single and matrix evaluation
        for i in 1:n_points
            individual_result = evaluate(M, Z_matrix[i, :])
            @test isapprox(individual_result, results_matrix[i, :], atol=1e-12)
        end
    end

    @testset "Jacobian Computation" begin
        M = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M, abs.(randn(numbercoefficients(M))) .+ 0.1)  # Positive coefficients

        # Test single vector
        z_single = [0.3, 0.7]
        jac_single = jacobian(M, z_single)
        @test jac_single isa Float64
        @test isfinite(jac_single)
        @test jac_single > 0  # Should be positive for Softplus

        # Test matrix input
        n_points = 5
        Z_matrix = randn(n_points, 2) * 0.5  # Keep values moderate
        jacs_matrix = jacobian(M, Z_matrix)
        @test length(jacs_matrix) == n_points
        @test all(isfinite.(jacs_matrix))
        @test all(jacs_matrix .> 0)  # Should be positive for Softplus

        # Test consistency
        for i in 1:n_points
            individual_jac = jacobian(M, Z_matrix[i, :])
            @test isapprox(individual_jac, jacs_matrix[i], atol=1e-10)
        end
    end

    @testset "Inverse Evaluation" begin
        M = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M, abs.(randn(numbercoefficients(M))) .+ 0.1)

        # Generate test points by forward evaluation
        n_points = 5
        Z_original = randn(n_points, 2) * 0.3
        X_test = evaluate(M, Z_original)

        # Test single vector inverse
        x_single = X_test[1, :]
        z_recovered_single = inverse(M, x_single)
        @test length(z_recovered_single) == 2
        @test all(isfinite.(z_recovered_single))

        # Test matrix inverse
        Z_recovered_matrix = inverse(M, X_test)
        @test size(Z_recovered_matrix) == (n_points, 2)
        @test all(isfinite.(Z_recovered_matrix))

        # Test round-trip accuracy
        for i in 1:n_points
            round_trip_error = norm(Z_recovered_matrix[i, :] - Z_original[i, :])
            @test round_trip_error < 1e-6  # Should be very small for good maps
        end

        # Test consistency between single and matrix inverse
        individual_inverse = inverse(M, x_single)
        @test isapprox(individual_inverse, Z_recovered_matrix[1, :], atol=1e-10)
    end

    @testset "Density Functions" begin
        M = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M, abs.(randn(numbercoefficients(M))) .+ 0.1)

        # Define a simple target density
        target_density(x) = exp(-0.5 * sum(x .^ 2))
        target = MapTargetDensity(target_density)

        n_points = 5
        Z_test = randn(n_points, 2) * 0.3
        X_test = evaluate(M, Z_test)

        @testset "Pushforward" begin
            # Test single vector
            z_single = Z_test[1, :]
            pf_single = pushforward(M, target, z_single)
            @test pf_single isa Float64
            @test isfinite(pf_single)
            @test pf_single ≥ 0  # Density should be non-negative

            # Test matrix input
            pfs_matrix = pushforward(M, target, Z_test)
            @test length(pfs_matrix) == n_points
            @test all(isfinite.(pfs_matrix))
            @test all(pfs_matrix .≥ 0)

            # Test consistency
            individual_pf = pushforward(M, target, z_single)
            @test isapprox(individual_pf, pfs_matrix[1], atol=1e-12)
        end

        @testset "Pullback" begin
            # Test single vector
            x_single = X_test[1, :]
            pb_single = pullback(M, x_single)
            @test pb_single isa Float64
            @test isfinite(pb_single)
            @test pb_single ≥ 0  # Density should be non-negative

            # Test matrix input
            pbs_matrix = pullback(M, X_test)
            @test length(pbs_matrix) == n_points
            @test all(isfinite.(pbs_matrix))
            @test all(pbs_matrix .≥ 0)

            # Test consistency
            individual_pb = pullback(M, x_single)
            @test isapprox(individual_pb, pbs_matrix[1], atol=1e-12)
        end
    end

    @testset "Inverse Jacobian" begin
        M = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M, abs.(randn(numbercoefficients(M))) .+ 0.1)

        n_points = 5
        Z_test = randn(n_points, 2) * 0.3
        X_test = evaluate(M, Z_test)

        # Test single vector
        x_single = X_test[1, :]
        inv_jac_single = inverse_jacobian(M, x_single)
        @test inv_jac_single isa Float64
        @test isfinite(inv_jac_single)
        @test inv_jac_single > 0  # Should be positive

        # Test matrix input
        inv_jacs_matrix = inverse_jacobian(M, X_test)
        @test length(inv_jacs_matrix) == n_points
        @test all(isfinite.(inv_jacs_matrix))
        @test all(inv_jacs_matrix .> 0)

        # Test consistency
        individual_inv_jac = inverse_jacobian(M, x_single)
        @test isapprox(individual_inv_jac, inv_jacs_matrix[1], atol=1e-10)

        # Test relationship: inv_jac(M, x) * jac(M, inv(M, x)) ≈ 1
        for i in 1:min(3, n_points)  # Test first few points
            x = X_test[i, :]
            z = inverse(M, x)
            inv_jac = inverse_jacobian(M, x)
            jac = jacobian(M, z)
            @test isapprox(inv_jac * jac, 1.0, atol=1e-8)
        end
    end

    @testset "Gradient Functions" begin
        M = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M, abs.(randn(numbercoefficients(M))) .+ 0.1)

        n_points = 5
        Z_test = randn(n_points, 2) * 0.3

        # Test single vector
        z_single = Z_test[1, :]
        grad_single = gradient_zk(M, z_single)
        @test length(grad_single) == 2
        @test all(isfinite.(grad_single))
        @test all(grad_single .> 0)  # Should be positive for Softplus

        # Test matrix input
        grads_matrix = gradient_zk(M, Z_test)
        @test size(grads_matrix) == (n_points, 2)
        @test all(isfinite.(grads_matrix))
        @test all(grads_matrix .> 0)

        # Test consistency
        individual_grad = gradient_zk(M, z_single)
        @test isapprox(individual_grad, grads_matrix[1, :], atol=1e-12)
    end

    @testset "Dimension Mismatch Errors" begin
        M = PolynomialMap(2, 1, :normal, Softplus())
        setcoefficients!(M, randn(numbercoefficients(M)))

        # Test wrong dimensions for matrix inputs
        @test_throws AssertionError evaluate(M, randn(5, 3))  # Wrong number of columns
        @test_throws AssertionError jacobian(M, randn(5, 1))   # Wrong number of columns
        @test_throws AssertionError inverse(M, randn(5, 3))    # Wrong number of columns
        @test_throws AssertionError pullback(M, randn(5, 1))   # Wrong number of columns

        target_density(x) = 1.0
        target = MapTargetDensity(target_density)
        @test_throws AssertionError pushforward(M, target, randn(5, 3))
    end
end
