using Test
using TransportMaps
using Distributions

# Test setup for conditional densities
@testset "Conditional Densities" begin
    # Create a simple PolynomialMap for testing
    M = PolynomialMap(2, 2, :normal)
    quadrature = SparseSmolyakWeights(2, 2)

    banana(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
    target = MapTargetDensity(banana)

    # Optimize the map coefficients
    res = optimize!(M, target, quadrature)

    # Define test inputs
    xₖ₋₁ = [0.5]
    xₖ = 0.8
    zₖ = 0.3

    # Test conditional_density for single value
    @testset "Single Value Density" begin
        @test conditional_density(M, xₖ, xₖ₋₁) ≈ 0.34151562832217836 atol = 1e-6
    end

    # Test conditional_density for multiple values
    @testset "Multiple Values Density" begin
        xₖ_values = [0.8, 0.9, 1.0]
        densities = conditional_density(M, xₖ_values, xₖ₋₁)
        @test length(densities) == length(xₖ_values)
    end

    # Test conditional_sample for single value
    @testset "Single Value Sampling" begin
        sample = conditional_sample(M, xₖ₋₁, zₖ)
        @test isa(sample, Number)
    end

    # Test conditional_sample for multiple values
    @testset "Multiple Values Sampling" begin
        zₖ_values = [0.3, 0.4, 0.5]
        samples = conditional_sample(M, xₖ₋₁, zₖ_values)
        @test length(samples) == length(zₖ_values)
    end

    # Edge case: xₖ₋₁ as a single value
    @testset "Edge Case" begin
        @test conditional_density(M, xₖ, 0.5) ≈ 0.34151562832217836 atol = 1e-6
        @test conditional_sample(M, 0.5, zₖ) ≈ conditional_sample(M, 0.5, zₖ) atol = 1e-6
    end
end

# Test multivariate conditional densities
@testset "Multivariate Conditional Densities" begin
    # Create a 3-dimensional PolynomialMap for testing (more stable than 4D)
    M3 = PolynomialMap(3, 2, :normal)
    quadrature = SparseSmolyakWeights(2, 3)

    # Define a 3D target density (extending the banana example)
    banana3d(x) = logpdf(Normal(), x[1]) +
                  logpdf(Normal(), x[2] - 0.5 * x[1]^2) +
                  logpdf(Normal(), x[3] - 0.3 * x[2])
    target3d = MapTargetDensity(banana3d)

    # Optimize the map coefficients
    res3d = optimize!(M3, target3d, quadrature)

    # Test inputs
    x_given = [0.5]      # x₁
    x_range = [0.3, 0.8] # x₂, x₃
    x_full = [0.5, 0.3, 0.8]  # x₁, x₂, x₃
    z_range = [0.2, 0.4] # z₂, z₃

    @testset "Joint Density (Full Vector)" begin
        # Test multivariate_conditional_density with full vector
        joint_density = multivariate_conditional_density(M3, x_full)
        @test isa(joint_density, Real)
        @test joint_density > 0

        # Compare with manual computation: p(x₁) * p(x₂|x₁) * p(x₃|x₁,x₂)
        manual_density = conditional_density(M3, x_full[1], Float64[])  # marginal p(x₁)
        manual_density *= conditional_density(M3, x_full[2], x_full[1:1])  # p(x₂|x₁)
        manual_density *= conditional_density(M3, x_full[3], x_full[1:2])  # p(x₃|x₁,x₂)

        @test joint_density ≈ manual_density atol = 1e-10
    end

    @testset "Conditional Density (Range Given)" begin
        # Test p(x₂, x₃ | x₁)
        cond_density = multivariate_conditional_density(M3, x_range, x_given)
        @test isa(cond_density, Real)
        @test cond_density > 0

        # Compare with manual computation: p(x₂|x₁) * p(x₃|x₁,x₂)
        manual_cond = conditional_density(M3, x_range[1], x_given)  # p(x₂|x₁)
        manual_cond *= conditional_density(M3, x_range[2], [x_given..., x_range[1]])  # p(x₃|x₁,x₂)

        @test cond_density ≈ manual_cond atol = 1e-10
    end

    @testset "Single Dimensional Cases" begin
        # Test 1D case (should be marginal density)
        x1 = [0.5]
        density_1d = multivariate_conditional_density(M3, x1)
        marginal_density = conditional_density(M3, x1[1], Float64[])
        @test density_1d ≈ marginal_density atol = 1e-10

        # Test single range with single given
        single_range = [0.8]
        single_given = [0.5]
        cond_single = multivariate_conditional_density(M3, single_range, single_given)
        expected_single = conditional_density(M3, single_range[1], single_given)
        @test cond_single ≈ expected_single atol = 1e-10
    end

    @testset "Convenience Function Variants" begin
        # Test with single given value (Float64)
        cond_density_float = multivariate_conditional_density(M3, x_range, x_given[1])
        expected_float = multivariate_conditional_density(M3, x_range, [x_given[1]])
        @test cond_density_float ≈ expected_float atol = 1e-10

        # Test with AbstractArray inputs
        x_range_array = convert(Vector{Float32}, x_range)
        x_given_array = convert(Vector{Float32}, x_given)
        cond_density_array = multivariate_conditional_density(M3, x_range_array, x_given_array)
        expected_array = multivariate_conditional_density(M3, x_range, x_given)
        @test cond_density_array ≈ expected_array atol = 1e-5
    end

    @testset "Multivariate Conditional Sampling" begin
        # Test basic sampling functionality
        samples = multivariate_conditional_sample(M3, x_given, z_range)
        @test length(samples) == length(z_range)
        @test all(isa.(samples, Real))

        # Compare with manual sequential sampling
        manual_samples = Float64[]
        x_current = copy(x_given)
        for z_val in z_range
            sample = conditional_sample(M3, x_current, z_val)
            push!(manual_samples, sample)
            push!(x_current, sample)
        end
        @test samples ≈ manual_samples atol = 1e-10

        # Test with single given value
        samples_single = multivariate_conditional_sample(M3, x_given[1], z_range)
        expected_single = multivariate_conditional_sample(M3, [x_given[1]], z_range)
        @test samples_single ≈ expected_single atol = 1e-10

        # Test with AbstractArray inputs
        z_range_array = convert(Vector{Float32}, z_range)
        samples_array = multivariate_conditional_sample(M3, x_given, z_range_array)
        @test length(samples_array) == length(z_range)
    end

    @testset "Error Handling" begin
        # Test dimension bounds checking
        @test_throws AssertionError multivariate_conditional_density(M3, [1.0, 2.0, 3.0, 4.0])  # Too many dimensions
        @test_throws AssertionError multivariate_conditional_density(M3, [1.0], [1.0, 2.0, 3.0])  # x_given too long
        @test_throws AssertionError multivariate_conditional_sample(M3, [1.0, 2.0, 3.0], [0.1])  # x_given too long

        # Test empty x_range
        @test_throws AssertionError multivariate_conditional_density(M3, Float64[], x_given)
        @test_throws AssertionError multivariate_conditional_sample(M3, x_given, Float64[])
    end

    @testset "Consistency with Univariate Functions" begin
        # For single variables, multivariate should match univariate
        x_single_range = [0.8]
        x_single_given = [0.5]

        multivar_result = multivariate_conditional_density(M3, x_single_range, x_single_given)
        univar_result = conditional_density(M3, x_single_range[1], x_single_given)
        @test multivar_result ≈ univar_result atol = 1e-10

        # Same for sampling
        z_single = [0.3]
        multivar_sample = multivariate_conditional_sample(M3, x_single_given, z_single)
        univar_sample = [conditional_sample(M3, x_single_given, z_single[1])]
        @test multivar_sample ≈ univar_sample atol = 1e-10
    end

    @testset "Mathematical Properties" begin
        # Test that joint density factors correctly
        # p(x₁,x₂,x₃) = p(x₁,x₂) * p(x₃|x₁,x₂)
        x123 = [0.5, 0.3, 0.8]
        x12 = [0.5, 0.3]
        x3_given_12 = [0.8]

        joint_123 = multivariate_conditional_density(M3, x123)
        joint_12 = multivariate_conditional_density(M3, x12)
        cond_3_given_12 = multivariate_conditional_density(M3, x3_given_12, x12)

        @test joint_123 ≈ joint_12 * cond_3_given_12 atol = 1e-10

        # Test that densities are positive and finite
        x_range_test = [0.0, 0.0]  # Test point
        density_test = multivariate_conditional_density(M3, x_range_test, x_given)
        @test density_test > 0 && isfinite(density_test)
    end

    @testset "Input Types" begin
        # Test conditional_density with various input type combinations
        @testset "conditional_density - Float64 x_range, AbstractArray x_given" begin
            x_range_float = 0.8
            x_given_array = [2]
            result = conditional_density(M3, x_range_float, x_given_array)
            @test isa(result, Float64)
            @test result > 0

            # Verify it matches the standard call
            expected = conditional_density(M3, x_range_float, float.(x_given_array))
            @test result ≈ expected atol = 1e-10
        end

        @testset "conditional_density - AbstractArray x_range, AbstractArray x_given" begin
            x_range_array = [1, 2]
            x_given_array = [2]
            results = conditional_density(M3, x_range_array, x_given_array)
            @test isa(results, AbstractArray)
            @test length(results) == length(x_range_array)
            @test all(r -> r > 0, results)

            # Verify each result matches individual calls
            for (i, xr) in enumerate(x_range_array)
                expected = conditional_density(M3, float(xr), x_given_array)
                @test results[i] ≈ expected atol = 1e-10
            end
        end

        @testset "conditional_density - AbstractArray x_range, Float64 x_given" begin
            x_range_array = [2, 3]
            x_given_float = 0.5
            results = conditional_density(M3, x_range_array, x_given_float)
            @test isa(results, AbstractArray)
            @test length(results) == length(x_range_array)
            @test all(r -> r > 0, results)

            # Verify it matches the call with x_given as array
            expected = conditional_density(M3, x_range_array, [x_given_float])
            @test results ≈ expected atol = 1e-10
        end

        @testset "conditional_sample - AbstractArray x_given, Float64 z_range" begin
            x_given_array = [3]
            z_float = 0.3
            result = conditional_sample(M3, x_given_array, z_float)
            @test isa(result, Real)

            # Verify consistency - calling twice with same inputs should give same result
            result2 = conditional_sample(M3, float.(x_given_array), z_float)
            @test result ≈ result2 atol = 1e-10
        end

        @testset "conditional_sample - Float64 x_given, AbstractArray z_range" begin
            x_given_float = 0.5
            z_array = [1, 2]
            results = conditional_sample(M3, x_given_float, z_array)
            @test isa(results, AbstractArray)
            @test length(results) == length(z_array)

            # Verify it matches the call with x_given as array
            expected = conditional_sample(M3, [x_given_float], z_array)
            @test results ≈ expected atol = 1e-10
        end

        @testset "conditional_sample - AbstractArray x_given, AbstractArray z_range" begin
            x_given_array = Float32[0.5]
            z_array = Float32[0.3, 0.4]
            results = conditional_sample(M3, x_given_array, z_array)
            @test isa(results, AbstractArray)
            @test length(results) == length(z_array)

            # Verify each result is consistent
            for (i, z) in enumerate(z_array)
                individual = conditional_sample(M3, Float64.(x_given_array), Float64(z))
                @test results[i] ≈ individual atol = 1e-5
            end
        end

        @testset "multivariate_conditional_density - AbstractArray x_range, Float64 x_given" begin
            x_range_array = [1, 2]
            x_given_float = 0.5
            result = multivariate_conditional_density(M3, x_range_array, x_given_float)
            @test isa(result, Real)
            @test result > 0

            # Verify it matches the call with x_given as array
            expected = multivariate_conditional_density(M3, float.(x_range_array), [x_given_float])
            @test result ≈ expected atol = 1e-10
        end

        @testset "multivariate_conditional_density - AbstractArray x (full vector)" begin
            x_full_array = [1, 2, 3]
            result = multivariate_conditional_density(M3, x_full_array)
            @test isa(result, Real)
            @test result > 0

            # Verify it matches manual computation
            manual = conditional_density(M3, float(x_full_array[1]), Float64[])
            manual *= conditional_density(M3, float(x_full_array[2]), float(x_full_array[1:1]))
            manual *= conditional_density(M3, float(x_full_array[3]), float(x_full_array[1:2]))
            @test result ≈ manual atol = 1e-10
        end

        @testset "multivariate_conditional_sample - Float64 x_given, AbstractArray z_range" begin
            x_given_float = 0.5
            z_array = Float32[0.2, 0.4]
            results = multivariate_conditional_sample(M3, x_given_float, z_array)
            @test isa(results, AbstractArray)
            @test length(results) == length(z_array)

            # Verify it matches the call with x_given as array
            expected = multivariate_conditional_sample(M3, [x_given_float], Float64.(z_array))
            @test results ≈ expected atol = 1e-10
        end

        @testset "Explicit Vector{Float64} inputs" begin
            # Test with explicitly typed Vector{Float64}
            x_range_vec = Vector{Float64}([0.8, 0.9])
            x_given_vec = Vector{Float64}([0.5])
            z_range_vec = Vector{Float64}([0.3, 0.4])

            # conditional_density with explicit vectors
            dens_result = conditional_density(M3, x_range_vec, x_given_vec)
            @test isa(dens_result, AbstractArray)
            @test length(dens_result) == 2

            # conditional_sample with explicit vectors
            samp_result = conditional_sample(M3, x_given_vec, z_range_vec)
            @test isa(samp_result, AbstractArray)
            @test length(samp_result) == 2

            # multivariate_conditional_density with explicit vectors
            mv_dens_result = multivariate_conditional_density(M3, x_range_vec, x_given_vec)
            @test isa(mv_dens_result, Real)
            @test mv_dens_result > 0

            # multivariate_conditional_sample with explicit vectors
            mv_samp_result = multivariate_conditional_sample(M3, x_given_vec, z_range_vec)
            @test isa(mv_samp_result, AbstractArray)
            @test length(mv_samp_result) == 2
        end

        @testset "Mixed numeric types" begin
            # Test with Float32 arrays
            x_range_f32 = Float32[0.8, 0.9]
            x_given_f32 = Float32[0.5]
            z_range_f32 = Float32[0.3, 0.4]

            # Should work with Float32 inputs
            dens_f32 = conditional_density(M3, x_range_f32, x_given_f32)
            @test isa(dens_f32, AbstractArray)

            samp_f32 = conditional_sample(M3, x_given_f32, z_range_f32)
            @test isa(samp_f32, AbstractArray)

            mv_dens_f32 = multivariate_conditional_density(M3, x_range_f32, x_given_f32)
            @test isa(mv_dens_f32, Real)

            mv_samp_f32 = multivariate_conditional_sample(M3, x_given_f32, z_range_f32)
            @test isa(mv_samp_f32, AbstractArray)

            # Results should be close to Float64 versions
            dens_f64 = conditional_density(M3, Float64.(x_range_f32), Float64.(x_given_f32))
            @test dens_f32 ≈ dens_f64 atol = 1e-5
        end
    end
end
