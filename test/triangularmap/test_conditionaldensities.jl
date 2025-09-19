using Test
using TransportMaps

# Test setup for conditional densities
@testset "Conditional Densities" begin
    # Create a simple PolynomialMap for testing
    M = PolynomialMap(2, 2, :normal)
    quadrature = SparseSmolyakWeights(2, 2)

    banana(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)
    target = MapTargetDensity(banana, :auto_diff)

    # Optimize the map coefficients
    res = optimize!(M, target, quadrature)

    # Define test inputs
    xₖ₋₁ = [0.5]
    xₖ = 0.8
    zₖ = 0.3

    # Test conditional_density for single value
    @testset "Single Value Density" begin
        @test conditional_density(M, xₖ, xₖ₋₁) ≈ 0.34151562832217836 atol=1e-6
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
        @test conditional_density(M, xₖ, 0.5) ≈ 0.34151562832217836 atol=1e-6
        @test conditional_sample(M, 0.5, zₖ) ≈ conditional_sample(M, 0.5, zₖ) atol=1e-6
    end
end
