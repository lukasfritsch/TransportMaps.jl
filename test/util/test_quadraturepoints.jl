using TransportMaps
using Test

@testset "Quadrature Points" begin
    @testset "GaussHermiteWeights" begin
        # Test basic construction
        ghw = GaussHermiteWeights(5, 2)
        @test ghw isa AbstractQuadratureWeights
        @test size(ghw.points, 1) == 2  # Dimension
        @test size(ghw.points, 2) == 25  # 5^2 points for 2D
        @test length(ghw.weights) == 25
        
        # Test 1D case
        ghw_1d = GaussHermiteWeights(3, 1)
        @test size(ghw_1d.points, 1) == 1
        @test size(ghw_1d.points, 2) == 3
        @test length(ghw_1d.weights) == 3
        
        # Test 3D case
        ghw_3d = GaussHermiteWeights(4, 3)
        @test size(ghw_3d.points, 1) == 3
        @test size(ghw_3d.points, 2) == 64  # 4^3 points
        @test length(ghw_3d.weights) == 64
        
        # Test weight properties (weights should be positive and finite)
        ghw_test = GaussHermiteWeights(10, 1)
        @test all(ghw_test.weights .> 0)
        @test all(isfinite.(ghw_test.weights))
        
        ghw_test_2d = GaussHermiteWeights(5, 2)
        @test all(ghw_test_2d.weights .> 0)
        @test all(isfinite.(ghw_test_2d.weights))
    end
    
    @testset "MonteCarloWeights" begin
        # Test basic construction
        mcw = MonteCarloWeights(100, 3)
        @test mcw isa AbstractQuadratureWeights
        @test size(mcw.points, 1) == 3  # Dimension
        @test size(mcw.points, 2) == 100  # Number of points
        @test length(mcw.weights) == 100
        
        # Test uniform weights
        @test all(mcw.weights .≈ 1/100)
        
        # Test different dimensions
        mcw_1d = MonteCarloWeights(50, 1)
        @test size(mcw_1d.points, 1) == 1
        @test size(mcw_1d.points, 2) == 50
        @test all(mcw_1d.weights .≈ 1/50)
        
        mcw_5d = MonteCarloWeights(200, 5)
        @test size(mcw_5d.points, 1) == 5
        @test size(mcw_5d.points, 2) == 200
        @test all(mcw_5d.weights .≈ 1/200)
        
        # Test that points are different (random)
        mcw1 = MonteCarloWeights(10, 2)
        mcw2 = MonteCarloWeights(10, 2)
        @test mcw1.points != mcw2.points  # Should be different due to randomness
    end
    
    @testset "LatinHypercubeWeights" begin
        # Test basic construction
        lhw = LatinHypercubeWeights(20, 2)
        @test lhw isa AbstractQuadratureWeights
        @test size(lhw.points, 1) == 2  # Dimension
        @test size(lhw.points, 2) == 20  # Number of points
        @test length(lhw.weights) == 20
        
        # Test uniform weights
        @test all(lhw.weights .≈ 1/20)
        
        # Test 1D case
        lhw_1d = LatinHypercubeWeights(15, 1)
        @test size(lhw_1d.points, 1) == 1
        @test size(lhw_1d.points, 2) == 15
        @test all(lhw_1d.weights .≈ 1/15)
        
        # Test higher dimensions
        lhw_4d = LatinHypercubeWeights(25, 4)
        @test size(lhw_4d.points, 1) == 4
        @test size(lhw_4d.points, 2) == 25
        @test all(lhw_4d.weights .≈ 1/25)
        
        # Test that points are distributed (Latin hypercube should give good space-filling)
        lhw_test = LatinHypercubeWeights(100, 2)
        points = lhw_test.points
        
        # Basic sanity checks - points should span a reasonable range
        @test minimum(points[1, :]) < -1.0
        @test maximum(points[1, :]) > 1.0
        @test minimum(points[2, :]) < -1.0
        @test maximum(points[2, :]) > 1.0
    end
    
    @testset "Helper Functions" begin
        # Test gausshermite_weights function directly
        points_2d, weights_2d = TransportMaps.gausshermite_weights(3, 2)
        @test size(points_2d, 1) == 2
        @test size(points_2d, 2) == 9  # 3^2
        @test length(weights_2d) == 9
        @test sum(weights_2d) ≈ 1.0 atol=1e-10  # For normalized weights
        
        # Test montecarlo_weights function directly
        points_mc, weights_mc = TransportMaps.montecarlo_weights(50, 3)
        @test size(points_mc, 1) == 3
        @test size(points_mc, 2) == 50
        @test length(weights_mc) == 50
        @test all(weights_mc .≈ 1/50)
        
        # Test latinhypercube_weights function directly
        points_lh, weights_lh = TransportMaps.latinhypercube_weights(30, 2)
        @test size(points_lh, 1) == 2
        @test size(points_lh, 2) == 30
        @test length(weights_lh) == 30
        @test all(weights_lh .≈ 1/30)
    end
    
    @testset "Integration Accuracy" begin
        # Test Gauss-Hermite quadrature accuracy with a simple Gaussian integral
        # ∫ exp(-x²) dx over ℝ = √π, but we test over a finite domain
        ghw = GaussHermiteWeights(10, 1)
        
        # Test integration of constant function (should equal sum of weights)
        f_const = x -> 1.0
        integral = sum(f_const(ghw.points[i]) * ghw.weights[i] for i in 1:length(ghw.weights))
        @test integral ≈ 1.0 atol=1e-10  # Sum of normalized weights
        
        # Test Monte Carlo integration convergence
        function mc_integrate_gaussian_2d(n_points)
            mcw = MonteCarloWeights(n_points, 2)
            # Integrate Gaussian exp(-x²-y²) over ℝ² using samples from N(0,1)
            # This should approximate ∫∫ exp(-x²-y²) * exp(x²+y²) dx dy = ∫∫ 1 dx dy = area
            integrand_values = [exp(-sum(mcw.points[:, i].^2)) for i in 1:n_points]
            return sum(integrand_values .* mcw.weights)
        end
        
        # This is just a basic functionality test - values should be reasonable
        result_100 = mc_integrate_gaussian_2d(100)
        result_1000 = mc_integrate_gaussian_2d(1000)
        
        @test 0.1 < result_100 < 2.0
        @test 0.1 < result_1000 < 2.0
    end
    
    @testset "Type Consistency" begin
        # Test that all quadrature methods return consistent types
        for QType in [GaussHermiteWeights, MonteCarloWeights, LatinHypercubeWeights]
            q = QType(10, 2)
            @test q.points isa Matrix{Float64}
            @test q.weights isa Vector{Float64}
            @test all(isfinite.(q.points))
            @test all(isfinite.(q.weights))
            @test all(q.weights .> 0)  # All weights should be positive
        end
    end
end