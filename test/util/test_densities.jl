using Test
using TransportMaps

@testset "Test Densities" begin
    @testset "capybara_density" begin
        # Test basic functionality
        @test capybara_density([0.0, 0.0]) isa Float64
        @test capybara_density([0.0, 0.0]) > 0.0
        
        # Test it works with different points
        @test capybara_density([1.0, 0.0]) > 0.0
        @test capybara_density([-1.0, 0.5]) > 0.0
        
        # Test it's roughly normalized (should be close to 1 when integrated)
        # We'll just test a few sample points to ensure reasonable magnitude
        density_values = [capybara_density([x, y]) for x in -2:0.5:2, y in -2:0.5:2]
        @test all(d >= 0 for d in density_values)  # All non-negative
        @test maximum(density_values) < 1.0  # Reasonable maximum
        
        # Test dimension validation
        @test_throws ArgumentError capybara_density([1.0])  # 1D should fail
        @test_throws ArgumentError capybara_density([1.0, 2.0, 3.0])  # 3D should fail
        
        # Test with TargetDensity
        target = TargetDensity(capybara_density, :auto_diff)
        @test target isa TargetDensity
        @test gradient(target, [0.0, 0.0]) isa Vector{Float64}
        @test length(gradient(target, [0.0, 0.0])) == 2
    end
    
    @testset "irregular_density" begin
        # Test basic functionality
        @test irregular_density([0.0, 0.0]) isa Float64
        @test irregular_density([0.0, 0.0]) > 0.0
        
        # Test it works with different points
        @test irregular_density([1.0, 0.0]) > 0.0
        @test irregular_density([-1.0, 0.5]) > 0.0
        
        # Test dimension validation
        @test_throws ArgumentError irregular_density([1.0])  # 1D should fail
        @test_throws ArgumentError irregular_density([1.0, 2.0, 3.0])  # 3D should fail
        
        # Test with TargetDensity
        target = TargetDensity(irregular_density, :auto_diff)
        @test target isa TargetDensity
        @test gradient(target, [0.0, 0.0]) isa Vector{Float64}
        @test length(gradient(target, [0.0, 0.0])) == 2
    end
end