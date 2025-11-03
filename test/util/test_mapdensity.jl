using TransportMaps
using Test
using Distributions

@testset "Map Density" begin

    @testset "MapTargetDensity" begin
        # Analytical gradient constructor
        target = MapTargetDensity(x -> pdf(Normal(), x[1]), :auto_diff)
        x = [0.0]
        g = gradient(target, x)
        @test length(g) == 1
        @test isfinite(g[1])

        @test_throws ArgumentError MapTargetDensity(x -> pdf(Normal(), x[1]), :finite_difference, x -> zeros(length(x)))
        @test_throws ArgumentError MapTargetDensity(x -> pdf(Normal(), x[1]), :analytical)

        target_fd = MapTargetDensity(x -> pdf(Normal(), x[1]), :finite_difference)
        g_fd = gradient(target_fd, x)
        @test length(g_fd) == 1
        @test isfinite(g_fd[1])

        target_analytical = MapTargetDensity(x -> pdf(Normal(), x[1]), x -> [-x[1] * pdf(Normal(), x[1])])
        g_analytical = gradient(target_analytical, x)
        @test length(g_analytical) == 1
        @test isfinite(g_analytical[1])

        target_analytical = MapTargetDensity(x -> pdf(Normal(), x[1]), :analytical, x -> [-x[1] * pdf(Normal(), x[1])])
        g_analytical = gradient(target_analytical, x)
        @test length(g_analytical) == 1
        @test isfinite(g_analytical[1])

        @testset "Gradient Types" begin
            # Test different gradient types
            gradient_types_ad = [:auto_diff, :autodiff, :ad, :automatic, :forward_diff, :forwarddiff]

            for type in gradient_types_ad
                t = MapTargetDensity(x -> pdf(Normal(), x[1]), type)
                @test t.gradient_type == :auto_diff
            end

            gradient_types_fd = [:finite_difference, :finitedifference, :finite_diff, :finitediff, :fd, :numerical, :numeric]

            for type in gradient_types_fd
                t = MapTargetDensity(x -> pdf(Normal(), x[1]), type)
                @test t.gradient_type == :finite_difference
            end

        end

    end

    @testset "MapReferenceDensity" begin
        # Auto-diff gradient constructor
        ref = MapReferenceDensity(Normal())
        x = [0.0]
        g = gradient(ref, x)
        @test length(g) == 1
        @test isfinite(g[1])

        @test_throws ErrorException MapReferenceDensity(Uniform())

    end

    @testset "Show Methods" begin
        target = MapTargetDensity(x -> pdf(Normal(), x[1]), :auto_diff)
        ref = MapReferenceDensity(Normal())

        @test_nowarn display(target)
        @test_nowarn display(ref)
    end
end
