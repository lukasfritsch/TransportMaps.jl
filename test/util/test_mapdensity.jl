using TransportMaps
using Test
using Distributions

@testset "Map Density" begin

    @testset "MapTargetDensity" begin
        # Analytical gradient constructor
        target = MapTargetDensity(x -> logpdf(Normal(), x[1]), :auto_diff)
        x = [0.0]
        g = grad_logpdf(target, x)
        @test length(g) == 1
        @test isfinite(g[1])

        # Test logpdf and pdf methods
        @test logpdf(target, [0.0]) ≈ logpdf(Normal(), 0.0)
        @test logpdf(target, 0.0) ≈ logpdf(Normal(), 0.0)
        @test pdf(target, [0.0]) ≈ pdf(Normal(), 0.0)
        @test pdf(target, 0.0) ≈ pdf(Normal(), 0.0)

        # Test logpdf with matrix input
        X = permutedims([-1.0 0.0 1.0])
        logpdfs = logpdf(target, X)
        @test length(logpdfs) == 3
        @test logpdfs[1] ≈ logpdf(Normal(), -1.0)
        @test logpdfs[2] ≈ logpdf(Normal(), 0.0)
        @test logpdfs[3] ≈ logpdf(Normal(), 1.0)

        # Test pdf with matrix input
        pdfs = pdf(target, X)
        @test length(pdfs) == 3
        @test pdfs[1] ≈ pdf(Normal(), -1.0)
        @test pdfs[2] ≈ pdf(Normal(), 0.0)
        @test pdfs[3] ≈ pdf(Normal(), 1.0)

        @test_throws ArgumentError MapTargetDensity(x -> pdf(Normal(), x[1]), :finite_difference, x -> zeros(length(x)))
        @test_throws ArgumentError MapTargetDensity(x -> pdf(Normal(), x[1]), :analytical)

        target_fd = MapTargetDensity(x -> logpdf(Normal(), x[1]), :finite_difference)
        g_fd = grad_logpdf(target_fd, x)
        @test length(g_fd) == 1
        @test isfinite(g_fd[1])

        target_analytical = MapTargetDensity(x -> logpdf(Normal(), x[1]), x -> [-x[1] * pdf(Normal(), x[1])])
        g_analytical = grad_logpdf(target_analytical, x)
        @test length(g_analytical) == 1
        @test isfinite(g_analytical[1])

        target_analytical = MapTargetDensity(x -> logpdf(Normal(), x[1]), :analytical, x -> [-x[1] * pdf(Normal(), x[1])])
        g_analytical = grad_logpdf(target_analytical, x)
        @test length(g_analytical) == 1
        @test isfinite(g_analytical[1])

        @testset "Gradient Types" begin
            # Test different gradient types
            gradient_types_ad = [:auto_diff, :autodiff, :ad, :automatic, :forward_diff, :forwarddiff]

            for type in gradient_types_ad
                t = MapTargetDensity(x -> logpdf(Normal(), x[1]), type)
                @test t.gradient_type == :auto_diff
            end

            gradient_types_fd = [:finite_difference, :finitedifference, :finite_diff, :finitediff, :fd, :numerical, :numeric]

            for type in gradient_types_fd
                t = MapTargetDensity(x -> logpdf(Normal(), x[1]), type)
                @test t.gradient_type == :finite_difference
            end

        end

    end

    @testset "MapReferenceDensity" begin
        # Auto-diff gradient constructor
        ref = MapReferenceDensity(Normal())
        x = [0.0]
        g = grad_logpdf(ref, x)
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
