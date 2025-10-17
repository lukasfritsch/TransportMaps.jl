using TransportMaps
using Test
using Distributions

@testset "Map Density Utilities" begin
    # Analytical gradient constructor
    target = TransportMaps.MapTargetDensity(x -> pdf(Normal(), x[1]), :auto_diff)
    x = [0.0]
    g = TransportMaps.gradient(target, x)
    @test length(g) == 1
    @test isfinite(g[1])

    # Reference density constructor
    ref = TransportMaps.MapReferenceDensity(Normal(0,1))
    gref = TransportMaps.gradient(ref, [0.0, 0.0])
    @test length(gref) == 2
    @test all(isfinite.(gref))
end
