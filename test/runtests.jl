using TransportMaps
using Test

@testset "TransportMaps.jl" begin
    # Test Map Components
    @testset "Map Components" begin
        include("mapcomponents/test_hermitebasis.jl")
        include("mapcomponents/test_rectifier.jl")
        include("mapcomponents/test_polynomialmapcomponent.jl")
    end

    # Test Triangular Maps
    @testset "Triangular Maps" begin
        include("triangularmap/test_polynomialmap.jl")
        include("triangularmap/test_optimization.jl")
        include("triangularmap/test_gradients.jl")
        include("triangularmap/test_multithreading.jl")
    end

    # Test Utilities
    @testset "Utilities" begin
        include("util/test_gaussquadrature.jl")
        include("util/test_hybridrootfinder.jl")
        include("util/test_quadraturepoints.jl")
    end
end
