using TransportMaps
using Test

@testset "Smolyak Utilities" begin
    # Test nested levels and a small smolyak grid
    levels = TransportMaps.nested_hermite_levels_quads()
    @test haskey(levels, 0)

    pts, w = TransportMaps.hermite_smolyak_points(2, 1)
    @test size(pts, 2) == 2
    @test length(w) == size(pts, 1)

    # Check create_tensor_product combination
    nodes_sets = [[0.0, 1.0], [0.0, -1.0]]
    weights_sets = [[1.0, 1.0], [0.5, 0.5]]
    nodes, weights = TransportMaps.create_tensor_product(nodes_sets, weights_sets)
    @test length(nodes) == 4
    @test length(weights) == 4
end
