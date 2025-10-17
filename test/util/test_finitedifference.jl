using TransportMaps
using Test

@testset "Finite Difference Utilities" begin
    f = x -> x[1]^2 + 3.0 * x[2]
    x = [1.5, -0.5]
    g = TransportMaps.central_difference_gradient(f, x, 1e-8)
    expected = [2.0 * x[1], 3.0]
    @test isapprox(g, expected; atol=1e-5)
end
