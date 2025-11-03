using TransportMaps
using Test

@testset "Finite Differences" begin
    f = x -> x[1] .^ 3 * x[2]
    x = [1.5, -0.5]

    @testset "Finite Difference Gradient" begin
        g = TransportMaps.central_difference_gradient(f, x)
        expected = [3 * x[1]^2 * x[2], x[1]^3]
        @test isapprox(g, expected; atol=1e-5)
    end

    @testset "Finite Difference Hessian" begin
        H = TransportMaps.central_difference_hessian(f, x)
        expected = [6.0*x[1]*x[2] 3*x[1]^2; 3*x[1]^2 0.0]
        @test isapprox(H, expected; atol=1e-5)
    end

end
