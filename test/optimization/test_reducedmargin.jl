using Test
using TransportMaps

@testset "reduced_margin" begin
    # empty set -> empty reduced margin
    Λ = Vector{Vector{Int}}()
    rm = reduced_margin(Λ)
    @test isempty(rm)

    # single zero index in 2D -> reduced margin should be {(1,0),(0,1)}
    Λ = [ [0,0] ]
    rm = reduced_margin(Λ)
    expected = [ [1,0], [0,1] ]
    @test length(rm) == 2
    @test all(x -> any(y -> y == x, rm), expected)

    # example: Λ contains (0,0),(1,0),(0,1) in 2D -> reduced margin {(1,1),(2,0),(0,2)}?
    Λ = [ [0,0], [1,0], [0,1] ]
    rm = reduced_margin(Λ)
    # expected candidates are (2,0),(0,2),(1,1)
    ex = [ [2,0], [0,2], [1,1] ]
    @test all(x -> any(y -> y == x, rm), ex)

    # larger example: total-order multi-indices up to degree 1 in 3D -> Λ = {000,100,010,001}
    Λ = [ [0,0,0], [1,0,0], [0,1,0], [0,0,1] ]
    rm = reduced_margin(Λ)
    ex = [ [1,1,0], [1,0,1], [0,1,1], [2,0,0], [0,2,0], [0,0,2] ]
    @test all(x -> any(y -> y == x, rm), ex)
end
