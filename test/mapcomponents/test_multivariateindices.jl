using TransportMaps
using Test

@testset "Multivariate Indices" begin
    # total-order for p=1, k=2 should produce: [0,0], [1,0], [0,1]
    idxs = multivariate_indices(1, 2, mode=:total)
    @test length(idxs) == 3
    @test any(x -> x == [0,0], idxs)
    @test any(x -> x == [1,0], idxs)
    @test any(x -> x == [0,1], idxs)

    # diagonal mode p=2, k=3: indices should be (0,0,0),(0,0,1),(0,0,2)
    diag = multivariate_indices(2, 3, mode=:diagonal)
    @test length(diag) == 3
    @test diag[1] == [0,0,0]
    @test diag[2] == [0,0,1]
    @test diag[3] == [0,0,2]

    # no_mixed p=2, k=2: should include [0,0],[1,0],[2,0],[0,1],[0,2]
    nm = multivariate_indices(2, 2, mode=:no_mixed)
    @test any(x -> x == [0,0], nm)
    @test any(x -> x == [1,0], nm)
    @test any(x -> x == [2,0], nm)
    @test any(x -> x == [0,1], nm)
    @test any(x -> x == [0,2], nm)

    # p=0, k=4 should return only the zero index
    zero_idx = multivariate_indices(0, 4)
    @test length(zero_idx) == 1
    @test zero_idx[1] == [0,0,0,0]

    # Unknown mode should throw error
    @test_throws AssertionError multivariate_indices(2, 2, mode=:unknown)
end

@testset "Reduced Margin" begin
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

    # example: Λ contains (0,0),(1,0),(0,1) in 2D -> reduced margin {(1,1),(2,0),(0,2)}
    Λ = [ [0,0], [1,0], [0,1] ]
    rm = reduced_margin(Λ)
    ex = [ [2,0], [0,2], [1,1] ]
    @test all(x -> any(y -> y == x, rm), ex)

    # 3D example
    Λ = [ [0,0,0], [1,0,0], [0,1,0], [0,0,1] ]
    rm = reduced_margin(Λ)
    ex = [ [1,1,0], [1,0,1], [0,1,1], [2,0,0], [0,2,0], [0,0,2] ]
    @test all(x -> any(y -> y == x, rm), ex)
end
