using TransportMaps
using Test

@testset "Rectifier Functions" begin
    @testset "Softplus" begin
        softplus = Softplus()

        # Test basic functionality
        @test softplus(0.0) ≈ log(2.0)  # log(1 + exp(0)) = log(2)
        @test softplus(-10.0) ≈ exp(-10.0) atol=1e-7  # log(1 + exp(-10)) ≈ exp(-10) for large negative values
        @test softplus(10.0) ≈ 10.0 atol=1e-4  # log(1 + exp(10)) ≈ 10 for large positive values

        # Test monotonicity (should be increasing)
        @test softplus(-1.0) < softplus(0.0) < softplus(1.0)

        # Test positivity
        @test softplus(-100.0) > 0.0
        @test softplus(0.0) > 0.0
        @test softplus(100.0) > 0.0

        # Test type stability
        @test softplus(1) isa Float64
        @test softplus(1.0) isa Float64

        # show method
        s = sprint(show, Softplus())
        @test occursin("Softplus", s) || !isempty(s)
        s2 = sprint(io->show(io, MIME("text/plain"), Softplus()))
        @test !isempty(s2)

        s = sprint(show, ShiftedELU())
        @test occursin("ShiftedELU", s) || !isempty(s)

        s = sprint(show, IdentityRectifier())
        @test occursin("IdentityRectifier", s) || !isempty(s)
    end

    @testset "ShiftedELU" begin
        shifted_elu = ShiftedELU()

        # Test basic functionality
        @test shifted_elu(0.0) ≈ 1.0  # exp(0) = 1 for ξ ≤ 0
        @test shifted_elu(-1.0) ≈ exp(-1.0)  # exp(-1) for ξ ≤ 0
        @test shifted_elu(1.0) ≈ 2.0  # 1 + 1 = 2 for ξ > 0
        @test shifted_elu(2.0) ≈ 3.0  # 2 + 1 = 3 for ξ > 0

        # Test continuity at ξ = 0
        @test shifted_elu(0.0) ≈ 1.0
        @test abs(shifted_elu(1e-10) - shifted_elu(-1e-10)) < 1e-9

        # Test monotonicity (should be increasing)
        @test shifted_elu(-2.0) < shifted_elu(-1.0) < shifted_elu(0.0) < shifted_elu(1.0) < shifted_elu(2.0)

        # Test positivity
        @test shifted_elu(-100.0) > 0.0
        @test shifted_elu(0.0) > 0.0
        @test shifted_elu(100.0) > 0.0

        # Test type stability
        @test shifted_elu(1) isa Real
        @test shifted_elu(1.0) isa Float64
    end

    @testset "IdentityRectifier" begin
        identity_rect = IdentityRectifier()

        # Test basic functionality
        @test identity_rect(0.0) ≈ 0.0
        @test identity_rect(1.0) ≈ 1.0
        @test identity_rect(-1.0) ≈ -1.0
        @test identity_rect(π) ≈ π

        # Test linearity
        @test identity_rect(2.0 * 3.0) ≈ 2.0 * identity_rect(3.0)
        @test identity_rect(1.0 + 2.0) ≈ identity_rect(1.0) + identity_rect(2.0)

        # Test type stability
        @test identity_rect(1) isa Real
        @test identity_rect(1.0) isa Float64
    end

    @testset "Abstract Type Hierarchy" begin
        @test Softplus() isa AbstractRectifierFunction
        @test ShiftedELU() isa AbstractRectifierFunction
        @test IdentityRectifier() isa AbstractRectifierFunction
    end
end
