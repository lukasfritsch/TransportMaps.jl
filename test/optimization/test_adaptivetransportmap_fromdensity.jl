using TransportMaps
using Test
using Distributions
using Optim

@testset "Adaptive Transport Map from density" begin

    # Define target and quadrature
    logtarget(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
    target = MapTargetDensity(logtarget, :ad)
    quadrature = GaussHermiteWeights(2, 2)
    maxterms = 5

    @testset "No Validation" begin
        T, hist = optimize_adaptive_transportmap(target, quadrature, maxterms)

        @test numbercoefficients(T) <= maxterms
        @test isnan(hist.test_objectives[1])
    end

    @testset "Validation" begin
        T, hist = optimize_adaptive_transportmap(target, quadrature, maxterms;
            validation=LatinHypercubeWeights(10, 2))

        @test numbercoefficients(T) <= maxterms
        @test !iszero(hist.test_objectives[1])
    end

    @testset "Options" begin
        rectifier = ShiftedELU()
        basis = HermiteBasis()
        optimizer = BFGS()
        options = Optim.Options(iterations=10)

        T, hist = optimize_adaptive_transportmap(target, quadrature, maxterms;
            rectifier=rectifier, basis=basis, optimizer=optimizer, options=options)

        @test basistype(T[1].basisfunctions[1]) == typeof(basis)
        @test T[1].rectifier == rectifier
        @test hist.optimization_results[1].method == optimizer

        max_iterations = maximum([res.iterations for res in hist.optimization_results])
        @test max_iterations <= 10
    end
end
