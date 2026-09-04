@testsnippet AdaptiveTransportMapDensitySetup begin
    using TransportMaps
    using Test
    using Distributions
    using Optim
    using Random
    using Logging

end

@testitem "Adaptive Transport Map from density" setup = [AdaptiveTransportMapDensitySetup] begin

    # Define target and quadrature
    logtarget(x) = logpdf(Normal(), x[1]) + logpdf(Normal(), x[2] - x[1]^2)
    target = MapTargetDensity(logtarget)
    quadrature = GaussHermiteWeights(2, 2)
    maxterms = 5

    @testset "No Validation" begin
        T, hist = @test_logs (:info, "Selected best adaptive map") match_mode = :any optimize_adaptive_transportmap(
            target, quadrature, maxterms,
        )

        @test numbercoefficients(T) <= maxterms
        @test isnan(hist.test_objectives[1])
    end

    @testset "Validation" begin
        logger = Test.TestLogger()
        T, hist = Logging.with_logger(logger) do
            optimize_adaptive_transportmap(
                target, quadrature, maxterms;
                validation = LatinHypercubeWeights(10, 2),
            )
        end

        selection_logs = filter(
            log -> log.level == Logging.Info && log.message == "Selected best adaptive map",
            logger.logs,
        )
        @test length(selection_logs) == 1
        @test haskey(selection_logs[1].kwargs, :validation_objective)

        @test numbercoefficients(T) <= maxterms
        @test !iszero(hist.test_objectives[1])
    end

    @testset "Warm start" begin
        T_init = DiagonalMap(2, 1)
        maxterms_exact = numbercoefficients(T_init)

        T, hist = optimize_adaptive_transportmap(
            target, quadrature, maxterms_exact;
            initial_map = T_init
        )
        @test numbercoefficients(T) == maxterms_exact
        @test T.forwarddirection == :target

        T_init_reference = DiagonalMap(2, 1)
        rng = MersenneTwister(123)
        TransportMaps.initializemapfromsamples!(T_init_reference, randn(rng, 10, 2))
        @test T_init_reference.forwarddirection == :reference
        T_reference, _ = optimize_adaptive_transportmap(
            target, quadrature, maxterms;
            initial_map = T_init_reference
        )
        @test T_reference.forwarddirection == :target

        @test_throws AssertionError optimize_adaptive_transportmap(
            target, quadrature, maxterms; initial_map = PolynomialMap(2, 2)
        )
    end

    @testset "Options" begin
        rectifier = ShiftedELU()
        basis = HermiteBasis()
        optimizer = BFGS()
        options = Optim.Options(iterations = 10)

        T, hist = optimize_adaptive_transportmap(
            target, quadrature, maxterms;
            rectifier = rectifier, basis = basis, optimizer = optimizer, options = options
        )

        @test basistype(T[1].basisfunctions[1]) == typeof(basis)
        @test T[1].rectifier == rectifier
        @test hist.optimization_results[1].method == optimizer

        max_iterations = maximum([res.iterations for res in hist.optimization_results])
        @test max_iterations <= 10
    end
end
