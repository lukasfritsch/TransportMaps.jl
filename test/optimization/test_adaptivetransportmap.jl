using TransportMaps
using Test
using Random
using Distributions
using Optim
using Statistics

@testset "Adaptive Transport Map" begin
    # Set random seed for reproducibility
    Random.seed!(42)

    # Create simple test samples from a standard normal distribution
    n_samples = 100
    n_dims = 2
    samples = randn(n_samples, n_dims)

    @testset "Basic ATM Construction and Optimization" begin
        maxterms = [3, 3]  # 3 terms max for each component
        opts = Optim.Options(iterations=10)  # Few iterations for fast testing

        M, histories = AdaptiveTransportMap(
            samples,
            maxterms;
            optimizer=LBFGS(),
            options=opts
        )

        # Check that we get a PolynomialMap back
        @test M isa PolynomialMap

        # Check that we get histories for each component
        @test length(histories) == n_dims
        @test all(h isa OptimizationHistory for h in histories)

        # Check that each history has the right number of iterations
        @test all(length(h.train_objectives) == maxterms[i] for (i, h) in enumerate(histories))
    end

    @testset "History Tracking" begin
        maxterms = [2, 2]
        opts = Optim.Options(iterations=5)

        M, histories = AdaptiveTransportMap(
            samples,
            maxterms;
            optimizer=LBFGS(),
            options=opts
        )

        for (k, history) in enumerate(histories)
            # Check that train objectives are positive and decreasing (generally)
            @test all(obj > 0 for obj in history.train_objectives)

            # Check that terms are stored
            @test all(length(term) > 0 for term in history.terms)

            # Check that gradients are stored for candidate selection
            @test length(history.gradients) == maxterms[k]
        end
    end

    @testset "Train/Test Split" begin
        maxterms = [2, 2]
        test_fraction = 0.2
        opts = Optim.Options(iterations=5)

        M, histories = AdaptiveTransportMap(
            samples,
            maxterms;
            test_fraction=test_fraction,
            optimizer=LBFGS(),
            options=opts
        )

        for history in histories
            # When test_fraction > 0, test_objectives should not be empty
            # Note: test_objectives will have entries for each iteration
            @test length(history.test_objectives) > 0
        end
    end

    @testset "Different Basis Functions and Rectifiers" begin
        maxterms = [2]  # 1D only for faster testing
        samples_1d = samples[:, 1:1]
        opts = Optim.Options(iterations=5)

        # Test with LinearizedHermiteBasis (default)
        M1, h1 = AdaptiveTransportMap(
            samples_1d,
            maxterms,
            LinearMap(),
            Softplus(),
            LinearizedHermiteBasis(),
            optimizer=LBFGS(),
            options=opts
        )
        @test M1 isa PolynomialMap
        @test h1[1] isa OptimizationHistory

        # Test with HermiteBasis
        M2, h2 = AdaptiveTransportMap(
            samples_1d,
            maxterms,
            LinearMap(),
            ShiftedELU(),
            HermiteBasis(),
            optimizer=LBFGS(),
            options=opts
        )
        @test M2 isa PolynomialMap
        @test h2[1] isa OptimizationHistory
    end

    @testset "Map Evaluation" begin
        maxterms = [2, 2]
        opts = Optim.Options(iterations=5)

        M, histories = AdaptiveTransportMap(
            samples,
            maxterms;
            optimizer=LBFGS(),
            options=opts
        )

        # Evaluate the map on test samples
        test_sample = [0.0, 0.0]
        result = evaluate(M, test_sample)

        @test isa(result, Vector)
        @test length(result) == n_dims
    end

    @testset "K-Fold Cross Validation" begin
        maxterms = [2, 3]
        k_folds = 3
        opts = Optim.Options(iterations=5)

        M, fold_histories, selected_terms = AdaptiveTransportMap(
            samples,
            maxterms,
            k_folds;
            optimizer=LBFGS(),
            options=opts,
        )

        @test M isa PolynomialMap
        @test length(fold_histories) == n_dims
        @test selected_terms isa Vector{Int}
        @test length(selected_terms) == n_dims
        @test all(length(component_histories) == k_folds for component_histories in fold_histories)
        @test all(all(history isa OptimizationHistory for history in component_histories) for component_histories in fold_histories)
        @test all(selected_terms[i] <= maxterms[i] for i in 1:n_dims)
        @test all(selected_terms .>= 1)
        @test all(all(length(history.train_objectives) == maxterms[i] for history in fold_histories[i]) for i in 1:n_dims)

        mean_validation = [Statistics.mean(fold_histories[i][fold].test_objectives[selected_terms[i]] for fold in 1:k_folds) for i in 1:n_dims]
        @test all(isfinite, mean_validation)
    end

end
