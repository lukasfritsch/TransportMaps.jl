@testsnippet OptimizationHistorySetup begin
    using TransportMaps
    using Test
    using Optim

end

@testitem "OptimizationHistory" setup = [OptimizationHistorySetup] begin
    # Test constructor
    max_iterations = 5
    history = OptimizationHistory(max_iterations)

    @test length(history.train_objectives) == max_iterations
    @test length(history.test_objectives) == max_iterations
    @test length(history.gradients) == max_iterations
    @test length(history.optimization_results) == max_iterations

    # Test update_optimization_history!
    term = [[0, 0], [1, 0], [0, 1]]
    train_obj = 0.5
    test_obj = 0.6
    gradient = [0.1, 0.2, 0.3]

    # Create a dummy optimization result
    result = optimize(x -> sum(x .^ 2), [1.0, 2.0], LBFGS())

    TransportMaps.update_optimization_history!(history, term, train_obj, test_obj, gradient, result, 1)

    @test history.terms[1] == permutedims(hcat(term...))
    @test history.train_objectives[1] == train_obj
    @test history.test_objectives[1] == test_obj
    @test history.gradients[1] == gradient
    @test history.optimization_results[1] == result

    @test_nowarn show(history)
end

@testitem "OptimizationResult" setup = [OptimizationHistorySetup] begin
    dim = 3
    result = OptimizationResult(dim)

    @test length(result.train_objectives) == dim
    @test length(result.test_objectives) == dim
    @test length(result.optimization_results) == dim

    @test_nowarn show(result)
end

@testitem "MapOptimizationResult" setup = [OptimizationHistorySetup] begin
    maxiter = 5
    result = MapOptimizationResult(maxiter)

    @test length(result.maps) == maxiter
    @test length(result.train_objectives) == maxiter
    @test length(result.test_objectives) == maxiter
    @test length(result.gradients) == maxiter
    @test length(result.optimization_results) == maxiter

    @test_nowarn show(result)
end
