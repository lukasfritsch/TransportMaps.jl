"""
    OptimizationHistory

A data structure to store the iteration history of adaptive transport map optimization.

# Fields
- `terms::Vector{Matrix{Int64}}`: Multi-index sets at each iteration
- `train_objectives::Vector{Float64}`: Normalized training objectives at each iteration
- `test_objectives::Vector{Float64}`: Normalized test objectives at each iteration (empty if no test split)
- `gradients::Vector{Vector{Float64}}`: Gradient values for candidate terms at each iteration
"""
struct OptimizationHistory
    terms::Vector{Matrix{Int64}}
    train_objectives::Vector{Float64}
    test_objectives::Vector{Float64}
    gradients::Vector{Vector{Float64}}
    optimization_results::Vector{Optim.MultivariateOptimizationResults}

    function OptimizationHistory(maxiterations::Int)
        terms = [Matrix{Int64}(undef, 0, 0) for _ in 1:maxiterations]
        train_objectives = Vector{Float64}(undef, maxiterations)
        test_objectives = Vector{Float64}(undef, maxiterations)
        gradients = [Float64[] for _ in 1:maxiterations]
        optimization_results = Vector{Optim.MultivariateOptimizationResults}(undef, maxiterations)
        return new(terms, train_objectives, test_objectives, gradients, optimization_results)
    end
end

function update_optimization_history!(
    history::OptimizationHistory,
    term::Vector{Vector{Int64}},
    train_objective::Float64,
    test_objective::Float64,
    gradient::Vector{Float64},
    optimization_result::Optim.MultivariateOptimizationResults,
    iteration::Int
)
    history.terms[iteration] = permutedims(hcat(term...))
    history.train_objectives[iteration] = train_objective
    history.test_objectives[iteration] = test_objective
    history.gradients[iteration] = gradient
    history.optimization_results[iteration] = optimization_result
end

function Base.show(io::IO, history::OptimizationHistory)
    n_iterations = length(history.train_objectives)
    println(io, "OptimizationHistory with $n_iterations iterations:")
    for i in 1:n_iterations
        println(io, " Iteration $i:")
        println(io, "  Terms: ", history.terms[i])
        println(io, "  Train Objective: ", history.train_objectives[i])
        println(io, "  Test Objective: ", history.test_objectives[i])
    end
end

"""
    OptimizationResult

A data structure to store the optimization results for each component of a transport map.
"""
struct OptimizationResult
    train_objectives::Vector{Float64}
    test_objectives::Vector{Float64}
    optimization_results::Vector{Optim.MultivariateOptimizationResults}

    function OptimizationResult(dim::Int)
        train_objectives = Vector{Float64}(undef, dim)
        test_objectives = Vector{Float64}(undef, dim)
        optimization_results = Vector{Optim.MultivariateOptimizationResults}(undef, dim)
        return new(train_objectives, test_objectives, optimization_results)
    end
end

function update_optimization_result!(
    result::OptimizationResult,
    component_index::Int,
    train_objective::Float64,
    test_objective::Float64,
    optimization_result::Optim.MultivariateOptimizationResults
)
    result.train_objectives[component_index] = train_objective
    result.test_objectives[component_index] = test_objective
    result.optimization_results[component_index] = optimization_result
end

function Base.show(io::IO, result::OptimizationResult)
    dim = length(result.train_objectives)
    println(io, "OptimizationResult for $dim components:")
    for i in 1:dim
        println(io, " Component $i:")
        println(io, "  Train : ", result.train_objectives[i])
        println(io, "  Test  : ", result.test_objectives[i])
    end
end


struct MapOptimizationResult
    maps::Vector{PolynomialMap}
    train_objectives::Vector{Float64}
    test_objectives::Vector{Float64}
    gradients::Vector{Vector{Float64}}
    optimization_results::Vector{Optim.MultivariateOptimizationResults}

    function MapOptimizationResult(maxiterations::Int)
        maps = Vector{PolynomialMap}(undef, maxiterations)
        train_objectives = Vector{Float64}(undef, maxiterations)
        test_objectives = Vector{Float64}(undef, maxiterations)
        gradients = [Float64[] for _ in 1:maxiterations]
        optimization_results = Vector{Optim.MultivariateOptimizationResults}(undef, maxiterations)
        return new(maps, train_objectives, test_objectives, gradients, optimization_results)
    end
end

function update_optimization_history!(
    result::MapOptimizationResult,
    map::PolynomialMap,
    train_objective::Float64,
    test_objective::Float64,
    gradient::Vector{Float64},
    optimization_result::Optim.MultivariateOptimizationResults,
    iteration::Int
)
    result.maps[iteration] = map
    result.train_objectives[iteration] = train_objective
    result.test_objectives[iteration] = test_objective
    result.gradients[iteration] = gradient
    result.optimization_results[iteration] = optimization_result
end
