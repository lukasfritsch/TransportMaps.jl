# Adaptive transport map optimization based on greedy basis selection
# as described in:
#   Baptista, R., Marzouk, Y., & Zahm, O. (2023). On the Representation and Learning of
#   Monotone Triangular Transport Maps. Foundations of Computational Mathematics.
#   https://doi.org/10.1007/s10208-023-09630-x

"""
    AdaptiveTransportMap(samples, maxterms; kwargs...)

Adaptively optimize a triangular transport map by greedily enriching the multi-index set.

# Arguments
- `samples::Matrix{Float64}`: Sample data where rows are samples and columns are dimensions
- `maxterms::Vector{Int64}`: Maximum number of terms for each component

# Keyword Arguments
- `lm::LinearMap=LinearMap()`: Linear map to standardize samples
- `rectifier::AbstractRectifierFunction=Softplus()`: Rectifier function to use
- `basis::AbstractPolynomialBasis=LinearizedHermiteBasis()`: Polynomial basis
- `optimizer::Optim.AbstractOptimizer=LBFGS()`: Optimization algorithm
- `options::Optim.Options=Optim.Options()`: Optimizer options
- `test_fraction::Float64=0.0`: Fraction of samples to use for testing (validation)

# Returns
- `M::PolynomialMap`: The optimized triangular transport map
- `iteration_histories::Vector{OptimizationHistory}`: History of optimization for each component
"""
function AdaptiveTransportMap(
    samples::Matrix{Float64},
    maxterms::Vector{Int64},
    lm::LinearMap=LinearMap(),
    rectifier::AbstractRectifierFunction=Softplus(),
    basis::AbstractPolynomialBasis=LinearizedHermiteBasis();
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options(),
    test_fraction::Float64=0.0
)
    @assert length(maxterms) == size(samples, 2) "Length of maxterms must equal number of dimensions in samples"
    # Extract number of dimensions from samples
    d = size(samples, 2)
    T = typeof(basis)
    map_components = Vector{PolynomialMapComponent{T}}()
    iteration_histories = Vector{OptimizationHistory}()

    # Standardize samples using linear map
    samples = evaluate(lm, samples)

    # Prepare train/test split
    train_samples, test_samples = _test_train_split(samples, test_fraction)

    for k in 1:d
        println("Start optimizing component $k:")
        component, history = adaptive_optimization(train_samples[:, 1:k], test_samples[:, 1:k], maxterms[k], rectifier, basis, optimizer, options)
        push!(map_components, component)
        push!(iteration_histories, history)
    end

    # Construct final map from optimized components
    M = PolynomialMap(map_components; forwarddirection=:reference)

    return M, iteration_histories
end

"""
    AdaptiveTransportMap(samples, maxterms, k_folds; kwargs...)

Adaptively optimize a triangular transport map using k-fold cross-validation to
select the number of terms per component.

# Arguments
- `samples::Matrix{Float64}`: Sample data where rows are samples and columns are dimensions
- `maxterms::Vector{Int64}`: Upper bound on the number of terms to consider for each component
- `k_folds::Int`: Number of folds to use for cross-validation

# Keyword Arguments
- `lm::LinearMap=LinearMap()`: Linear map to standardize samples
- `rectifier::AbstractRectifierFunction=Softplus()`: Rectifier function to use
- `basis::AbstractPolynomialBasis=LinearizedHermiteBasis()`: Polynomial basis
- `optimizer::Optim.AbstractOptimizer=LBFGS()`: Optimization algorithm
- `options::Optim.Options=Optim.Options()`: Optimizer options

# Returns
- `M::PolynomialMap`: The optimized triangular transport map trained on all samples
- `fold_histories::Vector{Vector{OptimizationHistory}}`: Optimization history for each component and fold
- `selected_terms::Vector{Int}`: Number of terms selected for each component
"""
function AdaptiveTransportMap(
    samples::Matrix{Float64},
    maxterms::Vector{Int64},
    k_folds::Int;
    lm::LinearMap=LinearMap(),
    rectifier::AbstractRectifierFunction=Softplus(),
    basis::AbstractPolynomialBasis=LinearizedHermiteBasis(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options()
)
    @assert length(maxterms) == size(samples, 2) "Length of maxterms must equal number of dimensions in samples"
    @assert k_folds >= 2 "k_folds must be at least 2"
    @assert k_folds <= size(samples, 1) "k_folds cannot exceed the number of samples"

    d = size(samples, 2)
    T = typeof(basis)
    map_components = Vector{PolynomialMapComponent{T}}(undef, d)
    fold_histories = Vector{Vector{OptimizationHistory}}(undef, d)
    selected_terms = Vector{Int}(undef, d)

    samples = evaluate(lm, samples)
    folds = _kfold_indices(size(samples, 1), k_folds)

    for k in 1:d
        println("Start k-fold optimization of component $k with $k_folds folds")
        component_histories = Vector{OptimizationHistory}(undef, k_folds)

        for fold_id in 1:k_folds
            test_idx = folds[fold_id]
            train_idx = [idx for (j, idxs) in enumerate(folds) if j != fold_id for idx in idxs]

            train_samples_fold = samples[train_idx, 1:k]
            test_samples_fold = samples[test_idx, 1:k]

            println("   * Fold $fold_id / $k_folds")
            _, history_fold = adaptive_optimization(
                train_samples_fold,
                test_samples_fold,
                maxterms[k],
                rectifier,
                basis,
                optimizer,
                options,
            )

            component_histories[fold_id] = history_fold
        end

        mean_test_objectives = [Statistics.mean(history.test_objectives[t] for history in component_histories) for t in 1:maxterms[k]]
        best_iteration = argmin(mean_test_objectives)
        selected_terms[k] = best_iteration

        fold_scores = [component_histories[i].test_objectives[best_iteration] for i in 1:k_folds]

        best_fold = argmin(fold_scores)
        Λ_best = deepcopy(component_histories[best_fold].terms[best_iteration])

        println("   Best fold: $best_fold")
        println("   Selected $(selected_terms[k]) terms (avg validation objective $(mean_test_objectives[best_iteration]))")

        component = PolynomialMapComponent(Λ_best, rectifier, basis, samples[:, 1:k])
        res = optimize!(component, samples[:, 1:k], optimizer, options)

        train_obj = objective(component, samples[:, 1:k]) / size(samples, 1)
        println("   Full-data objective: $train_obj")
        println("   Final optimizer status: $(Optim.converged(res) ? "Converged" : "Not Converged") with $(Optim.iterations(res)) iterations")

        map_components[k] = component
        fold_histories[k] = component_histories
    end

    M = PolynomialMap(map_components; forwarddirection=:reference)

    return M, fold_histories, selected_terms
end

"""
    adaptive_optimization(train_samples, test_samples, maxterms, rectifier, basis, optimizer, options)

Adaptively optimize a single transport map component by greedily enriching the multi-index set.

# Arguments
- `train_samples::Matrix{Float64}`: Training sample data
- `test_samples::Matrix{Float64}`: Test/validation sample data (can be empty)
- `maxterms::Int`: Maximum number of terms to add
- `rectifier::AbstractRectifierFunction`: Rectifier function to use
- `basis::AbstractPolynomialBasis`: Polynomial basis
- `optimizer::Optim.AbstractOptimizer`: Optimization algorithm
- `options::Optim.Options`: Optimizer options

# Returns
- `component::PolynomialMapComponent`: The optimized map component
- `history::OptimizationHistory`: Optimization history for this component
"""
function adaptive_optimization(
    train_samples::Matrix{Float64},
    test_samples::Matrix{Float64},
    maxterms::Int,
    rectifier::AbstractRectifierFunction,
    basis::AbstractPolynomialBasis,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options
)

    d = size(train_samples, 2)
    # Initialize multi-index set to contain only zero index (constant term)
    Λ = multivariate_indices(0, d)

    # Initialize history buffer
    history = OptimizationHistory(maxterms)

    # Initialize component with the multi-index set
    println("   * Optimizing term 1 / $maxterms")
    component = PolynomialMapComponent(Λ, rectifier, basis, train_samples)
    res = optimize!(component, train_samples, optimizer, options)

    # Compute and store first iteration
    train_obj = objective(component, train_samples) / size(train_samples, 1)
    test_obj = !isempty(test_samples) ? objective(component, test_samples) / size(test_samples, 1) : 0.
    update_optimization_history!(history, Λ, train_obj, test_obj, Float64[], res, 1)

    # start greedy optimization
    for t in 2:maxterms
        # Candidates: terms in the reduced margin of Λₜ
        Λ_rm = reduced_margin(Λ)

        # store objective of all candidates and gradients
        gradients = zeros(Float64, length(Λ_rm))
        coeffs = getcoefficients(component)

        # Loop over all candidates in the reduced margin
        for (i, α) in enumerate(Λ_rm)
            Λ_cand = copy(Λ)
            push!(Λ_cand, α)

            # Update component with new multi-index set
            component_cand = PolynomialMapComponent(Λ_cand, rectifier, basis, train_samples)
            # Copy the optimized component coefficients; set other to 0
            setcoefficients!(component_cand, [coeffs..., 0.0]) # set coefficients for existing terms

            # evaluate objective and gradient of objective function
            gradients[i] = abs(objective_gradient!(component_cand, train_samples)[end])
        end

        # select best candidate based on maximum gradient
        α⁺ = Λ_rm[argmax(gradients)]
        push!(Λ, α⁺)

        # Update component with new multi-index set
        println("   * Adding term $t / $maxterms")
        component = PolynomialMapComponent(Λ, rectifier, basis, train_samples)
        setcoefficients!(component, [coeffs..., 0.0]) # set coefficients for existing terms
        res = optimize!(component, train_samples, optimizer, options)

        # Compute and store iteration
        train_obj = objective(component, train_samples) / size(train_samples, 1)
        test_obj = !isempty(test_samples) ? objective(component, test_samples) / size(test_samples, 1) : 0.
        update_optimization_history!(history, Λ, train_obj, test_obj, gradients, res, t)
    end

    return component, history
end

function _kfold_indices(n_samples::Int, k_folds::Int)
    idx = randperm(n_samples)
    fold_sizes = fill(div(n_samples, k_folds), k_folds)
    remainder = n_samples % k_folds
    for i in 1:remainder
        fold_sizes[i] += 1
    end

    folds = Vector{Vector{Int}}(undef, k_folds)
    pointer = 1
    for fold_id in 1:k_folds
        fold_length = fold_sizes[fold_id]
        stop_idx = pointer + fold_length - 1
        folds[fold_id] = collect(idx[pointer:stop_idx])
        pointer = stop_idx + 1
    end

    return folds
end
