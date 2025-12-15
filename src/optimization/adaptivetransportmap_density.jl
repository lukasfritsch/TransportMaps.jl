"""
    optimize_adaptive_transportmap(
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights,
        maxterms::Int;
        kwargs...
    )

Adaptively optimize a triangular transport map from a target density by greedily enriching
the multi-index set across all components simultaneously.

# Arguments
- `target::AbstractMapDensity`: Target density to approximate
- `quadrature::AbstractQuadratureWeights`: Quadrature points and weights for integration
- `maxterms::Int`: Maximum total number of terms to add across all components

# Keyword Arguments
- `rectifier::AbstractRectifierFunction=Softplus()`: Rectifier function to use
- `basis::AbstractPolynomialBasis=LinearizedHermiteBasis()`: Polynomial basis
- `optimizer::Optim.AbstractOptimizer=LBFGS()`: Optimization algorithm
- `options::Optim.Options=Optim.Options()`: Optimizer options
- `validation_samples::Matrix{Float64}=Matrix{Float64}(undef,0,0)`: Samples for variance diagnostic validation

# Returns
- `M::PolynomialMap`: The optimized triangular transport map (with best validation variance diagnostic)
- `history::OptimizationHistory`: History of optimization iterations
"""
function optimize_adaptive_transportmap(
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights,
    maxterms::Int;
    rectifier::AbstractRectifierFunction=Softplus(),
    basis::AbstractPolynomialBasis=LinearizedHermiteBasis(),
    reference_density::Distributions.UnivariateDistribution=Normal(),
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options(),
    validation::Union{AbstractQuadratureWeights,Nothing}=nothing
)
    d = size(quadrature.points, 2)

    # Initialize map with constant terms only
    Λ = [multivariate_indices(0, k) for k in 1:d]

    M = PolynomialMap(Λ, rectifier, basis, reference_density)
    num_initial_coefficients = numbercoefficients(M)
    println("Initialized map with $(num_initial_coefficients) initial coefficients.")

    # Initialize history tracking
    history = MapOptimizationResult(maxterms - num_initial_coefficients + 1)

    # Precompute basis for quadrature points
    precomp = PrecomputedMapBasis(M, quadrature.points, quadrature.weights)

    # Optimize initial map
    res = optimize!(M, target, precomp, optimizer=optimizer, options=options)
    train_obj = minimum(res)

    println("Initial KL divergence (train): $train_obj")

    # Perform validation if not set to nothing
    if !isnothing(validation)
        validation_obj = kldivergence(M, target, validation)
        println("Initial KL divergence (valid): $validation_obj")
    else
        validation_obj = NaN
    end

    update_optimization_history!(
        history,
        deepcopy(M),
        train_obj,
        validation_obj,
        Float64[],
        res,
        1,
    )

    # Greedy optimization loop
    for iteration in (num_initial_coefficients+1):maxterms
        println("\nTerm $iteration / $maxterms")

        # Collect all candidate terms from reduced margins of all components
        candidates = Vector{Tuple{Int,Vector{Int}}}()  # (component_idx, multi_index)

        for k in 1:d
            Λᵣₘᵏ = reduced_margin(getmultivariateindices(M[k]))
            for α in Λᵣₘᵏ
                push!(candidates, (k, α))
            end
        end

        println("   Evaluating $(length(candidates)) candidates...")

        # Evaluate all candidates by computing gradient magnitude of KL divergence
        gradient_metrics = zeros(Float64, length(candidates))

        for (i, (k, α)) in enumerate(candidates)
            # Construct candidate map
            M_cand = deepcopy(M)
            update_multiindexset!(M_cand, α, k)

            # Compute gradient of KL divergence
            precomp_cand = PrecomputedMapBasis(M_cand, quadrature.points, quadrature.weights)
            grad = kldivergence_gradient(M_cand, target, precomp_cand)

            # Get gradient component corresponding to the new coefficient (last one for component k)
            # Find position of new coefficient in the full coefficient vector
            coeff_offset = k == 1 ? 0 : sum(numbercoefficients(M_cand[j]) for j in 1:(k-1))
            new_coeff_idx = coeff_offset + numbercoefficients(M_cand[k])

            # Use absolute value of gradient as metric
            gradient_metrics[i] = abs(grad[new_coeff_idx])
        end

        # Select candidate with maximum gradient magnitude
        best_idx = argmax(gradient_metrics)
        k_best, α_best = candidates[best_idx]

        println("   Best candidate: Component $k_best, adding term")
        println("   Gradient magnitude: $(gradient_metrics[best_idx])")

        # Add best term to the map
        update_multiindexset!(M, α_best, k_best)

        # Recompute precomputed basis
        precomp = PrecomputedMapBasis(M, quadrature.points, quadrature.weights)

        # Optimize map
        res = optimize!(M, target, precomp, optimizer=optimizer, options=options)

        # Compute objectives
        train_obj = Optim.minimum(res)
        println("   KL divergence (train): $train_obj")

        if !isnothing(validation)
            validation_obj = kldivergence(M, target, validation)
            println("   KL divergence (valid): $validation_obj")
        else
            validation_obj = NaN
        end

        # Store in history
        iter_idx = iteration - num_initial_coefficients + 1
        update_optimization_history!(
            history,
            deepcopy(M),
            train_obj,
            validation_obj,
            gradient_metrics,
            res,
            iter_idx,
        )

    end

    # Select model with best KL divergence
    if !isnothing(validation)
        best_iteration = argmin(history.test_objectives)
        println("\nBest iteration: $best_iteration")
        println("Final KL divergence (train): $(history.train_objectives[best_iteration])")
        println("Final KL divergence (valid): $(history.test_objectives[best_iteration])")
    else
        best_iteration = argmin(history.train_objectives)
        println("Final KL divergence (train): $(history.train_objectives[best_iteration])")
    end

    # Get best map
    M_best = history.maps[best_iteration]

    return M_best, history
end

# Update the polynomial map with a new multi-index α in component k
function update_multiindexset!(
    M::PolynomialMap,
    α::Vector{Int},
    k::Int,
)
    # Get k-th component to update
    component = M[k]
    coeffs = getcoefficients(component)

    # Get the current multi-index set for the k-th component and add new index
    Λ = getmultivariateindices(component)
    push!(Λ, α)

    # Reconstruct map component with updated multi-index set
    M.components[k] = PolynomialMapComponent(Λ, component.rectifier, getbasis(component), M.reference.densitytype)
    setcoefficients!(M.components[k], [coeffs..., 0.0])  # Initialize new coefficient to zero

end
