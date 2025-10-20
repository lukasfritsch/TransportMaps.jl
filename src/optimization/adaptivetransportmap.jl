# Adaptive transport map optimization based on greedy basis selection
# as described in:
#   Baptista, R., Marzouk, Y., & Zahm, O. (2023). On the Representation and Learning of
#   Monotone Triangular Transport Maps. Foundations of Computational Mathematics.
#   https://doi.org/10.1007/s10208-023-09630-x

# todo: add doc-string (when everything is done)
# todo: add test-train split (and later, k-fold cross-validation)
function AdaptiveTransportMap(
    samples::Matrix{Float64},
    maxterms::Vector{Int64},
    rectifier::AbstractRectifierFunction = Softplus(),
    basis::AbstractPolynomialBasis = LinearizedHermiteBasis();
    optimizer::Optim.AbstractOptimizer = LBFGS(),
    options::Optim.Options = Optim.Options()
)
    @assert length(maxterms) == size(samples, 2) "Length of maxterms must equal number of dimensions in samples"
    # Extract number of dimensions from samples
    d = size(samples, 2)
    T = typeof(basis)
    map_components = Vector{PolynomialMapComponent{T}}()

    for k in 1:d
        println("Optimizing component $k / $d")
        component = adaptive_optimization(samples[:, 1:k], maxterms[k], rectifier, basis, optimizer, options)
        push!(map_components, component)
    end

    # Construct final map from optimized components
    M = PolynomialMap(map_components; forwarddirection=:reference)

    return M
end

# todo: add doc-string
# todo: add test-train split (and later, k-fold cross-validation)
function adaptive_optimization(
    samples::Matrix{Float64},
    maxterms::Int,
    rectifier::AbstractRectifierFunction,
    basis::AbstractPolynomialBasis,
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options
)

    d = size(samples, 2)
    # Initialize multi-index set to contain only zero index (constant term)
    Λ = multivariate_indices(1, d)

    # Initialize component with the multi-index set
    component = PolynomialMapComponent(Λ, rectifier, basis, samples)
    optimize!(component, samples, optimizer, options)

    # start greedy optimization
    for t in 1:maxterms
        # Candidates: terms in the reduced margin of Λₜ
        Λ_rm = reduced_margin(Λ)

        # store objective of all candidates and gradients
        objectives = zeros(Float64, length(Λ_rm))
        gradients = zeros(Float64, length(Λ_rm))

        # Loop over all candidates in the reduced margin
        for (i, α) in enumerate(Λ_rm)
            Λ_cand = copy(Λ)
            push!(Λ_cand, α)

            # Update component with new multi-index set
            component_cand = PolynomialMapComponent(Λ_cand, rectifier, basis, samples)
            # Copy the optimized component coefficients; set other to 0
            coeff = copy(getcoefficients(component))
            setcoefficients!(component_cand, [coeff..., 0.0]) # set coefficients for existing terms

            # evaluate objective and gradient of objective function
            objectives[i] = objective(component_cand, samples)
            println("Gradient for candidate $α: $(abs(objective_gradient!(component_cand, samples)[end]))")
            gradients[i] = abs(objective_gradient!(component_cand, samples)[end]) #! maybe like this?
        end

        # select best candidate based on maximum gradient
        α⁺ = Λ_rm[argmax(gradients)]
        push!(Λ, α⁺)

        # Update component with new multi-index set
        println("Optimizing with term $t / $maxterms")
        component = PolynomialMapComponent(Λ, rectifier, basis, samples)
        optimize!(component, samples, optimizer, options) #! maybe keep track of optimization result?

    end

    return component

end
