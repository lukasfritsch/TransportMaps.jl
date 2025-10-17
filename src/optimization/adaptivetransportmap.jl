"""
reduced_margin(Λ)

Return the reduced margin of a multi-index set Λ.

Λ is expected to be a Vector{Vector{Int}} where each inner Vector is a multi-index
of length d. The reduced margin Λ^RM is defined as the set of multi-indices α not
in Λ such that for all i with α_i != 0 we have α - e_i ∈ Λ.
"""
function reduced_margin(Λ::Vector{<:Vector{Int}}) #! See where this goes
    # quick conversion to a Set for membership tests using tuples
    present = Set{NTuple{0, Int}}()
    # handle variable dimension by using tuples of variable length
    present = Set{Tuple{Int}}()
    # Instead of parametrized tuple type, we'll convert to tuples and use Any tuple
    present = Set{Any}()
    for α in Λ
        push!(present, tuple(α...))
    end

    # determine dimension d from first element; assume Λ not empty
    if isempty(Λ)
        return Vector{Vector{Int}}()
    end
    d = length(Λ[1])

    # A helper to generate candidate neighbors (α - e_i) and to check membership
    is_in_present = x -> (tuple(x...) in present)

    reduced = Vector{Vector{Int}}()

    # To build the reduced margin, we need to look at potential α not in Λ.
    # A practical approach: find all α that are one unit above some element in Λ
    # i.e., for each β in Λ and each coordinate i, consider α = β + e_i.
    # Then α is in reduced margin iff α ∉ Λ and for all j with α_j != 0, α - e_j ∈ Λ.
    candidates = Set{Any}()
    for β in Λ
        for i in 1:d
            α = copy(β)
            α[i] += 1
            push!(candidates, tuple(α...))
        end
    end

    for tup in candidates
        α = collect(tup)
        if tup in present
            continue
        end
        ok = true
        for i in 1:d
            if α[i] != 0
                nb = copy(α)
                nb[i] -= 1
                if !(tuple(nb...) in present)
                    ok = false
                    break
                end
            end
        end
        if ok
            push!(reduced, α)
        end
    end

    return reduced
end


# Placeholder for ATM implementation

# Todo: see if Map is given as argument or just options (same as polynomialmap)
#* add test-train split (and later, cross-validation)
function optimize_adaptive(
    M::PolynomialMap,
    samples::Matrix{Float64},
    maxterms::Vector{Int64}
)


    n_components = numberdimensions(M)
    for k in 1:n_components
        println("Optimizing component $k / $n_components")
        component = get_component(M, k)
        adaptive_optimization(component, samples[:, k], maxterms[k])
    end

    return M
end

function adaptive_optimization(
    component::PolynomialMapComponent, #! or some other options
    samples::Matrix{Float64},
    maxterms::Int;
    optimizer::Optim.AbstractOptimizer = LBFGS(),
    options::Optim.Options = Optim.Options()
)

    d = numberdimensions(component)
    # Initialize multi-index set to contain only zero index (constant term)
    Λ = multivariate_indices(1, d)

    # Initialize component with the multi-index set
    component = PolynomialMapComponent(Λ) # todo: pass options for initialization

    # start greedy optimization
    for t in 1:maxterms
        println("Optimizing with term $t / $maxterms")

        res = optimize!(component, samples, optimizer, options)

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
            component_cand = PolynomialMapComponent(Λ_cand) # copy the optimized component coefficients; set other to 0
            coeff = copy(getcoefficients(component))
            setcoefficients!(component_cand, coeff[1:length(getcoefficients(component_cand))]) # set coefficients for existing terms

            # evaluate objective and gradient of objective function
            objectives[i] = objective(component_cand, samples)
            gradients[i] = abs(objective_gradient!(component_cand, samples)) #! maybe like this?
        end

        # select best candidate based on maximum gradient
        α⁺ = Λ_rm[argmax(gradients)]
        push!(Λ, α⁺)

        # Update component with new multi-index set
        component = PolynomialMapComponent(Λ)
    end

    return component

end
