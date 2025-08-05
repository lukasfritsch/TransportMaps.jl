mutable struct PolynomialMap <: AbstractTriangularMap
    components::Vector{PolynomialMapComponent}  # Vector of map components
    reference::MapReferenceDensity
    forwarddirection::Symbol

    function PolynomialMap(
        dimension::Int,
        degree::Int,
        referencetype::Symbol = :normal,
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = HermiteBasis(),
    )
        components = [PolynomialMapComponent(k, degree, rectifier, basis) for k in 1:dimension]

        return PolynomialMap(components, referencetype)
    end

    function PolynomialMap(
        dimension::Int,
        degree::Int,
        reference::Distributions.UnivariateDistribution,
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = HermiteBasis(),
    )
        components = [PolynomialMapComponent(k, degree, rectifier, basis) for k in 1:dimension]

        return PolynomialMap(components, reference)
    end

    function PolynomialMap(components::Vector{PolynomialMapComponent})
        return PolynomialMap(components, :normal)
    end

    function PolynomialMap(components::Vector{PolynomialMapComponent}, referencetype::Symbol)
        if referencetype == :normal
            refdensity = Normal(0,1)
            reference = MapReferenceDensity(refdensity)
            return new(components, reference, :target)
        else
            error("Reference type $referencetype not supported")
        end
    end

    function PolynomialMap(components::Vector{PolynomialMapComponent}, reference::Distributions.UnivariateDistribution)
        return new(components, MapReferenceDensity(reference), :target)
    end
end

# Evaluate the polynomial map at z
function evaluate(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    return [evaluate(component, z[1:i]) for (i, component) in enumerate(M.components)]
end

# Gradient of the polynomial map at z
function gradient_zk(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(z) == length(M.components) "Dimension mismatch: z and components must have same length"
    # Compute the derivatives ∂Mᵏ/∂zₖ for each component
    return [partial_derivative_zk(component, z[1:i]) for (i, component) in enumerate(M.components)]
end

# Gradient of the map with respect to the coefficients at z
"""
    gradient_coefficients(M::PolynomialMap, z::AbstractArray{<:Real})

Compute the gradient of the polynomial map with respect to all its coefficients at point z.

For a triangular polynomial map M(z) = [M¹(z₁), M²(z₁,z₂), ..., Mᵈ(z₁,...,zᵈ)],
this function returns the gradient matrix where:
- Each row i corresponds to component Mⁱ
- Each column j corresponds to a coefficient across all components

The coefficients are ordered by component: [c₁₁, c₁₂, ..., c₂₁, c₂₂, ..., cᵈₙ]
where cᵢⱼ is the j-th coefficient of the i-th component.

# Arguments
- `M::PolynomialMap`: The polynomial map
- `z::AbstractArray{<:Real}`: Point at which to evaluate the gradient

# Returns
- `Matrix{Float64}`: Gradient matrix of size (dimension × total_coefficients)
  - Element (i,j) = ∂Mⁱ/∂cⱼ at point z

# Examples
```julia
# Create a 2D polynomial map
M = PolynomialMap(2, 2, Softplus())
setcoefficients!(M, randn(numbercoefficients(M)))

# Evaluate gradient at point z = [0.5, 1.2]
z = [0.5, 1.2]
grad_matrix = gradient_coefficients(M, z)

# grad_matrix[i, j] = ∂Mⁱ/∂cⱼ
```
"""
function gradient_coefficients(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(z) == length(M.components) "Dimension mismatch: z and components must have same length"

    n_dimensions = length(M.components)
    n_total_coeffs = numbercoefficients(M)

    # Initialize gradient matrix: (n_dimensions × n_total_coeffs)
    gradient_matrix = zeros(Float64, n_dimensions, n_total_coeffs)

    # Track coefficient index across all components
    coeff_offset = 1

    for (i, component) in enumerate(M.components)
        n_component_coeffs = length(component.coefficients)

        # For component i, compute gradient with respect to its own coefficients
        # This gives ∂Mⁱ/∂cᵢⱼ for all j in component i
        component_grad = gradient_coefficients(component, z[1:i])

        # Fill in the gradient matrix
        gradient_matrix[i, coeff_offset:coeff_offset + n_component_coeffs - 1] = component_grad

        # All other coefficients don't affect this component (triangular structure)
        # So gradient_matrix[i, other_indices] remains zero

        coeff_offset += n_component_coeffs
    end

    return gradient_matrix
end

# Compute the Jacobian determinant of the polynomial map at z
function jacobian(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"
    # Compute the jacobian determinant as ∏ₖ ∂Mᵏ/∂zₖ
    return prod(gradient_zk(M, z))
end

# Compute the gradient of log|det J_M| with respect to coefficients
function jacobian_logdet_gradient(M::PolynomialMap, z::AbstractVector{Float64})
    n_coeffs = numbercoefficients(M)
    gradient = zeros(Float64, n_coeffs)

    # For triangular maps: log|det J_M| = Σᵢ log(∂Mᵢ/∂zᵢ)
    # So ∂log|det J_M|/∂c = Σᵢ (1/(∂Mᵢ/∂zᵢ)) * ∂²Mᵢ/(∂zᵢ∂c)

    coeff_offset = 1
    for (i, component) in enumerate(M.components)
        n_comp_coeffs = length(component.coefficients)

        # Compute ∂Mᵢ/∂zᵢ (diagonal element of Jacobian)
        diagonal_deriv = partial_derivative_zk(component, z[1:i])

        # Compute gradient of ∂Mᵢ/∂zᵢ with respect to component's coefficients
        # This requires the gradient of partial_derivative_xk w.r.t. coefficients
        comp_grad = partial_derivative_zk_gradient_coefficients(component, z[1:i])

        # Add contribution: (1/diagonal_deriv) * comp_grad
        coeff_range = coeff_offset:coeff_offset + n_comp_coeffs - 1
        gradient[coeff_range] += comp_grad / diagonal_deriv

        coeff_offset += n_comp_coeffs
    end

    return gradient
end

# Compute the inverse of the polynomial map at x
function inverse(M::PolynomialMap, x::AbstractArray{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    # Initialize the inverse map
    z = Vector{Float64}(undef, length(x))
    for (i, component) in enumerate(M.components)
        z[i] = inverse(component, z[1:i-1], x[i])
    end

    return z
end

# Compute the Jacobian determinant of the inverse polynomial map at x
function inverse_jacobian(M::PolynomialMap, x::AbstractArray{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    # For triangular maps, det(J_{M⁻¹}(x)) = 1/det(J_M(M⁻¹(x)))
    # where J_M is the Jacobian of the forward map M
    return 1.0 / jacobian(M, inverse(M, x))
end

# Pullback density: Map from reference to target space
function pullback(M::PolynomialMap, x::AbstractArray{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    value = M.forwarddirection == :target ? M.reference.density(inverse(M, x)) * abs(inverse_jacobian(M, x)) :
                                            M.reference.density(evaluate(M, x)) * abs(jacobian(M, x))

    # Compute pull-back density π̂(x) = ρ(M⁻¹(x)) * |det J(M^-1(x))|
    return value
end

# Pushforward density: Map from target to reference space
function pushforward(M::PolynomialMap, target_density::Function, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    value = M.forwarddirection == :target ? target_density(evaluate(M, z)) * abs(jacobian(M, z)) :
                                            error("Can't evaluate pushforward for a map from samples!")

    # Compute push-forward density ρ(z) = π(M(z)) * |det J(M(z))|
    return value
end

# Set the coefficients in all map components.
function setcoefficients!(M::PolynomialMap, coefficients::Vector{<:Real})
    counter = 1
    for component in M.components
        setcoefficients!(component, coefficients[counter:counter+length(component.basisfunctions)-1])
        counter += length(component.basisfunctions)
    end
end

# Get the coefficients from all map components.
function getcoefficients(M::PolynomialMap)
    coefficients = Vector{Float64}(undef, numbercoefficients(M))
    counter = 1
    for component in M.components
        coefficients[counter:counter+length(component.basisfunctions)-1] .= component.coefficients
        counter += length(component.basisfunctions)
    end
    return coefficients
end

function setforwarddirection!(M::PolynomialMap, forwarddirection::Symbol)
    @assert forwarddirection in [:reference, :target] "Direction must be :reference, :target"
    M.forwarddirection = forwarddirection
end

function setoptimizationdirection!(M::PolynomialMap, optimizationdirection::Symbol)
    @assert optimizationdirection in [:forward, :backward] "Direction must be :forward, :backward"
    M.optimizationdirection = optimizationdirection
end

# Number of coefficients in the polynomial map
function numbercoefficients(M::PolynomialMap)
    return sum(length(component.coefficients) for component in M.components)
end

# Number of dimensions in the polynomial map
function numberdimensions(M::PolynomialMap)
    return length(M.components)
end

# Display method for PolynomialMap
function Base.show(io::IO, M::PolynomialMap)
    n_dims = length(M.components)
    n_coeffs = numbercoefficients(M)

    # Get common properties from the first component
    if n_dims > 0
        first_component = M.components[1]
        max_degree = maximum(sum(basis.multi_index) for basis in first_component.basisfunctions)

        # Get basis type
        basis_type = typeof(first_component.basisfunctions[1].basis_type)
        basis_name = string(basis_type)
        if basis_name == "HermiteBasis"
            basis_name = "Hermite"
        end

        # Get rectifier type
        rectifier_type = typeof(first_component.rectifier)
        rectifier_name = string(rectifier_type)

        print(io, "PolynomialMap(")
        print(io, "$n_dims-dimensional, ")
        print(io, "degree=$max_degree, ")
        print(io, "basis=$basis_name, ")
        print(io, "reference=$(M.reference.density), ")
        print(io, "rectifier=$rectifier_name, ")
        print(io, "$n_coeffs total coefficients)")
    else
        print(io, "PolynomialMap(empty)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", M::PolynomialMap)
    n_dims = length(M.components)
    n_coeffs = numbercoefficients(M)

    println(io, "PolynomialMap:")
    println(io, "  Dimensions: $n_dims")
    println(io, "  Total coefficients: $n_coeffs")
    println(io, "  Reference density: $(M.reference.density)")

    if n_dims > 0
        # Get properties from the first component (assuming all components have similar properties)
        first_component = M.components[1]
        max_degree = maximum(sum(basis.multi_index) for basis in first_component.basisfunctions)

        # Get basis type
        basis_type = typeof(first_component.basisfunctions[1].basis_type)
        basis_name = string(basis_type)
        if basis_name == "HermiteBasis"
            basis_name = "Hermite"
        end

        # Get rectifier type
        rectifier_type = typeof(first_component.rectifier)
        rectifier_name = string(rectifier_type)

        println(io, "  Maximum degree: $max_degree")
        println(io, "  Basis: $basis_name")
        println(io, "  Rectifier: $rectifier_name")

        # Show components summary
        println(io, "  Components:")
        for (i, component) in enumerate(M.components)
            n_basis = length(component.basisfunctions)
            println(io, "    Component $i: $n_basis basis functions")
        end

        # Show coefficient statistics if they're set
        all_coeffs = getcoefficients(M)
        if all(isfinite, all_coeffs)
            coeff_min = minimum(all_coeffs)
            coeff_max = maximum(all_coeffs)
            coeff_mean = sum(all_coeffs) / length(all_coeffs)
            println(io, "  Coefficients: min=$coeff_min, max=$coeff_max, mean=$coeff_mean")
        else
            println(io, "  Coefficients: uninitialized")
        end
    else
        println(io, "  (Empty map)")
    end
end
