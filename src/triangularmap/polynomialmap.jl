mutable struct PolynomialMap <: AbstractTriangularMap
    components::Vector{PolynomialMapComponent}  # Vector of map components

    function PolynomialMap(
        dimension::Int,
        degree::Int,
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = HermiteBasis()
    )
        components = [PolynomialMapComponent(k, degree, rectifier, basis) for k in 1:dimension]
        return new(components)
    end

    function PolynomialMap(components::Vector{PolynomialMapComponent})
        return new(components)
    end
end

# Evaluate the polynomial map at z
function evaluate(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    return [evaluate(component, z[1:i]) for (i, component) in enumerate(M.components)]
end

# Compute the Jacobian determinant of the polynomial map at z
function jacobian(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    # Compute the derivatives ∂Mᵏ/∂xₖ for each component
    diagonal_derivatives = [partial_derivative_xk(component, z[1:i]) for (i, component) in enumerate(M.components)]

    return prod(diagonal_derivatives)
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

    # Compute the Jacobian determinant of the inverse map
    J_inv = jacobian(M, inverse(M, x))

    return 1.0 / J_inv
end

# Pullback density: Map from reference to target space 𝑋 ↦ 𝑍
function pullback(M::PolynomialMap, x::AbstractArray{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    reference_density(z) = pdf(MvNormal(zeros(length(M.components)), I(length(M.components))), z)

    # Compute pull-back density π̂(x) = ρ(M⁻¹(x)) * |det J(M^-1(x))|
    return reference_density(inverse(M, x) * abs(inverse_jacobian(M, x)))
end

# Pushforward density: Map from target to reference space 𝑍 ↦ 𝑋
function pushforward(M::PolynomialMap, target_density::Function, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    # Compute push-forward density ρ(z) = π(M(z)) * |det J(M(z))|
    return target_density(evaluate(M, z)) * abs(jacobian(M, z))
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
