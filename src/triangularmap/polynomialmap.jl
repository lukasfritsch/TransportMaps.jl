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
