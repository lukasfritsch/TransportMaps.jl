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

# Evaluate the polynomial map at x
function evaluate(M::PolynomialMap, x::AbstractArray{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    return [evaluate(component, x[1:i]) for (i, component) in enumerate(M.components)]
end

# Compute the Jacobian determinant of the polynomial map at x
function jacobian(M::PolynomialMap, x::AbstractArray{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    # Compute the derivatives ∂Mᵏ/∂xₖ for each component
    diagonal_derivatives = [partial_derivative_xk(component, x[1:i]) for (i, component) in enumerate(M.components)]

    return prod(diagonal_derivatives)
end

# Compute the inverse of the polynomial map at z
function inverse(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    # Initialize the inverse map
    x = Vector{Float64}(undef, length(z))
    for (i, component) in enumerate(M.components)
        x[i] = inverse(component, x[1:i-1], z[i])
    end

    return x
end

# Compute the Jacobian determinant of the inverse polynomial map at z
function inverse_jacobian(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    # Compute the Jacobian determinant of the inverse map
    J_inv = jacobian(M, inverse(M, z))

    return 1.0 / J_inv
end

# Pullback density: Map from reference to target space
function pullback(M::PolynomialMap, z::AbstractArray{<:Real})
    @assert length(M.components) == length(z) "Number of components must match the dimension of z"

    rho(z) = pdf(MvNormal(zeros(length(M.components)), I(length(M.components))), z)

    # Compute pull-back density π̂(z) = ρ(M⁻¹(z)) * |det J|
    return rho(inverse(M, z) * abs(inverse_jacobian(M, z)))
end

# Pushforward density: Map from target to reference space
function pushforward(M::PolynomialMap, π::Function, x::AbstractArray{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"

    # Compute push-forward density ρ(x) = π(M(x)) * |det J|
    return π(evaluate(M, x)) * abs(jacobian(M, x))

end

# Set the coefficients in all map components.
function setcoefficients!(M::PolynomialMap, coefficients::Vector{<:Real})
    counter = 1
    for component in M.components
        setcoefficients!(component, coefficients[counter:counter+length(component.basisfunctions)-1])
        counter += length(component.basisfunctions)
    end
end

function getcoefficients(M::PolynomialMap)
    coefficients = Vector{Float64}(undef, numbercoefficients(M))
    counter = 1
    for component in M.components
        coefficients[counter:counter+length(component.basisfunctions)-1] .= component.coefficients
        counter += length(component.basisfunctions)
    end
    return coefficients
end

function numbercoefficients(M::PolynomialMap)
    return sum(length(component.coefficients) for component in M.components)
end
