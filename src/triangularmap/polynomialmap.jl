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
function evaluate(M::PolynomialMap, x::Vector{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"
    return [evaluate(component, x[1:i]) for (i, component) in enumerate(M.components)]
end

# Compute the Jacobian determinant of the polynomial map at x
function jacobian(M::PolynomialMap, x::Vector{<:Real})
    @assert length(M.components) == length(x) "Number of components must match the dimension of x"
    diagonal_derivatives = [partial_derivative_xk(component, x[1:i]) for (i, component) in enumerate(M.components)]
    return prod(diagonal_derivatives)
end

# Todo: Add inverse map

# Todo : Define push-forward and pull-back methods for PolynomialMap
