mutable struct PolynomialMapComponent <: AbstractMapComponent # mutable due to coefficients that are optimized
    basisfunctions::Vector{MultivariateBasis}  # Vector of MultivariateBasis objects
    coefficients::Vector{Float64}  # Coefficients for the basis functions
    rectifier::AbstractRectifierFunction  # Rectifier function to apply to the partial derivatives
    index::Int  # Index k and dimension of the map component

    function PolynomialMapComponent(
        index::Int,
        degree::Int,
        rectifier::AbstractRectifierFunction = Softplus(),
        basis::AbstractPolynomialBasis = HermiteBasis()
        )
        @assert index > 0 "Index must be a positive integer"
        @assert degree > 0 "Degree must be a positive integer"

        multi_indices = multivariate_indices(degree, index)
        basisfunctions = [MultivariateBasis(multi_index, basis) for multi_index in multi_indices]
        coefficients = Vector{Float64}(undef, length(basisfunctions))

        return new(basisfunctions, coefficients, rectifier, index)
    end

    function PolynomialMapComponent(basisfunctions::Vector{MultivariateBasis}, coefficients::Vector{<:Real}, rectifier::AbstractRectifierFunction, index::Int)
        @assert length(basisfunctions) == length(coefficients) "Number of basis functions must equal number of coefficients"
        @assert index > 0 "Index must be a positive integer"

        return new(basisfunctions, coefficients, rectifier, index)
    end
end

# Compute Mᵏ according to Eq. (4.13)
function evaluate(map_component::PolynomialMapComponent, x::Vector{<:Real})
    @assert length(map_component.basisfunctions) == length(map_component.coefficients) "Number of basis functions must equal number of coefficients"
    @assert map_component.index > 0 "index must be a positive integer"
    @assert map_component.index <= length(x) "index must not exceed the dimension of x"
    @assert length(x) == length(map_component.basisfunctions[1].multi_index) "Dimension mismatch: x and multi_index must have same length"

    # First part: f(x₁, ..., x_{k-1}, 0, a)
    x₀ = copy(x)
    x₀[map_component.index] = 0.0
    f₀ = f(map_component.basisfunctions, map_component.coefficients, x₀)

    # Integrand for the integral over x̄
    integrand(x̄) = begin
        x_temp = copy(x)
        x_temp[map_component.index] = x̄
        ∂f = partial_derivative_x(map_component.basisfunctions, map_component.coefficients, x_temp, map_component.index)
        return map_component.rectifier(∂f)
    end

    # Second part: ∫g∂f: Numerical integration using Gauss-Legendre quadrature
    ∫g∂f = gaussquadrature(integrand, 100, 0., x[map_component.index])

    return f₀ + ∫g∂f
end

# Partial derivative ∂Mᵏ/∂xₖ = g(∂ₖ f(x₁, ..., x_{k-1}, xₖ)) = g(∂f/∂xₖ)
function partial_derivative_xk(map_component::PolynomialMapComponent, x::Vector{<:Real})
    @assert length(x) == length(map_component.basisfunctions[1].multi_index) "Dimension mismatch: x and multi_index must have same length"

    # Define the integrand for the partial derivative
    integrand(xₖ) = begin
        x_temp = copy(x)
        x_temp[map_component.index] = xₖ
        return evaluate(map_component, x_temp)
    end

    # ∂Mᵏ/∂xₖ = g(∂ₖ f(x₁, ..., x_{k-1}, xₖ)) = g(∂f/∂xₖ)
    ∂fᵏ = partial_derivative_x(map_component.basisfunctions, map_component.coefficients, x, map_component.index)
    ∂Mᵏ = map_component.rectifier(∂fᵏ)

    return ∂Mᵏ
end

# Inverse map for the polynomial map component using one-dimensional root finding
function inverse(
    map_component::PolynomialMapComponent,
    xₖ₋₁::Vector{<:Real},
    zₖ::Real,
)
    @assert length(xₖ₋₁) == map_component.index - 1 "Length of xₖ₋₁ must be equal to index - 1"

    # Define the residual
    fun(xₖ) = evaluate(map_component, [xₖ₋₁..., xₖ]) - zₖ
    ∂fun(xₖ) = partial_derivative_xk(map_component, [xₖ₋₁..., xₖ])

    # Define bounds for the root-finding
    lower, upper = _inverse_bound(fun)

    # Use a root-finding method to find the inverse
    x⁺, _ = hybridrootfinder(fun, ∂fun, lower, upper)

    return x⁺
end
