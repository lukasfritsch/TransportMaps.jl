struct PolynomialMapComponent <: AbstractMapComponent # mutable due to coefficients that are optimized
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
        coefficients = zeros(length(basisfunctions))

        return new(basisfunctions, coefficients, rectifier, index)
    end

    function PolynomialMapComponent(basisfunctions::Vector{MultivariateBasis}, coefficients::Vector{<:Real}, rectifier::AbstractRectifierFunction, index::Int)
        @assert length(basisfunctions) == length(coefficients) "Number of basis functions must equal number of coefficients"
        @assert index > 0 "Index must be a positive integer"

        return new(basisfunctions, coefficients, rectifier, index)
    end
end

# Compute Mᵏ according to Eq. (4.13) for a single input vector
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
        ∂f = partial_derivative_z(map_component.basisfunctions, map_component.coefficients, x_temp, map_component.index)
        return map_component.rectifier(∂f)
    end

    # Second part: ∫g∂f: Numerical integration using Gauss-Legendre quadrature
    ∫g∂f = gaussquadrature(integrand, 100, 0., x[map_component.index])

    return f₀ + ∫g∂f
end

# Compute Mᵏ according to Eq. (4.13) for multiple input vectors using multithreading
function evaluate(map_component::PolynomialMapComponent, X::Matrix{<:Real})
    @assert length(map_component.basisfunctions) == length(map_component.coefficients) "Number of basis functions must equal number of coefficients"
    @assert map_component.index > 0 "index must be a positive integer"
    @assert size(X, 2) == length(map_component.basisfunctions[1].multi_index) "Dimension mismatch: X columns and multi_index must have same length"

    n_points = size(X, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to evaluate each point
    Threads.@threads for i in 1:n_points
        x_point = X[i, :]
        @assert map_component.index <= length(x_point) "index must not exceed the dimension of x_point"
        results[i] = evaluate(map_component, x_point)
    end

    return results
end

# Partial derivative ∂Mᵏ/∂xₖ = g(∂ₖ f(x₁, ..., x_{k-1}, xₖ)) = g(∂f/∂xₖ) for a single input vector
function partial_derivative_zk(map_component::PolynomialMapComponent, x::Vector{<:Real})
    @assert length(x) == length(map_component.basisfunctions[1].multi_index) "Dimension mismatch: x and multi_index must have same length"

    # Define the integrand for the partial derivative
    integrand(xₖ) = begin
        x_temp = copy(x)
        x_temp[map_component.index] = xₖ
        return evaluate(map_component, x_temp)
    end

    # ∂Mᵏ/∂xₖ = g(∂ₖ f(x₁, ..., x_{k-1}, xₖ)) = g(∂f/∂xₖ)
    ∂fᵏ = partial_derivative_z(map_component.basisfunctions, map_component.coefficients, x, map_component.index)
    ∂Mᵏ = map_component.rectifier(∂fᵏ)

    return ∂Mᵏ
end

# Partial derivative ∂Mᵏ/∂xₖ for multiple input vectors using multithreading
function partial_derivative_zk(map_component::PolynomialMapComponent, X::Matrix{<:Real})
    @assert size(X, 2) == length(map_component.basisfunctions[1].multi_index) "Dimension mismatch: X columns and multi_index must have same length"

    n_points = size(X, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to compute partial derivative for each point
    Threads.@threads for i in 1:n_points
        x_point = X[i, :]
        results[i] = partial_derivative_zk(map_component, x_point)
    end

    return results
end

# Compute gradient of ∂Mᵏ/∂zₖ with respect to coefficients
function partial_derivative_zk_gradient_coefficients(component::PolynomialMapComponent, z::Vector{Float64})
    # ∂Mᵏ/∂zₖ = g(∂f/∂zₖ), where g is the rectifier
    # So ∂²Mᵏ/(∂zₖ∂c) = g'(∂f/∂zₖ) * ∂²f/(∂zₖ∂c)

    # Compute ∂f/∂zₖ
    ∂f = partial_derivative_z(component.basisfunctions, component.coefficients, z, component.index)

    # Compute g'(∂f/∂zₖ)
    g_prime = derivative(component.rectifier, ∂f)

    # Compute ∂²f/(∂zₖ∂c) = [∂ψⱼ/∂zₖ for j in 1:n_coeffs]
    n_coeffs = length(component.coefficients)
    ∂²f_∂zₖ∂c = zeros(Float64, n_coeffs)

    for j in 1:n_coeffs
        ∂²f_∂zₖ∂c[j] = partial_derivative_z(component.basisfunctions[j], z, component.index)
    end

    return g_prime * ∂²f_∂zₖ∂c
end


# Gradient of the map component with respect to the coefficients at z
"""
    gradient_coefficients(map_component::PolynomialMapComponent, z::Vector{<:Real})

Compute the gradient of the polynomial map component with respect to its coefficients at point z.

For a polynomial map component Mᵏ(z) defined as:
Mᵏ(z) = f(z₁, ..., z_{k-1}, 0) + ∫₀^{z_k} g(∂f/∂x_k) dx_k

where f(z) = Σᵢ cᵢ ψᵢ(z) and g is the rectifier function, the gradient is:
∂Mᵏ/∂cⱼ = ψⱼ(z₁, ..., z_{k-1}, 0) + ∫₀^{z_k} g'(∂f/∂x_k) ∂ψⱼ/∂x_k dx_k

# Arguments
- `map_component::PolynomialMapComponent`: The polynomial map component
- `z::Vector{<:Real}`: Point at which to evaluate the gradient (must match dimension of basis functions)

# Returns
- `Vector{Float64}`: Gradient vector with respect to coefficients, same length as `map_component.coefficients`

# Examples
```julia
# Create a 2D polynomial map component for the 2nd coordinate
pmc = PolynomialMapComponent(2, 2, Softplus())
setcoefficients!(pmc, randn(length(pmc.coefficients)))

# Evaluate gradient at point z = [0.5, 1.2]
z = [0.5, 1.2]
grad = gradient_coefficients(pmc, z)
```
"""
function gradient_coefficients(map_component::PolynomialMapComponent, z::Vector{<:Real})
    @assert length(z) == length(map_component.basisfunctions[1].multi_index) "Dimension mismatch: z and basis functions must have same length"

    # First part: gradient of f₀ = f(x₁, ..., x_{k-1}, 0) w.r.t. coefficients
    z₀ = copy(z)
    z₀[map_component.index] = 0.0
    ∇f₀ = gradient_coefficients(map_component.basisfunctions, z₀)

    # Second part: gradient of the integral ∫₀^{z_k} g(∂f/∂x_k) dx_k w.r.t. coefficients
    # For each coefficient cⱼ, we need: ∫₀^{z_k} g'(∂f/∂x_k) * ∂²f/∂x_k∂cⱼ dx_k
    # Since ∂²f/∂x_k∂cⱼ = ∂ψⱼ/∂x_k (basis function derivative), we can compute this

    n_coeffs = length(map_component.coefficients)
    ∇integral = zeros(Float64, n_coeffs)

    for j in 1:n_coeffs
        # Integrand for coefficient j: g'(∂f/∂x_k) * ∂ψⱼ/∂x_k
        integrand_j(x̄) = begin
            z_temp = copy(z)
            z_temp[map_component.index] = x̄

            # Compute ∂f/∂x_k at this point
            ∂f = partial_derivative_z(map_component.basisfunctions, map_component.coefficients, z_temp, map_component.index)

            # Compute derivative of rectifier g'(∂f/∂x_k)
            g_prime = derivative(map_component.rectifier, ∂f)

            # Compute ∂ψⱼ/∂x_k (derivative of j-th basis function w.r.t. x_k)
            ∂ψⱼ = partial_derivative_z(map_component.basisfunctions[j], z_temp, map_component.index)

            return g_prime * ∂ψⱼ
        end

        # Integrate from 0 to z[k]
        ∇integral[j] = gaussquadrature(integrand_j, 100, 0., z[map_component.index])
    end

    return ∇f₀ + ∇integral
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
    ∂fun(xₖ) = partial_derivative_zk(map_component, [xₖ₋₁..., xₖ])

    # Define bounds for the root-finding
    lower, upper = _inverse_bound(fun)

    # Use a root-finding method to find the inverse
    x⁺, _ = hybridrootfinder(fun, ∂fun, lower, upper)

    return x⁺
end

function setcoefficients!(map_component::PolynomialMapComponent, coefficients::Vector{<:Real})
    @assert length(coefficients) == length(map_component.coefficients) "Length of coefficients must match the number of basis functions."
    map_component.coefficients .= coefficients
end

# Display method for PolynomialMapComponent
function Base.show(io::IO, component::PolynomialMapComponent)
    n_basis = length(component.basisfunctions)
    n_coeffs = length(component.coefficients)

    # Get the maximum degree from the basis functions
    max_degree = maximum(sum(basis.multi_index) for basis in component.basisfunctions)

    # Get basis type from the first basis function
    basis_type = typeof(component.basisfunctions[1].basis_type)
    basis_name = string(basis_type)
    if basis_name == "HermiteBasis"
        basis_name = "Hermite"
    end

    # Get rectifier type
    rectifier_type = typeof(component.rectifier)
    rectifier_name = string(rectifier_type)

    print(io, "PolynomialMapComponent(")
    print(io, "index=$(component.index), ")
    print(io, "degree=$max_degree, ")
    print(io, "basis=$basis_name, ")
    print(io, "rectifier=$rectifier_name, ")
    print(io, "$n_basis basis functions)")
end

function Base.show(io::IO, ::MIME"text/plain", component::PolynomialMapComponent)
    n_basis = length(component.basisfunctions)
    max_degree = maximum(sum(basis.multi_index) for basis in component.basisfunctions)

    # Get basis type
    basis_type = typeof(component.basisfunctions[1].basis_type)
    basis_name = string(basis_type)
    if basis_name == "HermiteBasis"
        basis_name = "Hermite"
    end

    # Get rectifier type
    rectifier_type = typeof(component.rectifier)
    rectifier_name = string(rectifier_type)

    println(io, "PolynomialMapComponent:")
    println(io, "  Index: $(component.index)")
    println(io, "  Maximum degree: $max_degree")
    println(io, "  Basis: $basis_name")
    println(io, "  Rectifier: $rectifier_name")
    println(io, "  Number of basis functions: $n_basis")

    # Show first few multi-indices as examples
    if n_basis > 0
        println(io, "  Multi-indices (first $(min(5, n_basis))):")
        for i in 1:min(5, n_basis)
            println(io, "    $(component.basisfunctions[i].multi_index)")
        end
        if n_basis > 5
            println(io, "    ... and $(n_basis - 5) more")
        end
    end

    # Show coefficient statistics if they're set
    if all(isfinite, component.coefficients)
        coeff_min = minimum(component.coefficients)
        coeff_max = maximum(component.coefficients)
        coeff_mean = sum(component.coefficients) / length(component.coefficients)
        println(io, "  Coefficients: min=$coeff_min, max=$coeff_max, mean=$coeff_mean")
    else
        println(io, "  Coefficients: uninitialized")
    end
end
