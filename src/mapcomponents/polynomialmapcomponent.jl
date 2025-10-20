struct PolynomialMapComponent{T<:AbstractPolynomialBasis} <: AbstractMapComponent # mutable due to coefficients that are optimized
    basisfunctions::Vector{MultivariateBasis{T}}  # Vector of MultivariateBasis objects
    coefficients::Vector{Float64}  # Coefficients for the basis functions
    rectifier::AbstractRectifierFunction  # Rectifier function to apply to the partial derivatives
    index::Int  # Index k and dimension of the map component

    function PolynomialMapComponent(
        index::Int,
        degree::Int,
        rectifier::AbstractRectifierFunction=Softplus(),
        basis::AbstractPolynomialBasis=HermiteBasis(),
        map_type::Symbol=:total
    )
        @assert index > 0 "Index must be a positive integer"
        @assert degree > 0 "Degree must be a positive integer"
        @assert map_type in [:total, :diagonal, :no_mixed] "Invalid map_type. Supported types are :total, :diagonal, :no_mixed"

        multi_indices = multivariate_indices(degree, index, mode=map_type)
        basisfunctions = [MultivariateBasis(multiindexset, typeof(basis)) for multiindexset in multi_indices]
        coefficients = zeros(length(basisfunctions))
        T = typeof(basis)

        return new{T}(basisfunctions, coefficients, rectifier, index)
    end

    # Constructor that builds basis functions using an analytical reference density
    function PolynomialMapComponent(
        index::Int,
        degree::Int,
        rectifier::AbstractRectifierFunction,
        basis::AbstractPolynomialBasis,
        density::Distributions.UnivariateDistribution,
        map_type::Symbol=:total
    )
        @assert index > 0 "Index must be a positive integer"
        @assert degree > 0 "Degree must be a positive integer"
        @assert map_type in [:total, :diagonal, :no_mixed] "Invalid map_type. Supported types are :total, :diagonal, :no_mixed"

        T = typeof(basis)
        multi_indices = multivariate_indices(degree, index, mode=map_type)
        basisfunctions = Vector{MultivariateBasis{T}}(undef, length(multi_indices))

        for (i, multiindexset) in enumerate(multi_indices)
            # Build per-dimension univariate bases with the correct degree
            dim = length(multiindexset)
            uni_bases = Vector{T}(undef, dim)

            for j in 1:dim
                deg_j = multiindexset[j]

                if isa(basis, HermiteBasis)
                    uni_bases[j] = HermiteBasis()
                elseif isa(basis, LinearizedHermiteBasis)
                    uni_bases[j] = LinearizedHermiteBasis(density, deg_j, index)
                elseif isa(basis, GaussianWeightedHermiteBasis)
                    uni_bases[j] = GaussianWeightedHermiteBasis()
                elseif isa(basis, CubicSplineHermiteBasis)
                    uni_bases[j] = CubicSplineHermiteBasis(density)
                end
            end

            basisfunctions[i] = MultivariateBasis(multiindexset, uni_bases)
        end

        coefficients = zeros(length(basisfunctions))
        return new{T}(basisfunctions, coefficients, rectifier, index)
    end

    # Constructor that builds basis functions using an analytical reference density
    function PolynomialMapComponent(
        index::Int,
        degree::Int,
        rectifier::AbstractRectifierFunction,
        basis::AbstractPolynomialBasis,
        samples::AbstractMatrix{Float64},
        map_type::Symbol=:total
    )
        @assert index > 0 "Index must be a positive integer"
        @assert degree > 0 "Degree must be a positive integer"
        @assert map_type in [:total, :diagonal, :no_mixed] "Invalid map_type. Supported types are :total, :diagonal, :no_mixed"

        # Construct multi-indices for the polynomial basis
        multi_indices = multivariate_indices(degree, index, mode=map_type)
        return PolynomialMapComponent(multi_indices, rectifier, basis, samples)
    end

    function PolynomialMapComponent(
        multi_indices::Vector{Vector{Int}},
        rectifier::AbstractRectifierFunction,
        basis::AbstractPolynomialBasis,
        samples::AbstractMatrix{Float64}
    )
        # Determine index from multi_indices and type of basis
        index = length(multi_indices[1])
        T = typeof(basis)
        basisfunctions = Vector{MultivariateBasis{T}}(undef, length(multi_indices))

        for (i, multiindexset) in enumerate(multi_indices)
            # Build per-dimension univariate bases with the correct degree
            dim = length(multiindexset)
            uni_bases = Vector{typeof(basis)}(undef, dim)

            for j in 1:dim
                deg_j = multiindexset[j]
                if isa(basis, HermiteBasis)
                    uni_bases[j] = HermiteBasis()
                elseif isa(basis, LinearizedHermiteBasis)
                    uni_bases[j] = LinearizedHermiteBasis(samples[:, j], deg_j, index)
                elseif isa(basis, GaussianWeightedHermiteBasis)
                    uni_bases[j] = GaussianWeightedHermiteBasis()
                elseif isa(basis, CubicSplineHermiteBasis)
                    uni_bases[j] = CubicSplineHermiteBasis(samples[:, j])
                end
            end

            basisfunctions[i] = MultivariateBasis(multiindexset, uni_bases)
        end

        coefficients = zeros(length(basisfunctions))
        return new{T}(basisfunctions, coefficients, rectifier, index)
    end

    function PolynomialMapComponent(basisfunctions::Vector{MultivariateBasis{T}}, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction, index::Int) where T<:AbstractPolynomialBasis
        @assert length(basisfunctions) == length(coefficients) "Number of basis functions must equal number of coefficients"
        @assert index > 0 "Index must be a positive integer"

        return new{T}(basisfunctions, coefficients, rectifier, index)
    end
end

# Compute Mᵏ according to Eq. (4.13) for a single input vector
function evaluate(map_component::PolynomialMapComponent, z::Vector{Float64})
    @assert length(map_component.basisfunctions) == length(map_component.coefficients) "Number of basis functions must equal number of coefficients"
    @assert map_component.index > 0 "index must be a positive integer"
    @assert map_component.index <= length(z) "index must not exceed the dimension of z"
    @assert length(z) == length(map_component.basisfunctions[1].multiindexset) "Dimension mismatch: z and multiindexset must have same length"

    # First part: f(z₁, ..., z_{k-1}, 0, a)
    z₀ = copy(z)
    z₀[map_component.index] = 0.0
    f₀ = f(map_component.basisfunctions, map_component.coefficients, z₀)

    # Integrand for the integral over z̄
    integrand(z̄) = begin
        z_temp = copy(z)
        z_temp[map_component.index] = z̄
        ∂f = partial_derivative_z(map_component.basisfunctions, map_component.coefficients, z_temp, map_component.index)
        return map_component.rectifier(∂f)
    end

    # Second part: ∫g∂f: Numerical integration using Gauss-Legendre quadrature
    ∫g∂f = gaussquadrature(integrand, 100, 0., z[map_component.index])

    return f₀ + ∫g∂f
end

evaluate(map_component::PolynomialMapComponent, z::AbstractVector{<:Real}) = evaluate(map_component, Vector{Float64}(z))

# Compute Mᵏ according to Eq. (4.13) for multiple input vectors using multithreading
function evaluate(map_component::PolynomialMapComponent, Z::Matrix{Float64})
    @assert length(map_component.basisfunctions) == length(map_component.coefficients) "Number of basis functions must equal number of coefficients"
    @assert map_component.index > 0 "index must be a positive integer"
    @assert size(Z, 2) == length(map_component.basisfunctions[1].multiindexset) "Dimension mismatch: Z columns and multiindexset must have same length"

    n_points = size(Z, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to evaluate each point
    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        @assert map_component.index <= length(z_point) "index must not exceed the dimension of z_point"
        results[i] = evaluate(map_component, z_point)
    end

    return results
end

evaluate(map_component::PolynomialMapComponent, Z::AbstractMatrix{<:Real}) = evaluate(map_component, Matrix{Float64}(Z))

# Partial derivative ∂Mᵏ/∂zₖ = g(∂ₖ f(x₁, ..., x_{k-1}, zₖ)) = g(∂f/∂zₖ) for a single input vector
function partial_derivative_zk(map_component::PolynomialMapComponent, z::Vector{Float64})
    @assert length(z) == length(map_component.basisfunctions[1].multiindexset) "Dimension mismatch: z and multiindexset must have same length"

    # Define the integrand for the partial derivative
    integrand(zₖ) = begin
        z_temp = copy(z)
        z_temp[map_component.index] = zₖ
        return evaluate(map_component, z_temp)
    end

    # ∂Mᵏ/∂zₖ = g(∂ₖ f(x₁, ..., x_{k-1}, zₖ)) = g(∂f/∂zₖ)
    ∂fᵏ = partial_derivative_z(map_component.basisfunctions, map_component.coefficients, z, map_component.index)
    ∂Mᵏ = map_component.rectifier(∂fᵏ)

    return ∂Mᵏ
end

# Partial derivative ∂Mᵏ/∂zₖ for multiple input vectors using multithreading
function partial_derivative_zk(map_component::PolynomialMapComponent, Z::Matrix{Float64})
    @assert size(Z, 2) == length(map_component.basisfunctions[1].multiindexset) "Dimension mismatch: Z columns and multiindexset must have same length"

    n_points = size(Z, 1)

    # Preallocate result vector
    results = Vector{Float64}(undef, n_points)

    # Use multithreading to compute partial derivative for each point
    Threads.@threads for i in 1:n_points
        z_point = Z[i, :]
        results[i] = partial_derivative_zk(map_component, z_point)
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
    gradient_coefficients(map_component::PolynomialMapComponent, z::Vector{Float64})

Compute the gradient of the polynomial map component with respect to its coefficients at point z.

For a polynomial map component Mᵏ(z) defined as:
Mᵏ(z) = f(z₁, ..., z_{k-1}, 0) + ∫₀^{z_k} g(∂f/∂x_k) dx_k

where f(z) = Σᵢ cᵢ ψᵢ(z) and g is the rectifier function, the gradient is:
∂Mᵏ/∂cⱼ = ψⱼ(z₁, ..., z_{k-1}, 0) + ∫₀^{z_k} g'(∂f/∂x_k) ∂ψⱼ/∂x_k dx_k

# Arguments
- `map_component::PolynomialMapComponent`: The polynomial map component
- `z::Vector{Float64}`: Point at which to evaluate the gradient (must match dimension of basis functions)

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
function gradient_coefficients(map_component::PolynomialMapComponent, z::Vector{Float64})
    @assert length(z) == length(map_component.basisfunctions[1].multiindexset) "Dimension mismatch: z and basis functions must have same length"

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

function degree(map_component::PolynomialMapComponent)
    return maximum(sum(basis.multiindexset) for basis in map_component.basisfunctions)
end

# Inverse map for the polynomial map component using one-dimensional root finding
function inverse(
    map_component::PolynomialMapComponent,
    zₖ₋₁::Vector{Float64},
    xₖ::Real,
)
    @assert length(zₖ₋₁) == map_component.index - 1 "Length of zₖ₋₁ must be equal to index - 1"

    # Define the residual
    fun(zₖ) = evaluate(map_component, [zₖ₋₁..., zₖ]) - xₖ
    ∂fun(zₖ) = partial_derivative_zk(map_component, [zₖ₋₁..., zₖ])

    # Define bounds for the root-finding
    lower, upper = _inverse_bound(fun)

    # Use a root-finding method to find the inverse
    z⁺, _ = hybridrootfinder(fun, ∂fun, lower, upper)

    return z⁺
end

function setcoefficients!(map_component::PolynomialMapComponent, coefficients::Vector{Float64})
    @assert length(coefficients) == length(map_component.coefficients) "Length of coefficients must match the number of basis functions."
    map_component.coefficients .= coefficients
end

# Allow setting coefficients with any real-valued vector (converts to Float64)
setcoefficients!(map_component::PolynomialMapComponent, coefficients::AbstractVector{<:Real}) =
    setcoefficients!(map_component, Vector{Float64}(coefficients))

# Get coefficients from a single PolynomialMapComponent
function getcoefficients(map_component::PolynomialMapComponent)
    return copy(map_component.coefficients)
end

function getmultiindexsets(map_component::PolynomialMapComponent)
    # Stack each basis.multiindexset as a row in a matrix
    multiindices = [basis.multiindexset for basis in map_component.basisfunctions]
    return permutedims(hcat(multiindices...))
end

# Make PolynomialMapComponent callable: component(z) instead of evaluate(component, z)
Base.@propagate_inbounds (component::PolynomialMapComponent)(z::AbstractVector{<:Real}) = evaluate(component, z)
Base.@propagate_inbounds (component::PolynomialMapComponent)(Z::AbstractMatrix{<:Real}) = evaluate(component, Z)

# Display method for PolynomialMapComponent
function Base.show(io::IO, component::PolynomialMapComponent)
    n_basis = length(component.basisfunctions)
    n_coeffs = length(component.coefficients)

    # Get the maximum degree from the basis functions
    max_degree = degree(component)

    # Get basis type from the first basis function
    basis_type = typeof(component.basisfunctions[1].univariatebases[1])
    basis_name = string(basis_type)

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
    max_degree = degree(component)

    # Get basis type
    basis_type = typeof(component.basisfunctions[1].univariatebases[1])
    basis_name = string(basis_type)

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
            println(io, "    $(component.basisfunctions[i].multiindexset)")
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
