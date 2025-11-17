"""
    PrecomputedBasis

Stores precomputed basis function evaluations for efficient optimization.

When optimizing map coefficients from samples, the basis functions are evaluated
repeatedly at the same sample points with different coefficients. This structure
precomputes and stores these evaluations to avoid redundant computation.

For exact computation of Mᵏ(z) = f(z₁,...,z_{k-1},0) + ∫₀^{z_k} g(∂f/∂x_k) dx_k,
we precompute basis evaluations at quadrature nodes for each sample.

# Fields
- `Ψ₀::Matrix{Float64}`: Basis evaluations at (z₁,...,z_{k-1},0) (n_samples × n_basis)
- `∂Ψ_z::Matrix{Float64}`: Partial derivatives ∂ψⱼ/∂zₖ at the sample points z (n_samples × n_basis)
- `Ψ_quad::Array{Float64,3}`: Basis evaluations at quadrature nodes (n_samples × n_quad × n_basis)
- `∂Ψ_quad::Array{Float64,3}`: Partial derivatives at quadrature nodes (n_samples × n_quad × n_basis)
- `quad_weights::Vector{Float64}`: Quadrature weights (n_quad,)
- `quad_scales::Vector{Float64}`: Integration scale factors 0.5*(z_k - 0) for each sample (n_samples,)
- `n_samples::Int`: Number of sample points
- `n_basis::Int`: Number of basis functions
- `n_quad::Int`: Number of quadrature points
"""
struct PrecomputedBasis
    Ψ₀::Matrix{Float64}              # Ψ₀[i,j] = ψⱼ(z₁,...,z_{k-1},0) for sample i
    ∂Ψ_z::Matrix{Float64}            # ∂Ψ_z[i,j] = ∂ψⱼ/∂zₖ(z) at sample point z
    Ψ_quad::Array{Float64,3}         # Ψ_quad[i,q,j] = ψⱼ at quadrature point q for sample i
    ∂Ψ_quad::Array{Float64,3}        # ∂Ψ_quad[i,q,j] = ∂ψⱼ/∂zₖ at quadrature point q for sample i
    quad_weights::Vector{Float64}    # Quadrature weights
    quad_scales::Vector{Float64}     # Scale factor for each sample (0.5 * z_k)
    n_samples::Int
    n_basis::Int
    n_quad::Int

    function PrecomputedBasis(
        Ψ₀::Matrix{Float64},
        ∂Ψ_z::Matrix{Float64},
        Ψ_quad::Array{Float64,3},
        ∂Ψ_quad::Array{Float64,3},
        quad_weights::Vector{Float64},
        quad_scales::Vector{Float64}
    )
        n_samples, n_basis = size(Ψ₀)
        n_quad = length(quad_weights)

        @assert size(∂Ψ_z) == (n_samples, n_basis) "∂Ψ_z dimensions mismatch"
        @assert size(Ψ_quad) == (n_samples, n_quad, n_basis) "Ψ_quad dimensions mismatch"
        @assert size(∂Ψ_quad) == (n_samples, n_quad, n_basis) "∂Ψ_quad dimensions mismatch"
        @assert length(quad_scales) == n_samples "quad_scales length must match n_samples"

        new(Ψ₀, ∂Ψ_z, Ψ_quad, ∂Ψ_quad, quad_weights, quad_scales, n_samples, n_basis, n_quad)
    end
end

"""
    PrecomputedBasis(component::PolynomialMapComponent, samples::Matrix{Float64}; n_quad::Int=64)

Precompute basis function evaluations and their partial derivatives at quadrature nodes
for all samples. This enables exact computation of the map component and its gradient
using precomputed values.

# Arguments
- `component::PolynomialMapComponent`: The map component whose basis functions to evaluate
- `samples::Matrix{Float64}`: Sample points (n_samples × dimension)
- `n_quad::Int=64`: Number of Gauss-Legendre quadrature points (default: 64)

# Returns
- `PrecomputedBasis`: Structure containing precomputed evaluations at quadrature nodes
"""
function PrecomputedBasis(component::PolynomialMapComponent, samples::Matrix{Float64}; n_quad::Int=64)
    n_samples = size(samples, 1)
    n_basis = length(component.basisfunctions)
    k = component.index  # The component index

    # Get Gauss-Legendre quadrature points and weights on [-1, 1]
    quad_points_std, quad_weights = gausslegendre(n_quad)

    # Preallocate arrays
    Ψ₀ = Matrix{Float64}(undef, n_samples, n_basis)
    ∂Ψ_z = Matrix{Float64}(undef, n_samples, n_basis)
    Ψ_quad = Array{Float64,3}(undef, n_samples, n_quad, n_basis)
    ∂Ψ_quad = Array{Float64,3}(undef, n_samples, n_quad, n_basis)
    quad_scales = Vector{Float64}(undef, n_samples)

    # Precompute for each sample
    Threads.@threads for i in 1:n_samples
        # Create thread-local buffer to avoid repeated allocations
        z_buffer = Vector{Float64}(undef, k)

        z = samples[i, :]
        z_k = z[k]

        # Scale factor for this sample: integral from 0 to z_k
        # Gauss-Legendre is on [-1,1], we map to [0, z_k]
        scale = 0.5 * z_k
        shift = 0.5 * z_k
        quad_scales[i] = scale

        # Evaluate at z with z_k = 0 for f₀ term
        # Reuse z_buffer to avoid allocation
        @inbounds for idx in 1:k-1
            z_buffer[idx] = z[idx]
        end
        z_buffer[k] = 0.0

        @inbounds for j in 1:n_basis
            Ψ₀[i, j] = evaluate(component.basisfunctions[j], z_buffer)
        end

        # Evaluate ∂ψⱼ/∂zₖ at the actual sample point z
        @inbounds for j in 1:n_basis
            ∂Ψ_z[i, j] = partial_derivative_z(component.basisfunctions[j], z, k)
        end

        # Evaluate at quadrature points
        for q in 1:n_quad
            # Map quadrature point from [-1, 1] to [0, z_k]
            x_k = quad_points_std[q] * scale + shift

            # Reuse z_buffer to avoid allocation
            @inbounds for idx in 1:k-1
                z_buffer[idx] = z[idx]
            end
            z_buffer[k] = x_k

            @inbounds for j in 1:n_basis
                Ψ_quad[i, q, j] = evaluate(component.basisfunctions[j], z_buffer)
                ∂Ψ_quad[i, q, j] = partial_derivative_z(component.basisfunctions[j], z_buffer, k)
            end
        end
    end

    return PrecomputedBasis(Ψ₀, ∂Ψ_z, Ψ_quad, ∂Ψ_quad, quad_weights, quad_scales)
end

"""
    evaluate_f₀(precomp::PrecomputedBasis, coefficients::Vector{Float64})

Evaluate f₀(z) = f(z₁,...,z_{k-1},0) = Σⱼ cⱼ ψⱼ(z₁,...,z_{k-1},0) for all samples.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations
- `coefficients::Vector{Float64}`: Coefficient vector

# Returns
- `Vector{Float64}`: f₀ evaluated at all sample points (n_samples,)
"""
function evaluate_f₀(precomp::PrecomputedBasis, coefficients::Vector{Float64})
    @assert length(coefficients) == precomp.n_basis "Coefficients length must match number of basis functions"
    return precomp.Ψ₀ * coefficients
end

"""
    evaluate_integral(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)

Evaluate the integral ∫₀^{z_k} g(∂f/∂x_k) dx_k for all samples using precomputed
quadrature node evaluations.

For each sample, this computes:
∫₀^{z_k} g(Σⱼ cⱼ ∂ψⱼ/∂x_k) dx_k

using Gauss-Legendre quadrature with precomputed basis derivative values.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations at quadrature nodes
- `coefficients::Vector{Float64}`: Coefficient vector
- `rectifier::AbstractRectifierFunction`: Rectifier function g

# Returns
- `Vector{Float64}`: Integral evaluated at all sample points (n_samples,)
"""
function evaluate_integral(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)
    @assert length(coefficients) == precomp.n_basis "Coefficients length must match number of basis functions"

    n_samples = precomp.n_samples
    n_quad = precomp.n_quad
    integrals = Vector{Float64}(undef, n_samples)

    # For each sample, compute the integral using quadrature
    @inbounds for i in 1:n_samples
        integral_val = 0.0
        scale = precomp.quad_scales[i]

        # Sum over quadrature points
        for q in 1:n_quad
            # Vectorized computation of ∂f/∂x_k at this quadrature point
            ∂f = dot(view(precomp.∂Ψ_quad, i, q, :), coefficients)

            # Apply rectifier and add weighted contribution
            integral_val += precomp.quad_weights[q] * rectifier(∂f)
        end

        # Scale by integration domain (0.5 * z_k for mapping from [-1,1] to [0, z_k])
        integrals[i] = integral_val * scale
    end

    return integrals
end

"""
    evaluate_M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)

Evaluate the full map component Mᵏ(z) = f₀ + ∫₀^{z_k} g(∂f/∂x_k) dx_k for all samples.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations
- `coefficients::Vector{Float64}`: Coefficient vector
- `rectifier::AbstractRectifierFunction`: Rectifier function g

# Returns
- `Vector{Float64}`: Mᵏ evaluated at all sample points (n_samples,)
"""
function evaluate_M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)
    f₀ = evaluate_f₀(precomp, coefficients)
    integral = evaluate_integral(precomp, coefficients, rectifier)
    return f₀ .+ integral
end

"""
    evaluate_∂M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)

Evaluate ∂Mᵏ/∂zₖ = g(∂f/∂zₖ) for all samples at their actual z_k values.

# Arguments
- `precomp::PrecomputedBasis`: Precomputed basis evaluations
- `coefficients::Vector{Float64}`: Coefficient vector
- `rectifier::AbstractRectifierFunction`: Rectifier function g

# Returns
- `Vector{Float64}`: ∂Mᵏ/∂zₖ evaluated at all sample points (n_samples,)
"""
function evaluate_∂M(precomp::PrecomputedBasis, coefficients::Vector{Float64}, rectifier::AbstractRectifierFunction)
    @assert length(coefficients) == precomp.n_basis "Coefficients length must match number of basis functions"

    # Compute ∂f/∂zₖ at each sample point z using precomputed derivatives
    ∂f_vals = precomp.∂Ψ_z * coefficients

    # Apply rectifier
    return rectifier.(∂f_vals)
end

# Display methods
function Base.show(io::IO, pb::PrecomputedBasis)
    print(io, "PrecomputedBasis($(pb.n_samples) samples, $(pb.n_basis) basis functions, $(pb.n_quad) quad points)")
end

function Base.show(io::IO, ::MIME"text/plain", pb::PrecomputedBasis)
    println(io, "PrecomputedBasis:")
    println(io, "  Number of samples: $(pb.n_samples)")
    println(io, "  Number of basis functions: $(pb.n_basis)")
    println(io, "  Number of quadrature points: $(pb.n_quad)")

    # Memory usage estimate
    memory_mb = (sizeof(pb.Ψ₀) + sizeof(pb.∂Ψ_z) + sizeof(pb.Ψ_quad) + sizeof(pb.∂Ψ_quad) +
                 sizeof(pb.quad_weights) + sizeof(pb.quad_scales)) / (1024^2)
    println(io, "  Memory usage: $(round(memory_mb, digits=2)) MB")
end

"""
    PrecomputedMapBasis

Stores precomputed basis evaluations for a full PolynomialMap at quadrature points.

Used for density-based optimization where the same quadrature points are used throughout
optimization with different coefficient values.

# Fields
- `component_data::Vector{PrecomputedBasis}`: Precomputed data for each component
- `quad_points::Matrix{Float64}`: Quadrature points (n_quad × dimension)
- `quad_weights::Vector{Float64}`: Quadrature weights
- `n_quad::Int`: Number of quadrature points
- `dimension::Int`: Number of map components/dimensions
"""
struct PrecomputedMapBasis
    component_data::Vector{PrecomputedBasis}
    quad_points::Matrix{Float64}
    quad_weights::Vector{Float64}
    n_quad::Int
    dimension::Int

    function PrecomputedMapBasis(
        component_data::Vector{PrecomputedBasis},
        quad_points::Matrix{Float64},
        quad_weights::Vector{Float64}
    )
        n_quad = length(quad_weights)
        dimension = length(component_data)

        @assert size(quad_points, 1) == n_quad "quad_points rows must match n_quad"
        @assert size(quad_points, 2) == dimension "quad_points columns must match dimension"

        new(component_data, quad_points, quad_weights, n_quad, dimension)
    end
end

"""
    PrecomputedMapBasis(M::PolynomialMap, quad_points::Matrix{Float64}, quad_weights::Vector{Float64}; n_quad::Int=64)

Precompute basis evaluations for all components of a PolynomialMap at given quadrature points.

# Arguments
- `M::PolynomialMap`: The polynomial map
- `quad_points::Matrix{Float64}`: Quadrature points (n_quad × dimension)
- `quad_weights::Vector{Float64}`: Quadrature weights (n_quad,)
- `n_quad::Int=64`: Number of internal quadrature points for component integrals

# Returns
- `PrecomputedMapBasis`: Structure containing precomputed evaluations for all components
"""
function PrecomputedMapBasis(
    M::PolynomialMap,
    quad_points::Matrix{Float64},
    quad_weights::Vector{Float64};
    n_quad::Int=64
)
    dimension = numberdimensions(M)
    n_quad_pts = size(quad_points, 1)

    @assert size(quad_points, 2) == dimension "Quadrature points must match map dimension"
    @assert length(quad_weights) == n_quad_pts "Weights must match number of quadrature points"

    # Precompute basis for each component
    component_data = Vector{PrecomputedBasis}(undef, dimension)

    for k in 1:dimension
        component = M[k]
        # Extract relevant columns (z₁, ..., zₖ) from quadrature points
        samples_k = quad_points[:, 1:k]
        component_data[k] = PrecomputedBasis(component, samples_k, n_quad=n_quad)
    end

    return PrecomputedMapBasis(component_data, quad_points, quad_weights)
end

# Display methods
function Base.show(io::IO, pmb::PrecomputedMapBasis)
    print(io, "PrecomputedMapBasis($(pmb.n_quad) quadrature points, $(pmb.dimension) dimensions)")
end

function Base.show(io::IO, ::MIME"text/plain", pmb::PrecomputedMapBasis)
    println(io, "PrecomputedMapBasis:")
    println(io, "  Number of quadrature points: $(pmb.n_quad)")
    println(io, "  Map dimensions: $(pmb.dimension)")

    # Memory usage estimate
    total_memory = sizeof(pmb.quad_points) + sizeof(pmb.quad_weights)
    for comp_data in pmb.component_data
        total_memory += (sizeof(comp_data.Ψ₀) + sizeof(comp_data.∂Ψ_z) +
                        sizeof(comp_data.Ψ_quad) + sizeof(comp_data.∂Ψ_quad) +
                        sizeof(comp_data.quad_weights) + sizeof(comp_data.quad_scales))
    end
    memory_mb = total_memory / (1024^2)
    println(io, "  Memory usage: $(round(memory_mb, digits=2)) MB")
end

"""
    evaluate_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)

Evaluate the polynomial map at a specific quadrature point using precomputed basis.

# Arguments
- `M::PolynomialMap`: The polynomial map with current coefficients
- `precomp::PrecomputedMapBasis`: Precomputed basis evaluations
- `quad_idx::Int`: Index of the quadrature point to evaluate at

# Returns
- `Vector{Float64}`: Map evaluation [M¹(z), M²(z), ..., Mᵈ(z)]
"""
function evaluate_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)
    dimension = precomp.dimension
    result = Vector{Float64}(undef, dimension)

    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]

        # Get coefficients for this component
        c = component.coefficients

        # Evaluate Mᵏ using precomputed basis
        # M^k(z) = f₀ + ∫₀^{z_k} g(∂f/∂x_k) dx_k

        # f₀ term: vectorized dot product
        f₀ = dot(view(comp_precomp.Ψ₀, quad_idx, :), c)

        # Integral term using precomputed quadrature evaluations
        scale = comp_precomp.quad_scales[quad_idx]
        integral_val = 0.0
        for q in 1:comp_precomp.n_quad
            # Vectorized computation of ∂f at quadrature point q
            ∂f = dot(view(comp_precomp.∂Ψ_quad, quad_idx, q, :), c)
            integral_val += comp_precomp.quad_weights[q] * component.rectifier(∂f)
        end
        integral_val *= scale

        result[k] = f₀ + integral_val
    end

    return result
end

"""
    gradient_coefficients_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)

Compute the gradient of the map w.r.t. all coefficients at a quadrature point.

# Arguments
- `M::PolynomialMap`: The polynomial map with current coefficients
- `precomp::PrecomputedMapBasis`: Precomputed basis evaluations
- `quad_idx::Int`: Index of the quadrature point

# Returns
- `Matrix{Float64}`: Gradient matrix (dimension × total_coefficients)
"""
function gradient_coefficients_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)
    dimension = precomp.dimension
    n_total_coeffs = numbercoefficients(M)

    grad_matrix = zeros(Float64, dimension, n_total_coeffs)

    coeff_offset = 1
    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]
        c = component.coefficients
        n_comp_coeffs = length(c)

        # For each coefficient of this component
        for j in 1:n_comp_coeffs
            # ∂M^k/∂c_j = ∂f₀/∂c_j + ∂(integral)/∂c_j

            # f₀ term contribution
            grad_val = comp_precomp.Ψ₀[quad_idx, j]

            # Integral term contribution
            # ∂(integral)/∂c_j = ∫₀^{z_k} g'(∂f/∂x_k) * ∂²f/(∂x_k ∂c_j) dx_k
            #                  = ∫₀^{z_k} g'(∂f/∂x_k) * ∂ψ_j/∂x_k dx_k
            scale = comp_precomp.quad_scales[quad_idx]
            for q in 1:comp_precomp.n_quad
                # Vectorized computation of ∂f/∂x_k at this quadrature point
                ∂f = dot(view(comp_precomp.∂Ψ_quad, quad_idx, q, :), c)

                g_prime = derivative(component.rectifier, ∂f)
                grad_val += comp_precomp.quad_weights[q] * g_prime * comp_precomp.∂Ψ_quad[quad_idx, q, j] * scale
            end

            grad_matrix[k, coeff_offset + j - 1] = grad_val
        end

        coeff_offset += n_comp_coeffs
    end

    return grad_matrix
end

"""
    jacobian_diagonal_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)

Compute the diagonal of the Jacobian (∂M^k/∂z_k for each k) at a quadrature point.

# Arguments
- `M::PolynomialMap`: The polynomial map with current coefficients
- `precomp::PrecomputedMapBasis`: Precomputed basis evaluations
- `quad_idx::Int`: Index of the quadrature point

# Returns
- `Vector{Float64}`: Diagonal elements [∂M¹/∂z₁, ∂M²/∂z₂, ..., ∂Mᵈ/∂zᵈ]
"""
function jacobian_diagonal_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)
    dimension = precomp.dimension
    diag = Vector{Float64}(undef, dimension)

    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]
        c = component.coefficients

        # ∂M^k/∂z_k = g(∂f/∂z_k)
        # Vectorized computation of ∂f/∂z_k at the quadrature point
        ∂f = dot(view(comp_precomp.∂Ψ_z, quad_idx, :), c)

        diag[k] = component.rectifier(∂f)
    end

    return diag
end

"""
    jacobian_logdet_gradient_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)

Compute the gradient of log|det(J_M)| w.r.t. all coefficients at a quadrature point.

For triangular maps: log|det J_M| = Σᵢ log(∂Mⁱ/∂zᵢ)

# Arguments
- `M::PolynomialMap`: The polynomial map with current coefficients
- `precomp::PrecomputedMapBasis`: Precomputed basis evaluations
- `quad_idx::Int`: Index of the quadrature point

# Returns
- `Vector{Float64}`: Gradient vector w.r.t. all coefficients
"""
function jacobian_logdet_gradient_map(M::PolynomialMap, precomp::PrecomputedMapBasis, quad_idx::Int)
    dimension = precomp.dimension
    n_total_coeffs = numbercoefficients(M)

    gradient = zeros(Float64, n_total_coeffs)

    coeff_offset = 1
    @inbounds for k in 1:dimension
        component = M[k]
        comp_precomp = precomp.component_data[k]
        c = component.coefficients
        n_comp_coeffs = length(c)

        # Vectorized computation of ∂M^k/∂z_k (diagonal element)
        ∂f = dot(view(comp_precomp.∂Ψ_z, quad_idx, :), c)

        diagonal_deriv = component.rectifier(∂f)
        g_prime = derivative(component.rectifier, ∂f)

        # ∂(log|∂M^k/∂z_k|)/∂c_j = (1/∂M^k/∂z_k) * g'(∂f/∂z_k) * ∂ψ_j/∂z_k
        # Vectorized assignment of gradient components
        gradient[coeff_offset:coeff_offset + n_comp_coeffs - 1] .= (g_prime / diagonal_deriv) .* view(comp_precomp.∂Ψ_z, quad_idx, :)

        coeff_offset += n_comp_coeffs
    end

    return gradient
end
