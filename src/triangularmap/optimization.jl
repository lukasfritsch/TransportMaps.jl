# Todo: Also implement Hessian
# Todo: Implement map from samples

"""
    kldivergence(M::PolynomialMap, target_density::Function, quadrature::AbstractQuadratureWeights)

Compute the Kullback-Leibler divergence between the polynomial map and a target density.

"""
function kldivergence(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights,
    )

    total = 0.0
    δ = eps()  # Small value to avoid log(0)

    for (i, zᵢ) in enumerate(eachrow(quadrature.points))
        # Add δ for regularization
        Mᵢ = evaluate(M, zᵢ) .+ δ*zᵢ
        log_π = log(target.density(Mᵢ)+δ)
        # Log determinant
        log_detJ = log(abs(jacobian(M, zᵢ)))

        total += quadrature.weights[i] * (-log_π - log_detJ)
    end

    return total
end

"""
    kldivergence_gradient(M::PolynomialMap, target::MapTargetDensity, quadrature::AbstractQuadratureWeights)

Compute the gradient of the KL divergence with respect to polynomial map coefficients.

For KL divergence KL = ∫ w(z) [-log π(M(z)) - log |det J_M(z)|] dz,
the gradient is: ∂KL/∂c = ∫ w(z) [-∇π(M(z))/π(M(z)) · ∂M/∂c - ∂log|det J_M|/∂c] dz

# Arguments
- `M::PolynomialMap`: The polynomial map
- `target::MapTargetDensity`: Target density object
- `quadrature::AbstractQuadratureWeights`: Quadrature points and weights

# Returns
- `Vector{Float64}`: Gradient vector with respect to all coefficients
"""
function kldivergence_gradient(
        M::PolynomialMap,
        target::AbstractMapDensity,
        quadrature::AbstractQuadratureWeights,
    )

    n_coeffs = numbercoefficients(M)
    gradient_total = zeros(Float64, n_coeffs)

    for (i, zᵢ) in enumerate(eachrow(quadrature.points))
        # Evaluate map
        Mᵢ = evaluate(M, zᵢ)

        # Evaluate gradient of target density w.r.t. x
        ∇π = gradient(target, Mᵢ)

        π_val = max(target.density(Mᵢ), 1e-12)

        # Compute gradient of map with respect to coefficients
        ∂M_∂c = gradient_coefficients(M, zᵢ)  # Shape: (n_dims, n_coeffs)

        # First term: -(∇π(M(z))/π(M(z))) · ∂M/∂c from ∂(-log π)/∂c
        weight_factor = -quadrature.weights[i] / π_val

        for j in 1:n_coeffs
            for k in axes(∇π, 1)  # Iterate over dimensions
                gradient_total[j] += weight_factor * ∇π[k] * ∂M_∂c[k, j]
            end
        end

        # Second term: ∂(-log|det J_M|)/∂c
        jacobian_contrib = jacobian_logdet_gradient(M, zᵢ)
        for j in 1:n_coeffs
            gradient_total[j] -= quadrature.weights[i] * jacobian_contrib[j]
        end
    end

    return gradient_total
end

"""
    optimize!(M::PolynomialMap, target_density::Function, quadrature::AbstractQuadratureWeights; use_gradient=true)

Optimize polynomial map coefficients to minimize KL divergence to target density.

# Arguments
- `M::PolynomialMap`: The polynomial map to optimize
- `target_density::Function`: Target density function π(x)
- `quadrature::AbstractQuadratureWeights`: Quadrature points and weights
- `use_gradient::Bool=true`: Whether to use analytical gradient (much faster)

# Returns
- Optimization result from Optim.jl
"""
function optimize!(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights,
    )

    # Define objective function and gradient
    function objective_with_gradient(a)
        setcoefficients!(M, a)
        return kldivergence(M, target, quadrature)
    end

    function gradient_function!(grad_storage, a)
        setcoefficients!(M, a)
        grad_storage .= kldivergence_gradient(M, target, quadrature)
    end

    # Initialize coefficients: all zeros
    initial_coefficients = zeros(numbercoefficients(M))

    # Optimize with analytical gradient
    result = optimize(objective_with_gradient, gradient_function!, initial_coefficients, LBFGS())

    setcoefficients!(M, result.minimizer)  # Update the polynomial map with optimized coefficients

    return result
end

function optimize!(M::PolynomialMap, samples::AbstractArray{<:Real})

    setforwarddirection!(M, :reference)

    # Create quadrature weights based on the number of dimensions
    quadrature = MonteCarloWeights(samples)
    target = M.reference
    # Optimize the polynomial map
    return optimize!(M, target, quadrature)
end

# Compute the variance diagnostic for the polynomial map
function variance_diagnostic(
    M::PolynomialMap,
    target::MapTargetDensity,
    Z::AbstractArray{<:Real},
)
    @assert size(Z, 2) == numberdimensions(M) "Z must have the same number of columns as number of map components in M"

    # Initialize
    δ = eps()  # Small value to avoid log(0)
    total = zeros(Float64, size(Z, 1))
    mvn = M.reference.density

    for (i, zᵢ) in enumerate(eachrow(Z))
        Mᵢ = evaluate(M, zᵢ) .+ δ*zᵢ
        log_π = log(target.density(Mᵢ) + δ)
        log_detJ = log(abs(jacobian(M, zᵢ)))
        total[i] = log_π + log_detJ - log.(mvn(zᵢ))
    end

    return 0.5 * var(total)
end
