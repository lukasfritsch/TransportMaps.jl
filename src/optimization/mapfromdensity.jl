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

# Gradient of KL divergence with respect to map coefficients
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
    optimize!(M::PolynomialMap, target::AbstractMapDensity, quadrature::AbstractQuadratureWeights;
              optimizer::Optim.AbstractOptimizer = LBFGS(),
              options::Optim.Options = Optim.Options())

Optimize polynomial map coefficients to minimize KL divergence to a target density.

# Arguments
- `M::PolynomialMap`: The polynomial map to optimize.
- `target::AbstractMapDensity`: Target map density object (provides the target density π(x) and any needed operations).
- `quadrature::AbstractQuadratureWeights`: Quadrature points and weights used for numerical integration.

# Optional keyword arguments:
- `optimizer::Optim.AbstractOptimizer = LBFGS()`: Optimizer from Optim.jl to use (default: `LBFGS()`).
- `options::Optim.Options = Optim.Options()`: Options passed to the optimizer (default: `Optim.Options()`).

# Returns
- Optimization result from Optim.jl. The optimized coefficients are written back into `M`.
"""
function optimize!(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights;
    optimizer::Optim.AbstractOptimizer = LBFGS(),
    options::Optim.Options = Optim.Options()
    )

    # Define objective function and gradient
    function objective_function(a)
        setcoefficients!(M, a)
        return kldivergence(M, target, quadrature)
    end

    function gradient_function!(grad_storage, a)
        setcoefficients!(M, a)
        grad_storage .= kldivergence_gradient(M, target, quadrature)
    end

    # Optimize with analytical gradient
    initial_coefficients = getcoefficients(M)
    result = optimize(objective_function, gradient_function!, initial_coefficients, optimizer, options)

    setcoefficients!(M, result.minimizer)  # Update the polynomial map with optimized coefficients

    return result
end

# Compute the variance diagnostic for the polynomial map
function variance_diagnostic(
    M::PolynomialMap,
    target::MapTargetDensity,
    Z::AbstractArray{<:Real},
)
    @assert size(Z, 2) == numberdimensions(M) "Z must have the same number of columns as number of map components in M"

    # Initialize
    total = zeros(Float64, size(Z, 1))

    for (i, zᵢ) in enumerate(eachrow(Z))
        total[i] = log(pushforward(M, target, zᵢ)) - log.(M.reference.density(zᵢ))
    end

    return 0.5 * var(total)
end
