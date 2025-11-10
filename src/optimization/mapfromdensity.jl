"""
    kldivergence(M::PolynomialMap, target_density::Function, quadrature::AbstractQuadratureWeights)

Compute the Kullback-Leibler divergence between the polynomial map and a target density.

"""
function kldivergence(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights,
)
    # Small regularization term
    δ = 1e-9

    # Evaluate map and add small δ for regularization
    Mᵢ = evaluate(M, quadrature.points) + δ * quadrature.points
    # Evaluate target logdensity
    log_π = logdensity(target, Mᵢ)
    # Evaluate log determinant of Jacobian
    log_detJ = log.(abs.(jacobian(M, quadrature.points)))

    return sum(quadrature.weights .* (-log_π .- log_detJ))
end

#! Finally, make one function of kldivergence and gradient that uses both
#! also, use precomputed basis here

# Gradient of KL divergence with respect to map coefficients
function kldivergence_gradient(
    M::PolynomialMap,
    target::AbstractMapDensity,
    quadrature::AbstractQuadratureWeights,
)

    n_coeffs = numbercoefficients(M)
    gradient_total = zeros(Float64, n_coeffs)

    #! vectorize this later!
    for (i, zᵢ) in enumerate(eachrow(quadrature.points))
        # Evaluate map
        Mᵢ = evaluate(M, zᵢ)

        # Evaluate gradient of target density w.r.t. x
        ∇π = gradient_log(target, Mᵢ)

        # Compute gradient of map with respect to coefficients
        ∂M_∂c = gradient_coefficients(M, zᵢ)

        # First term: -∇[log π(M(z))]· ∂M/∂c from ∂(-log π)/∂c
        for j in 1:n_coeffs
            for k in axes(∇π, 1)  # Iterate over dimensions
                gradient_total[j] += -quadrature.weights[i] * ∇π[k] * ∂M_∂c[k, j]
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

# KL divergence using precomputed basis
function kldivergence(
    M::PolynomialMap,
    target::AbstractMapDensity,
    precomp::PrecomputedMapBasis
)
    total = 0.0
    δ = 1e-9  # Small value to avoid log(0)

    #! Vectorize this
    for i in 1:precomp.n_quad
        # Evaluate map using precomputed basis
        Mᵢ = evaluate_map(M, precomp, i)
        Mᵢ .+= δ .* precomp.quad_points[i, :]

        log_π = logdensity(target, Mᵢ)

        # Jacobian determinant (product of diagonal for triangular map)
        diag = jacobian_diagonal_map(M, precomp, i)
        log_detJ = sum(log.(abs.(diag)))

        total += precomp.quad_weights[i] * (-log_π - log_detJ)
    end

    return total
end

# Gradient of KL divergence using precomputed basis
function kldivergence_gradient(
    M::PolynomialMap,
    target::AbstractMapDensity,
    precomp::PrecomputedMapBasis
)
    n_coeffs = numbercoefficients(M)
    gradient_total = zeros(Float64, n_coeffs)

    for i in 1:precomp.n_quad
        # Evaluate map using precomputed basis
        Mᵢ = evaluate_map(M, precomp, i)

        # Evaluate gradient of target density w.r.t. x
        ∇π = gradient_log(target, Mᵢ)

        # Compute gradient of map with respect to coefficients using precomputed basis
        ∂M_∂c = gradient_coefficients_map(M, precomp, i)  # Shape: (n_dims, n_coeffs)

        # First term: -∇[log π(M(z))]· ∂M/∂c from ∂(-log π)/∂c

        for j in 1:n_coeffs
            for k in axes(∇π, 1)  # Iterate over dimensions
                gradient_total[j] += -precomp.quad_weights[i] * ∇π[k] * ∂M_∂c[k, j]
            end
        end

        # Second term: ∂(-log|det J_M|)/∂c using precomputed basis
        jacobian_contrib = jacobian_logdet_gradient_map(M, precomp, i)
        for j in 1:n_coeffs
            gradient_total[j] -= precomp.quad_weights[i] * jacobian_contrib[j]
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
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options()
)
    # Precompute basis evaluations at quadrature points
    precomp = PrecomputedMapBasis(M, quadrature.points, quadrature.weights)

    # Call the optimized version
    return optimize!(M, target, precomp, optimizer=optimizer, options=options)
end

# Optimized version using precomputed basis
function optimize!(
    M::PolynomialMap,
    target::AbstractMapDensity,
    precomp::PrecomputedMapBasis;
    optimizer::Optim.AbstractOptimizer=LBFGS(),
    options::Optim.Options=Optim.Options()
)

    # Define objective function and gradient
    function objective_function(a)
        setcoefficients!(M, a)
        return kldivergence(M, target, precomp)
    end

    function gradient_function!(grad_storage, a)
        setcoefficients!(M, a)
        grad_storage .= kldivergence_gradient(M, target, precomp)
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
