# Todo: Also implement gradient and Hessian
# Todo: Implement map from samples

# Compute the Kullback-Leibler divergence between the polynomial map and a target density
function kldivergence(
    M::PolynomialMap,
    target_density::Function,
    quadrature::AbstractQuadratureWeights,
    )

    total = 0.0
    δ = eps()  # Small value to avoid log(0)

    for (i, zᵢ) in enumerate(eachrow(quadrature.points))
        # Add δ for regularization
        Mᵢ = evaluate(M, zᵢ) .+ δ*zᵢ
        log_π = log(target_density(Mᵢ)+δ)
        # Log determinant
        log_detJ = log(abs(jacobian(M, zᵢ)))

        total += quadrature.weights[i] * (-log_π - log_detJ)
    end

    return total
end

# Optimize the polynomial map to minimize the Kullback-Leibler divergence to a target density
function optimize!(
    M::PolynomialMap,
    target_density::Function,
    quadrature::AbstractQuadratureWeights,
    )

    # Define the objective function
    objective_function(a) = begin
        setcoefficients!(M, a)  # Set the coefficients in the polynomial map
        return kldivergence(M, target_density, quadrature)
    end

    # Optimize the polynomial coefficients
    initial_coefficients = zeros(numbercoefficients(M))
    result = optimize(objective_function, initial_coefficients, LBFGS())

    setcoefficients!(M, result.minimizer)  # Update the polynomial map with the optimized coefficients

    return result
end

# Compute the variance diagnostic for the polynomial map
function variance_diagnostic(
    M::PolynomialMap,
    target_density::Function,
    Z::AbstractArray{<:Real},
)
    @assert size(Z, 2) == numberdimensions(M) "Z must have the same number of columns as number of map components in M"

    # Initialize
    δ = eps()  # Small value to avoid log(0)
    total = zeros(Float64, size(Z, 1))
    mvn = MvNormal(I(numberdimensions(M)))

    for (i, zᵢ) in enumerate(eachrow(Z))
        Mᵢ = evaluate(M, zᵢ) .+ δ*zᵢ
        log_π = log(target_density(Mᵢ) + δ)
        log_detJ = log(abs(jacobian(M, zᵢ)))
        total[i] = log_π + log_detJ - logpdf(mvn, zᵢ)
    end

    return 0.5 * var(total)
end
