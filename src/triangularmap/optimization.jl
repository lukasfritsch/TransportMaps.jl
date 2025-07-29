# Todo: Make it function of the polynomial coefficients
# Todo: Also implement gradient and Hessian

# Compute the Kullback-Leibler divergence between the polynomial map and a target density
function kldivergence(
    M::PolynomialMap,
    target_density::Function,
    quadrature::AbstractQuadratureWeights,
    )

    total = 0.0

    for (i, xᵢ) in enumerate(eachrow(quadrature.points))
        Mᵢ = evaluate(M, xᵢ)
        log_π = log(target_density(Mᵢ))
        log_detJ = log(abs(jacobian(M, xᵢ)))

        total += quadrature.weights[i] * (-log_π - log_detJ)
    end

    return total
end

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
    initial_coefficients = rand(numbercoefficients(M))
    result = optimize(objective_function, initial_coefficients, SimulatedAnnealing())

    return result

end
