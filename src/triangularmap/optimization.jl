# Todo: Make it function of the polynomial coefficients
# Todo: Also implement gradient and Hessian

function objective(
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
    objective_function = (a) -> objective(M, quadrature, target_density)

    # Optimize the polynomial coefficients


end
