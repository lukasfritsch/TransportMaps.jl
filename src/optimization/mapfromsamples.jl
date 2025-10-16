
# Objective for optimization of component coefficients from samples
function objective(component::PolynomialMapComponent, samples::Matrix{Float64})

    # Evaluate map component Mk and its partial derivative w.r.t. xk
    Mₖ = evaluate(component, samples)
    ∂M = partial_derivative_zk(component, samples)

    # Monte Carlo estimate of the objective
    obj = sum(.5 * Mₖ .^ 2 - log.(abs.(∂M)))
    return obj

end

# Gradient of objective for optimization of component coefficients from samples
function objective_gradient!(Mk::PolynomialMapComponent, X::Matrix{Float64})
    # Analytical gradient of objective w.r.t. coefficients c of component Mk
    # Objective per sample: J(z) = 0.5 * Mₖ(z)^2 - log|∂Mₖ/∂zₖ(z)|
    # ∂J/∂c = Mₖ * ∂Mₖ/∂c - (1 / (∂Mₖ/∂zₖ)) * ∂(∂Mₖ/∂zₖ)/∂c

    n_coeffs = length(Mk.coefficients)
    grad = zeros(Float64, n_coeffs)

    n_points = size(X, 1)
    for i in 1:n_points
        z = X[i, :]

        # Evaluate scalar map value and its diagonal partial derivative
        M_val = evaluate(Mk, z)
        ∂M = partial_derivative_zk(Mk, z)

        # Gradient of M w.r.t. coefficients: vector length n_coeffs
        ∂M_∂c = gradient_coefficients(Mk, z)

        # Gradient of ∂M/∂zₖ w.r.t. coefficients: vector length n_coeffs
        ∂∂M_∂c = partial_derivative_zk_gradient_coefficients(Mk, z)

        # Accumulate gradient contribution from this sample
        # Handle potential zero ∂M (should be unlikely with rectifier) by adding small eps
        denom = max(abs(∂M), eps()) * sign(∂M)

        grad .+= M_val .* ∂M_∂c .- (1.0 ./ denom) .* ∂∂M_∂c
    end

    return grad
end


"""
    optimize!(M::PolynomialMap, samples::Matrix{Float64};
              optimizer::Optim.AbstractOptimizer = LBFGS(),
              options::Optim.Options = Optim.Options())

Optimize polynomial map coefficients to minimize KL divergence to a target density.

# Arguments
- `M::PolynomialMap`: The polynomial map to optimize.
- `samples::Matrix{Float64}`: A matrix of sample data used to initialize and fit the map. Columns are interpreted as components/dimensions and rows as individual sample points.
- `lm::LinearMap`: A linear map used to standardize the samples before optimization (default: identity map).

# Optional keyword arguments:
- `optimizer::Optim.AbstractOptimizer = LBFGS()`: Optimizer from Optim.jl to use (default: `LBFGS()`).
- `options::Optim.Options = Optim.Options()`: Options passed to the optimizer (default: `Optim.Options()`).

# Returns
- Optimization result from Optim.jl. The optimized coefficients are written back into `M`.
"""
function optimize!(
    M::PolynomialMap,
    samples::Matrix{Float64},
    lm::LinearMap = LinearMap();
    optimizer::Optim.AbstractOptimizer = LBFGS(),
    options::Optim.Options = Optim.Options()
)
    @assert size(samples, 2) == numberdimensions(M) "Samples must have the same number of columns as number of map components in M"
    # Standardize samples using linear map
    samples = evaluate(lm, samples)

    # Initialize map from samples: set map direction and bounds
    initializemapfromsamples!(M, samples)

    # Store optimization results
    results = Vector{Any}(undef, numberdimensions(M))

    # Optimize each component sequentially
    for k in 1:numberdimensions(M)
        component = M[k]
        println("Optimizing component $(k) / $(numberdimensions(M))")
        results[k] = optimize!(component, samples[:, 1:k], optimizer, options)
    end

    return results
end

# Optimize a single map component
function optimize!(
    component::PolynomialMapComponent,
    samples::Matrix{Float64},
    optimizer::Optim.AbstractOptimizer,
    options::Optim.Options
)

    obj_fun = c -> begin
        setcoefficients!(component, c)
        objective(component, samples)
    end

    grad_fun! = (g, c) -> begin
        setcoefficients!(component, c)
        grad = objective_gradient!(component, samples)
        g .= grad
    end

    initial_coefficients = getcoefficients(component)
    result = optimize(obj_fun, grad_fun!, initial_coefficients, optimizer, options)

    # Update map component with optimized coefficients
    setcoefficients!(component, result.minimizer)

    return result
end

# Todo (maybe): cross-validation and standardization of samples to N(0,1)
