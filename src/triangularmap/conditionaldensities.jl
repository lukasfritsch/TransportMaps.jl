# Conditional densities for triangular maps making use of the Knothe-Rosenblatt transform

# Conditional density: π(xₖ | x₁, ..., xₖ₋₁) for triangular maps (single value)
function conditional_density(M::PolynomialMap, x_range::Float64, x_given::Vector{Float64})
    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    # invert the first k components to get z_range
    z_range = inverse(M, [x_given..., x_range], k)

    # conditional density: π(xₖ | x₁, ..., xₖ₋₁) = ρ(zₖ) * |∂Tₖ/∂zₖ|^{-1}
    cond_density = M.reference.density(z_range[k]) * abs(1 / partial_derivative_zk(M.components[k], z_range))

    return cond_density
end

# For convenience when x_given is a single value
function conditional_density(M::PolynomialMap, x_range::Float64, x_given::Float64)
    return conditional_density(M, x_range, [x_given])
end

function conditional_density(M::PolynomialMap, x_range::Float64, x_given::AbstractArray{<:Real})
    return conditional_density(M, x_range, Vector{Float64}(x_given))
end

# Conditional density: π(xₖ | x₁, ..., xₖ₋₁) for triangular maps (multiple values)
function conditional_density(M::PolynomialMap, x_range::Vector{Float64}, x_given::Vector{Float64})
    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    n_points = length(x_range)
    cond_densities = Vector{Float64}(undef, n_points)

    # Use multithreading to compute conditional densities for each point
    Threads.@threads for i in 1:n_points
        cond_densities[i] = conditional_density(M, x_range[i], x_given)
    end

    return cond_densities
end

function conditional_density(M::PolynomialMap, x_range::AbstractArray{<:Real}, x_given::AbstractArray{<:Real})
    return conditional_density(M, Vector{Float64}(x_range), Vector{Float64}(x_given))
end

function conditional_density(M::PolynomialMap, x_range::AbstractArray{<:Real}, x_given::Float64)
    return conditional_density(M, Vector{Float64}(x_range), [x_given])
end

# Generate samples from the conditional distribution π(xₖ | x₁, ..., xₖ₋₁) by pushing forward z_range ~ ρ(z_range)
function conditional_sample(M::PolynomialMap, x_given::Vector{Float64}, z_range::Float64)

    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    # invert the first k components to get z_given
    z_given = inverse(M, x_given, k-1)
    # Push through the k-th component of the map
    return evaluate(M.components[k], [z_given..., z_range])
end

# For convenience when x_given is a single value
function conditional_sample(M::PolynomialMap, x_given::Float64, z_range::Float64)
    return conditional_sample(M, [x_given], z_range)
end

function conditional_sample(M::PolynomialMap, x_given::AbstractArray{<:Real}, z_range::Float64)
    return conditional_sample(M, Vector{Float64}(x_given), z_range)
end

# Generate samples from the conditional distribution π(xₖ | x₁, ..., xₖ₋₁) by pushing forward z_range ~ ρ(z_range) (multiple values)
function conditional_sample(M::PolynomialMap, x_given::Vector{Float64}, z_range::Vector{Float64})
    k = length(x_given) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    n_points = length(z_range)
    samples = Vector{Float64}(undef, n_points)
    # Use multithreading to compute samples for each point
    Threads.@threads for i in 1:n_points
        samples[i] = conditional_sample(M, x_given, z_range[i])
    end
    return samples
end

# For convenience when x_given is a single value
function conditional_sample(M::PolynomialMap, x_given::Float64, z_range::AbstractArray{<:Real})
    return conditional_sample(M, [x_given], Vector{Float64}(z_range))
end

function conditional_sample(M::PolynomialMap, x_given::AbstractArray{<:Real}, z_range::AbstractArray{<:Real})
    return conditional_sample(M, Vector{Float64}(x_given), Vector{Float64}(z_range))
end

# Multivariate conditional density: π(xⱼ, xⱼ₊₁, ..., xₖ | x₁, ..., xⱼ₋₁)
# Computed as the product: ∏ᵢ₌ⱼᵏ π(xᵢ | x₁, ..., xᵢ₋₁)
function multivariate_conditional_density(M::PolynomialMap, x::Vector{Float64})
    d = length(x)
    @assert d <= numberdimensions(M) "Length of x cannot exceed the dimension of the map"

    if d == 1
        # For single dimension, this is just the marginal density of x₁
        z = inverse(M, x, 1)
        return M.reference.density(z[1]) * abs(1 / partial_derivative_zk(M.components[1], z))
    end

    # Start with the first variable (marginal density)
    z₁ = inverse(M, x[1:1], 1)
    density = M.reference.density(z₁[1]) * abs(1 / partial_derivative_zk(M.components[1], z₁))

    # Multiply by conditional densities for subsequent variables
    for k in 2:d
        x_given_k = x[1:k-1]
        x_range_k = x[k]
        density *= conditional_density(M, x_range_k, x_given_k)
    end

    return density
end

# Multivariate conditional density for a specific range: π(xⱼ, ..., xₖ | x₁, ..., xⱼ₋₁)
function multivariate_conditional_density(M::PolynomialMap, x_range::Vector{Float64}, x_given::Vector{Float64})
    j = length(x_given) + 1  # Starting index for the range
    k = j + length(x_range) - 1  # Ending index for the range

    @assert j >= 1 "Starting index must be >= 1"
    @assert k <= numberdimensions(M) "Ending index cannot exceed the dimension of the map"
    @assert length(x_range) >= 1 "x_range must have at least one element"

    # Construct the full vector [x_given..., x_range...]
    x_full = [x_given..., x_range...]

    # Start with the conditional density of the first variable in the range
    density = conditional_density(M, x_range[1], x_given)

    # Multiply by conditional densities for subsequent variables in the range
    for i in 2:length(x_range)
        x_given_i = x_full[1:j+i-2]  # x₁, ..., xⱼ₊ᵢ₋₂
        x_range_i = x_range[i]
        density *= conditional_density(M, x_range_i, x_given_i)
    end

    return density
end

# Convenience function for single given value
function multivariate_conditional_density(M::PolynomialMap, x_range::Vector{Float64}, x_given::Float64)
    return multivariate_conditional_density(M, x_range, [x_given])
end

# Convenience function for AbstractArray inputs
function multivariate_conditional_density(M::PolynomialMap, x_range::AbstractArray{<:Real}, x_given::AbstractArray{<:Real})
    return multivariate_conditional_density(M, Vector{Float64}(x_range), Vector{Float64}(x_given))
end

function multivariate_conditional_density(M::PolynomialMap, x_range::AbstractArray{<:Real}, x_given::Float64)
    return multivariate_conditional_density(M, Vector{Float64}(x_range), [x_given])
end

function multivariate_conditional_density(M::PolynomialMap, x::AbstractArray{<:Real})
    return multivariate_conditional_density(M, Vector{Float64}(x))
end

# Multivariate conditional sampling: Generate samples from π(xⱼ, ..., xₖ | x₁, ..., xⱼ₋₁)
function multivariate_conditional_sample(M::PolynomialMap, x_given::Vector{Float64}, z_range::Vector{Float64})
    j = length(x_given) + 1  # Starting index for sampling
    k = j + length(z_range) - 1  # Ending index for sampling

    @assert j >= 1 "Starting index must be >= 1"
    @assert k <= numberdimensions(M) "Ending index cannot exceed the dimension of the map"
    @assert length(z_range) >= 1 "z_range must have at least one element"

    x_samples = Vector{Float64}(undef, length(z_range))
    x_current = copy(x_given)

    # Generate samples sequentially using conditional sampling
    for i in 1:length(z_range)
        x_i = conditional_sample(M, x_current, z_range[i])
        x_samples[i] = x_i
        push!(x_current, x_i)  # Add to conditioning variables for next iteration
    end

    return x_samples
end

# Convenience function for single given value
function multivariate_conditional_sample(M::PolynomialMap, x_given::Float64, z_range::Vector{Float64})
    return multivariate_conditional_sample(M, [x_given], z_range)
end

# Convenience function for AbstractArray inputs
function multivariate_conditional_sample(M::PolynomialMap, x_given::AbstractArray{<:Real}, z_range::AbstractArray{<:Real})
    return multivariate_conditional_sample(M, Vector{Float64}(x_given), Vector{Float64}(z_range))
end

function multivariate_conditional_sample(M::PolynomialMap, x_given::Float64, z_range::AbstractArray{<:Real})
    return multivariate_conditional_sample(M, [x_given], Vector{Float64}(z_range))
end
