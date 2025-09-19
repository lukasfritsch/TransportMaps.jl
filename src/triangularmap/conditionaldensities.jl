# Conditional densities for triangular maps making use of the Knothe-Rosenblatt transform

# Conditional density: π(xₖ | x₁, ..., xₖ₋₁) for triangular maps (single value)
function conditional_density(M::PolynomialMap, xₖ::Float64, xₖ₋₁::AbstractVector{<:Real})
    k = length(xₖ₋₁) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    # invert the first k components to get zₖ
    zₖ = inverse(M, [xₖ₋₁..., xₖ], k)

    # conditional density: π(xₖ | x₁, ..., xₖ₋₁) = ρ(zₖ) * |∂Tₖ/∂zₖ|^{-1}
    cond_density = M.reference.density(zₖ[k]) * abs(1 / partial_derivative_zk(M.components[k], zₖ))

    return cond_density
end

# For convenience when xₖ₋₁ is a single value
function conditional_density(M::PolynomialMap, xₖ::Float64, xₖ₋₁::Float64)
    return conditional_density(M, xₖ, [xₖ₋₁])
end

# Conditional density: π(xₖ | x₁, ..., xₖ₋₁) for triangular maps (multiple values)
function conditional_density(M::PolynomialMap, xₖ::AbstractVector{<:Real}, xₖ₋₁::AbstractVector{<:Real})
    k = length(xₖ₋₁) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    n_points = length(xₖ)
    cond_densities = Vector{Float64}(undef, n_points)

    # Use multithreading to compute conditional densities for each point
    Threads.@threads for i in 1:n_points
        cond_densities[i] = conditional_density(M, xₖ[i], xₖ₋₁)
    end

    return cond_densities
end

# For convenience when xₖ₋₁ is a single value
function conditional_density(M::PolynomialMap, xₖ::AbstractVector{<:Real}, xₖ₋₁::Float64)
    return conditional_density(M, xₖ, [xₖ₋₁])
end

# Generate samples from the conditional distribution π(xₖ | x₁, ..., xₖ₋₁) by pushing forward zₖ ~ ρ(zₖ)
function conditional_sample(M::PolynomialMap, xₖ₋₁::AbstractVector{<:Real}, zₖ::Float64)

    k = length(xₖ₋₁) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    # invert the first k components to get zₖ
    zₖ₋₁ = inverse(M, xₖ₋₁, k-1)
    # Push through the k-th component of the map
    return evaluate(M.components[k], [zₖ₋₁..., zₖ])
end

# For convenience when xₖ₋₁ is a single value
function conditional_sample(M::PolynomialMap, xₖ₋₁::Float64, zₖ::Float64)
    return conditional_sample(M, [xₖ₋₁], zₖ)
end

# Generate samples from the conditional distribution π(xₖ | x₁, ..., xₖ₋₁) by pushing forward zₖ ~ ρ(zₖ) (multiple values)
function conditional_sample(M::PolynomialMap, xₖ₋₁::AbstractVector{<:Real}, zₖ::AbstractVector{<:Real})
    k = length(xₖ₋₁) + 1
    @assert 1 <= k <= numberdimensions(M) "k must be between 1 and the dimension of the map"

    n_points = length(zₖ)
    samples = Vector{Float64}(undef, n_points)
    # Use multithreading to compute samples for each point
    Threads.@threads for i in 1:n_points
        samples[i] = conditional_sample(M, xₖ₋₁, zₖ[i])
    end
    return samples
end

# For convenience when xₖ₋₁ is a single value
function conditional_sample(M::PolynomialMap, xₖ₋₁::Float64, zₖ::AbstractVector{<:Real})
    return conditional_sample(M, [xₖ₋₁], zₖ)
end
