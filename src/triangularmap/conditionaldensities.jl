#! This is a draft implementation

# Case 1: p(xₖ | x₁, ..., xₖ₋₁)
function conditional_density(M::PolynomialMap, x::AbstractVector{<:Real}, k::Int)
    @assert 1 <= k <= M.dim "k must be between 1 and the dimension of the map"

    # Compute the pullback density at x
    pullback_density = pullback(M, x)

    # Compute the marginal density for x₁, ..., xₖ₋₁
    marginal_density = pullback(M, x[1:k-1])

    # Return the conditional density
    return pullback_density / marginal_density
end

# Case 2: p(xₘ | x₁, ..., xₘ₋₁, xₘ₊₁, ..., xₖ)
function conditional_density_general(M::PolynomialMap, x::AbstractVector{<:Real}, m::Int, k::Int)
    @assert 1 <= m <= k <= M.dim "m and k must be within the dimension of the map"

    # Joint density: p(x₁, ..., xₖ)
    joint_density = pullback(M, x[1:k])

    # Marginal density: ∫ p(x₁, ..., xₘ₋₁, xₘ₊₁, ..., xₖ) dxₘ
    function marginal_integrand(xₘ)
        x_copy = copy(x)
        x_copy[m] = xₘ
        return pullback(M, x_copy)
    end

    marginal_density = gaussquadrature(marginal_integrand, 100, -100., 100.)

    # Return the conditional density
    return joint_density / marginal_density
end
