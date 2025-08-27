struct CubicSplineHermiteBasis <: AbstractPolynomialBasis
    radius::Float64

    function CubicSplineHermiteBasis(; radius::Float64=3.0)
        return new(radius)
    end
end

function CubicSplineHermiteBasis(samples::Vector{<:Real})
    bounds = [quantile(samples, 0.01), quantile(samples, 0.99)]
    r = 2 * maximum(abs.(bounds))
    return CubicSplineHermiteBasis(radius=r)
end

function CubicSplineHermiteBasis(density::Distributions.UnivariateDistribution)
    bounds = [quantile(density, 0.01), quantile(density, 0.99)]
    r = 2 * maximum(abs.(bounds))
    return CubicSplineHermiteBasis(radius=r)
end

@inline function cubic_weight(z::Real, r::Float64)
    m = min(1.0, abs(z) / r)
    return 2 * m^3 - 3 * m^2 + 1
end

@inline function cubic_weight_derivative(z::Real, r::Float64)
    if abs(z) < r
        m = abs(z) / r
        return (6 / r) * (m^2 - m) * sign(z)
    else
        return 0.0
    end
end

@inline function basisfunction(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    return hermite_polynomial(n, zᵢ) * cubic_weight(zᵢ, basis.radius)
end

@inline function basisfunction_derivative(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    return hermite_derivative(n, zᵢ) * cubic_weight(zᵢ, basis.radius) + hermite_polynomial(n, zᵢ) * cubic_weight_derivative(zᵢ, basis.radius)
end

function Base.show(io::IO, basis::CubicSplineHermiteBasis)
    print(io, "CubicSplineHermiteBasis(radius=$(basis.radius))")
end
