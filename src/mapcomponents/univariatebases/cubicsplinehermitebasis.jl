struct CubicSplineHermiteBasis <: AbstractPolynomialBasis
    radius::Float64

    function CubicSplineHermiteBasis(radius::Float64=3.0)
        return new(radius)
    end
end

function CubicSplineHermiteBasis(samples::Vector{<:Real})
    bounds = [quantile(samples, 0.01), quantile(samples, 0.99)]
    r = 2 * maximum(abs.(bounds))
    return CubicSplineHermiteBasis(r)
end

function CubicSplineHermiteBasis(density::Distributions.UnivariateDistribution)
    bounds = [quantile(density, 0.01), quantile(density, 0.99)]
    r = 2 * maximum(abs.(bounds))
    return CubicSplineHermiteBasis(r)
end

function _cubic_weight(z::Float64, r::Float64)
    m = min(1.0, abs(z) / r)
    return 2 * m^3 - 3 * m^2 + 1
end

function _cubic_weight_derivative(z::Float64, r::Float64)
    if abs(z) < r
        m = abs(z) / r
        return (6 / r) * (m^2 - m) * sign(z)
    else
        return 0.0
    end
end

function basisfunction(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    r = basis.radius
    return hermite_polynomial(n, zᵢ) * _cubic_weight(zᵢ, r)
end

function basisfunction_derivative(basis::CubicSplineHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    r = basis.radius
    return hermite_derivative(n, zᵢ) * _cubic_weight(zᵢ, r) + hermite_polynomial(n, zᵢ) * _cubic_weight_derivative(zᵢ, r)
end

function Base.show(io::IO, basis::CubicSplineHermiteBasis)
    print(io, "CubicSplineHermiteBasis(radius=$(basis.radius))")
end
