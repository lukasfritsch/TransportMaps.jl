struct GaussianWeightedHermiteBasis <: AbstractPolynomialBasis
    function GaussianWeightedHermiteBasis()
        return new()
    end
end

@inline function basisfunction(basis::GaussianWeightedHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    weight = exp(-0.25 * zᵢ^2)
    return hermite_polynomial(n, zᵢ) * weight
end

@inline function basisfunction_derivative(basis::GaussianWeightedHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    return exp(-0.25 * zᵢ^2) * (n * hermite_polynomial(n - 1, zᵢ) - zᵢ / 2 * hermite_polynomial(n, zᵢ))
end

function Base.show(io::IO, ::GaussianWeightedHermiteBasis)
    print(io, "GaussianWeightedHermiteBasis()")
end
