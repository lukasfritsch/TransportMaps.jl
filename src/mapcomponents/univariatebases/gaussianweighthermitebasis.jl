"""
    GaussianWeightedHermiteBasis

Probabilist Hermite polynomial basis with Gaussian weight for edge control.
"""
struct GaussianWeightedHermiteBasis <: AbstractPolynomialBasis end

function _gaussian_weight_hermite(n::Int, z::Real)
    return hermite_polynomial(n, z) * exp(-0.25 * z^2)
end

function _gaussian_weight_hermite_derivative(n::Int, z::Real)
    return n / 2 * _gaussian_weight_hermite(n - 1, z) - 0.5 * _gaussian_weight_hermite(n + 1, z)
end

"""
    basisfunction(basis::GaussianWeightedHermiteBasis, αᵢ::Real, zᵢ::Real)

Evaluate `GaussianWeightedHermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction(basis::GaussianWeightedHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)

    if n <= 1
        return hermite_polynomial(n, zᵢ)
    else
        return _gaussian_weight_hermite(n, zᵢ)
    end
end

"""
    basisfunction_derivative(basis::GaussianWeightedHermiteBasis, αᵢ::Real, zᵢ::Real)

Evaluate derivative of `GaussianWeightedHermiteBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction_derivative(basis::GaussianWeightedHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)

    if n <= 1
        return hermite_derivative(n, zᵢ)
    else
        return _gaussian_weight_hermite_derivative(n, zᵢ)
    end
end

function Base.show(io::IO, ::GaussianWeightedHermiteBasis)
    print(io, "GaussianWeightedHermiteBasis()")
end
