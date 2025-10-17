struct GaussianWeightedHermiteBasis <: AbstractPolynomialBasis
    function GaussianWeightedHermiteBasis()
        return new()
    end
end

function _gaussian_weight_hermite(n::Int, z::Float64)
    return hermite_polynomial(n, z) * exp(-0.25 * z^2)
end

function _gaussian_weight_hermite_derivative(n::Int, z::Float64)
    if n == 0
        return -0.5 * z * exp(-0.25 * z^2)
    else
        return n / 2 * _gaussian_weight_hermite(n - 1, z) - 0.5 * _gaussian_weight_hermite(n + 1, z)
    end
end

@inline function basisfunction(basis::GaussianWeightedHermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)

    if n <= 1
        return hermite_polynomial(n, zᵢ)
    else
        return _gaussian_weight_hermite(n, zᵢ)
    end
end

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
