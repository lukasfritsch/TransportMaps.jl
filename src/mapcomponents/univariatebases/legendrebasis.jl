"""
    LegendreBasis

Legendre polynomial basis, orthogonal with respect to uniform measure on [-1, 1].
"""
struct LegendreBasis <: AbstractLegendreBasis end

# Univariate Legendre polynomials using recurrence relation
@inline function legendre_polynomial(n::Int64, x::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return x
    else
        P_nm2 = 1.0
        P_nm1 = x
        for k in 1:(n-1)
            P_n = ((2k + 1) * x * P_nm1 - k * P_nm2) / (k + 1)
            P_nm2, P_nm1 = P_nm1, P_n
        end
        return P_nm1
    end
end

# Derivative of univariate Legendre polynomial
# Using relation: (2n+1)P_n(x) = P'_{n+1}(x) - P'_{n-1}(x)
@inline function legendre_derivative(n::Int64, x::Real)
    if n == 0
        return 0.0
    elseif n == 1
        return 1.0
    else
        # For numerical stability, use recurrence for derivatives:
        # (1-x²)P'_n = n(P_{n-1} - xP_n)
        P_n = legendre_polynomial(n, x)
        P_nm1 = legendre_polynomial(n - 1, x)

        # Handle near-singular points
        if abs(1 - x^2) < 1e-10
            # Use alternative formula: P'_n = n*P_{n-1} + x*P'_{n-1}
            return n * P_nm1 + x * legendre_derivative(n - 1, x)
        else
            return n * (P_nm1 - x * P_n) / (1 - x^2)
        end
    end
end

"""
    basisfunction(basis::LegendreBasis, αᵢ::Int, zᵢ::Real)

Evaluate `LegendreBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction(basis::LegendreBasis, αᵢ::Int, zᵢ::Real)
    return legendre_polynomial(Int(αᵢ), zᵢ)
end

"""
    basisfunction_derivative(basis::LegendreBasis, αᵢ::Int, zᵢ::Real)

Evaluate derivative of `LegendreBasis` with degree `αᵢ` at `zᵢ`.
"""
@inline function basisfunction_derivative(basis::LegendreBasis, αᵢ::Int, zᵢ::Real)
    return legendre_derivative(Int(αᵢ), zᵢ)
end

function Base.show(io::IO, ::LegendreBasis)
    print(io, "LegendreBasis()")
end

"""
    ShiftedLegendreBasis

Shifted Legendre polynomial basis, orthogonal with respect to uniform measure on [0, 1].
The shifted Legendre polynomials ``P_n^*([0,1])(x)`` are obtained by transforming the standard
Legendre polynomials: ``P_n^*(x) = P_n(2x - 1)``.
"""
struct ShiftedLegendreBasis <: AbstractLegendreBasis end

# Shifted Legendre polynomials on [0, 1]
# P_n^*([0,1])(x) = P_n(2x - 1)
@inline function shifted_legendre_polynomial(n::Int64, x::Real)
    # Transform x ∈ [0,1] to ξ ∈ [-1,1]
    ξ = 2x - 1
    return legendre_polynomial(n, ξ)
end

# Derivative of shifted Legendre polynomial
# d/dx P_n^*(x) = d/dx P_n(2x-1) = 2 * P'_n(2x-1)
@inline function shifted_legendre_derivative(n::Int64, x::Real)
    # Transform x ∈ [0,1] to ξ ∈ [-1,1]
    ξ = 2x - 1
    # Chain rule: d/dx P_n(2x-1) = 2 * P'_n(2x-1)
    return 2 * legendre_derivative(n, ξ)
end

"""
    basisfunction(basis::ShiftedLegendreBasis, αᵢ::Int, zᵢ::Real)

Evaluate `ShiftedLegendreBasis` with degree `αᵢ` at `zᵢ` ∈ [0,1].
"""
@inline function basisfunction(basis::ShiftedLegendreBasis, αᵢ::Int, zᵢ::Real)
    return shifted_legendre_polynomial(Int(αᵢ), zᵢ)
end

"""
    basisfunction_derivative(basis::ShiftedLegendreBasis, αᵢ::Int, zᵢ::Real)

Evaluate derivative of `ShiftedLegendreBasis` with degree `αᵢ` at `zᵢ` ∈ [0,1].
"""
@inline function basisfunction_derivative(basis::ShiftedLegendreBasis, αᵢ::Int, zᵢ::Real)
    return shifted_legendre_derivative(Int(αᵢ), zᵢ)
end

function Base.show(io::IO, ::ShiftedLegendreBasis)
    print(io, "ShiftedLegendreBasis()")
end

support(basis::LegendreBasis) = RealInterval(-1, 1)
support(basis::ShiftedLegendreBasis) = RealInterval(0, 1)
