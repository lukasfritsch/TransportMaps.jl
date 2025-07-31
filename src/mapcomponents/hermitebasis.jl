# Concrete type for Hermite polynomials
struct HermiteBasis <: AbstractPolynomialBasis end

# Constructor for MultivariateBasis with default Hermite basis
MultivariateBasis(multi_index::Vector{Int}) = MultivariateBasis(multi_index, HermiteBasis())

# Univariate Hermite polynomial evaluation
# Using probabilist's Hermite polynomials (physical polynomials)
function hermite_polynomial(n::Int, z::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return z
    else
        H_nm2 = 1.0  # H_{n-2}
        H_nm1 = z    # H_{n-1}
        for k in 2:n
            H_n = z * H_nm1 - (k - 1) * H_nm2
            H_nm2, H_nm1 = H_nm1, H_n
        end
        return H_nm1
    end
end

# Univariate basis function Psi(alpha_i::Real, z_i::Real)
function Psi(alpha_i::Real, z_i::Real)
    return hermite_polynomial(Int(alpha_i), z_i)
end

# Multivariate basis function Psi(alpha::Vector{<:Real}, z::Vector{<:Real})
function Psi(alpha::Vector{<:Real}, z::Vector{<:Real})
    @assert length(alpha) == length(z) "Dimension mismatch: alpha and z must have same length"
    return prod(Psi(alpha_i, z_i) for (alpha_i, z_i) in zip(alpha, z))
end

# Evaluate MultivariateBasis at point z
function evaluate(mvb::MultivariateBasis, z::Vector{<:Real})
    @assert length(mvb.multi_index) == length(z) "Dimension mismatch: multi_index and z must have same length"
    alpha = Real.(mvb.multi_index)
    return Psi(alpha, z)
end

# Multivariate function f(Psi::Vector{MultivariateBasis}, coefficients::Vector{<:Real})
function f(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real}, z::Vector{<:Real})
    @assert length(Ψ) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * evaluate(mvb, z) for (coeff, mvb) in zip(coefficients, Ψ))
end

# Alternative interface matching the exact specification
function f(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real})
    return (z::Vector{<:Real}) -> f(Ψ, coefficients, z)
end

# Derivative of univariate Hermite polynomial
function hermite_derivative(n::Int, z::Real)
    if n == 0
        return 0.0
    else
        return n * hermite_polynomial(n - 1, z)
    end
end

# Partial derivative of multivariate basis w.r.t. z_j
function partial_derivative_z(mvb::MultivariateBasis, z::Vector{<:Real}, j::Int)
    @assert 1 <= j <= length(z) "Index j must be within bounds of z"
    @assert length(mvb.multi_index) == length(z) "Dimension mismatch"

    # Compute the product of all terms except the j-th, times the derivative of the j-th term
    result = hermite_derivative(mvb.multi_index[j], z[j])
    for (i, (alpha_i, z_i)) in enumerate(zip(mvb.multi_index, z))
        if i != j
            result *= hermite_polynomial(alpha_i, z_i)
        end
    end
    return result
end

# Gradient of MultivariateBasis w.r.t. z
function gradient_z(mvb::MultivariateBasis, z::Vector{<:Real})
    return [partial_derivative_z(mvb, z, j) for j in 1:length(z)]
end

# Partial derivative of f w.r.t. z_j
function partial_derivative_z(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real}, z::Vector{<:Real}, j::Int)
    @assert length(Ψ) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * partial_derivative_z(mvb, z, j) for (coeff, mvb) in zip(coefficients, Ψ))
end

# Gradient of f w.r.t. z
function gradient_z(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real}, z::Vector{<:Real})
    return [partial_derivative_z(Ψ, coefficients, z, j) for j in 1:length(z)]
end

# Derivative of f w.r.t. coefficients (this is just the basis function values)
function gradient_coefficients(Ψ::Vector{MultivariateBasis}, z::Vector{<:Real})
    return [evaluate(mvb, z) for mvb in Ψ]
end

# Display methods for HermiteBasis
function Base.show(io::IO, ::HermiteBasis)
    print(io, "HermiteBasis()")
end

function Base.show(io::IO, ::MIME"text/plain", ::HermiteBasis)
    println(io, "HermiteBasis:")
    println(io, "  Type: Probabilist's Hermite polynomials")
    println(io, "  Orthogonal with respect to: standard Gaussian measure")
    println(io, "  Recursion: H_n(z) = z·H_{n-1}(z) - (n-1)·H_{n-2}(z)")
    println(io, "  First few polynomials: H_0(z)=1, H_1(z)=z, H_2(z)=z²-1, ...")
end
