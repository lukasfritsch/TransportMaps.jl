# Hermite polynomial implementation with the specific interface requested

# Abstract types for basis functions
abstract type AbstractBasisFunction end
abstract type AbstractPolynomialBasis <: AbstractBasisFunction end

# Concrete type for Hermite polynomials
struct HermiteBasis <: AbstractPolynomialBasis end

# MVBasis struct for multi-indices as requested
struct MVBasis
    multi_index::Vector{Int}
    basis_type::AbstractPolynomialBasis
end

# Constructor for MVBasis with default Hermite basis
MVBasis(multi_index::Vector{Int}) = MVBasis(multi_index, HermiteBasis())

# Univariate Hermite polynomial evaluation
# Using probabilist's Hermite polynomials (physical polynomials)
function hermite_polynomial(n::Int, x::Float64)
    if n == 0
        return 1.0
    elseif n == 1
        return x
    else
        H_nm2 = 1.0  # H_{n-2}
        H_nm1 = x    # H_{n-1}
        for k in 2:n
            H_n = x * H_nm1 - (k - 1) * H_nm2
            H_nm2, H_nm1 = H_nm1, H_n
        end
        return H_nm1
    end
end

# Univariate basis function Psi(alpha_i::Float64, x_i::Float64)
function Psi(alpha_i::Float64, x_i::Float64)
    return hermite_polynomial(Int(alpha_i), x_i)
end

# Multivariate basis function Psi(alpha::Vector{Float64}, x::Vector{Float64})
function Psi(alpha::Vector{Float64}, x::Vector{Float64})
    @assert length(alpha) == length(x) "Dimension mismatch: alpha and x must have same length"
    return prod(Psi(alpha_i, x_i) for (alpha_i, x_i) in zip(alpha, x))
end

# Evaluate MVBasis at point x
function evaluate(mvb::MVBasis, x::Vector{Float64})
    @assert length(mvb.multi_index) == length(x) "Dimension mismatch: multi_index and x must have same length"
    alpha = Float64.(mvb.multi_index)
    return Psi(alpha, x)
end

# Multivariate function f(Psi::Vector{MVBasis}, coefficients::Vector{Float64})
function f(Psi_vec::Vector{MVBasis}, coefficients::Vector{Float64}, x::Vector{Float64})
    @assert length(Psi_vec) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * evaluate(mvb, x) for (coeff, mvb) in zip(coefficients, Psi_vec))
end

# Alternative interface matching the exact specification
function f(Psi_vec::Vector{MVBasis}, coefficients::Vector{Float64})
    return (x::Vector{Float64}) -> f(Psi_vec, coefficients, x)
end

# Derivative of univariate Hermite polynomial
function hermite_derivative(n::Int, x::Float64)
    if n == 0
        return 0.0
    else
        return n * hermite_polynomial(n - 1, x)
    end
end

# Partial derivative of multivariate basis w.r.t. x_j
function partial_derivative_x(mvb::MVBasis, x::Vector{Float64}, j::Int)
    @assert 1 <= j <= length(x) "Index j must be within bounds of x"
    @assert length(mvb.multi_index) == length(x) "Dimension mismatch"
    
    # Compute the product of all terms except the j-th, times the derivative of the j-th term
    result = hermite_derivative(mvb.multi_index[j], x[j])
    for (i, (alpha_i, x_i)) in enumerate(zip(mvb.multi_index, x))
        if i != j
            result *= hermite_polynomial(alpha_i, x_i)
        end
    end
    return result
end

# Gradient of MVBasis w.r.t. x
function gradient_x(mvb::MVBasis, x::Vector{Float64})
    return [partial_derivative_x(mvb, x, j) for j in 1:length(x)]
end

# Partial derivative of f w.r.t. x_j
function partial_derivative_x(Psi_vec::Vector{MVBasis}, coefficients::Vector{Float64}, x::Vector{Float64}, j::Int)
    @assert length(Psi_vec) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * partial_derivative_x(mvb, x, j) for (coeff, mvb) in zip(coefficients, Psi_vec))
end

# Gradient of f w.r.t. x
function gradient_x(Psi_vec::Vector{MVBasis}, coefficients::Vector{Float64}, x::Vector{Float64})
    return [partial_derivative_x(Psi_vec, coefficients, x, j) for j in 1:length(x)]
end

# Derivative of f w.r.t. coefficients (this is just the basis function values)
function gradient_coefficients(Psi_vec::Vector{MVBasis}, x::Vector{Float64})
    return [evaluate(mvb, x) for mvb in Psi_vec]
end