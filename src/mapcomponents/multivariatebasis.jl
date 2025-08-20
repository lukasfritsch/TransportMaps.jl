# MVBasis struct for multi-indices
struct MultivariateBasis <: AbstractMultivariateBasis
    multi_index::Vector{Int}
    basis_type::AbstractPolynomialBasis
end

# Multivariate basis function Psi(alpha::Vector{<:Real}, z::Vector{<:Real})
function Psi(alpha::Vector{<:Real}, z::Vector{<:Real}, basis::AbstractPolynomialBasis)
    @assert length(alpha) == length(z) "Dimension mismatch: alpha and z must have same length"
    return prod(basisfunction(basis, αᵢ, zᵢ) for (αᵢ, zᵢ) in zip(alpha, z))
end

# Evaluate MultivariateBasis at point z
function evaluate(mvb::MultivariateBasis, z::Vector{<:Real})
    @assert length(mvb.multi_index) == length(z) "Dimension mismatch: multi_index and z must have same length"
    alpha = Real.(mvb.multi_index)
    return Psi(alpha, z, mvb.basis_type)
end

# Multivariate function f(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real})
function f(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real}, z::Vector{<:Real})
    @assert length(Ψ) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * evaluate(mvb, z) for (coeff, mvb) in zip(coefficients, Ψ))
end

# Alternative interface matching the exact specification
function f(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real})
    return (z::Vector{<:Real}) -> f(Ψ, coefficients, z)
end

# Gradient of MultivariateBasis w.r.t. z
function gradient_z(mvb::MultivariateBasis, z::Vector{<:Real})
    return [partial_derivative_z(mvb, z, j) for j in 1:length(z)]
end

# Partial derivative of multivariate basis w.r.t. z_j
function partial_derivative_z(mvb::MultivariateBasis, z::Vector{<:Real}, j::Int)
    @assert 1 <= j <= length(z) "Index j must be within bounds of z"
    @assert length(mvb.multi_index) == length(z) "Dimension mismatch"

    # Compute the product of all terms except the j-th, times the derivative of the j-th term
    result = basisfunction_derivative(mvb.basis_type, mvb.multi_index[j], z[j])
    for (i, (αᵢ, zᵢ)) in enumerate(zip(mvb.multi_index, z))
        if i != j
            result *= basisfunction(mvb.basis_type, αᵢ, zᵢ)
        end
    end
    return result
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

# Return multi-indices for a polynomial of degree p in d dimensions
# shamelessly copied from UncertaintyQuantification.jl
function multivariate_indices(p::Int, d::Int)
    No = Int64(factorial(p + d) / factorial(p) / factorial(d))

    idx = vcat(zeros(Int64, 1, d), Matrix(I, d, d), zeros(Int64, No - d - 1, d))

    pᵢ = ones(Int64, d, No)

    for k in 2:No
        for i in 1:d
            pᵢ[i, k] = sum(pᵢ[i:d, k - 1])
        end
    end

    P = d + 1
    for k in 2:p
        L = P
        for j in 1:d, m in (L - pᵢ[j, k] + 1):L
            P += 1
            idx[P, :] = idx[m, :]
            idx[P, j] = idx[P, j] + 1
        end
    end

    return map(collect, eachrow(idx))
end

# Display methods for MultivariateBasis
function Base.show(io::IO, basis::MultivariateBasis)
    basis_type = typeof(basis.basis_type)
    basis_name = string(basis_type)
    if basis_name == "HermiteBasis"
        basis_name = "HermiteBasis(edge_control=:$(basis.basis_type.edge_control))"
    end

    degree = sum(basis.multi_index)
    dimension = length(basis.multi_index)

    print(io, "MultivariateBasis(")
    print(io, "$(basis.multi_index), ")
    print(io, "degree=$degree, ")
    print(io, "dim=$dimension, ")
    print(io, "basis=$basis_name)")
end

function Base.show(io::IO, ::MIME"text/plain", basis::MultivariateBasis)
    basis_type = typeof(basis.basis_type)
    basis_name = string(basis_type)
    if basis_name == "HermiteBasis"
        basis_name = "HermiteBasis(edge_control=:$(basis.basis_type.edge_control))"
    end

    degree = sum(basis.multi_index)
    dimension = length(basis.multi_index)

    println(io, "MultivariateBasis:")
    println(io, "  Multi-index: $(basis.multi_index)")
    println(io, "  Total degree: $degree")
    println(io, "  Dimension: $dimension")
    println(io, "  Basis type: $basis_name")

    # Show individual polynomial degrees for each dimension
    if dimension > 1
        println(io, "  Individual degrees: $(basis.multi_index)")
    end
end
