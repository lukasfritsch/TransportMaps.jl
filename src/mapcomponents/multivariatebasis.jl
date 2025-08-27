struct MultivariateBasis <: AbstractMultivariateBasis
    multi_index::Vector{Int}
    univariate_bases::Vector{AbstractPolynomialBasis}  # per-dimension univariate bases

    # Constructor taking a single univariate basis and expanding it to all dimensions
    function MultivariateBasis(multi_index::Vector{Int}, basis_type::AbstractPolynomialBasis)
        # expand the single basis to all dimensions
        univariate_bases = [basis_type for _ in 1:length(multi_index)]
        return new(multi_index, univariate_bases)
    end

    # Constructor taking explicit per-dimension bases
    function MultivariateBasis(multi_index::Vector{Int}, univariate_bases::Vector{AbstractPolynomialBasis})
        @assert length(univariate_bases) == length(multi_index) "basis_functions length must match multi_index length"
        return new(multi_index, univariate_bases)
    end
end

# Multivariate basis function Psi(alpha::Vector{<:Real}, z::Vector{<:Real})
function Psi(alpha::Vector{<:Real}, z::Vector{<:Real}, bases::AbstractVector{<:AbstractPolynomialBasis})
    @assert length(alpha) == length(z) "Dimension mismatch: alpha and z must have same length"
    @assert length(bases) == length(z) "Dimension mismatch: bases and z must have same length"
    return prod(basisfunction(b, αᵢ, zᵢ) for (b, αᵢ, zᵢ) in zip(bases, alpha, z))
end

# Convenience overload: single univariate basis expanded to all dimensions
function Psi(alpha::Vector{<:Real}, z::Vector{<:Real}, basis::AbstractPolynomialBasis)
    bases = [basis for _ in 1:length(z)]
    return Psi(alpha, z, bases)
end

# Evaluate MultivariateBasis at point z
function evaluate(mvb::MultivariateBasis, z::Vector{<:Real})
    @assert length(mvb.multi_index) == length(z) "Dimension mismatch: multi_index and z must have same length"
    alpha = Real.(mvb.multi_index)
    return Psi(alpha, z, mvb.univariate_bases)
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
    result = basisfunction_derivative(mvb.univariate_bases[j], mvb.multi_index[j], z[j])
    for (i, (αᵢ, zᵢ)) in enumerate(zip(mvb.multi_index, z))
        if i != j
            result *= basisfunction(mvb.univariate_bases[i], αᵢ, zᵢ)
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

# Return multi-indices for a polynomial of degree p in k dimensions.
function multivariate_indices(p::Int, k::Int; mode::Symbol = :total)
    @assert p >= 0 "Degree p must be non-negative"
    @assert k >= 1 "Dimension k must be at least 1"

    if mode == :total
        # original total-order implementation
        No = Int64(factorial(p + k) / factorial(p) / factorial(k))

        idx = vcat(zeros(Int64, 1, k), Matrix(I, k, k), zeros(Int64, No - k - 1, k))

        pᵢ = ones(Int64, k, No)

        for kk in 2:No
            for i in 1:k
                pᵢ[i, kk] = sum(pᵢ[i:k, kk - 1])
            end
        end

        P = k + 1
        for kk in 2:p
            L = P
            for j in 1:k, m in (L - pᵢ[j, kk] + 1):L
                P += 1
                idx[P, :] = idx[m, :]
                idx[P, j] = idx[P, j] + 1
            end
        end

        return map(collect, eachrow(idx))

    elseif mode == :diagonal
        # Diagonal multi-index set for a fixed coordinate k:
        # J_k^D = { j : ||j||_1 <= p  and j_i = 0 for all i != k }
        # i.e., vectors with only the k-th entry possibly non-zero (0..p)
        # diagonal for the fixed component k: only j_{k} may be non-zero
        inds = Vector{Vector{Int}}()
        for t in 0:p
            v = zeros(Int, k)
            v[k] = t
            push!(inds, v)
        end
        return inds

    elseif mode == :no_mixed
        # No-mixed terms (possibly restricted to first k coordinates):
        # J_k^{NM} = { j : ||j||_1 <= p, j_i * j_l = 0 for i != l, and j_i = 0 for i > k }
        inds = Vector{Vector{Int}}()
        # constant term
        push!(inds, zeros(Int, k))
        # for each allowed coordinate 1..k, add pure powers 1..p
        for j in 1:k
            for t in 1:p
                v = zeros(Int, k)
                v[j] = t
                push!(inds, v)
            end
        end
        return inds

    else
        error("Unknown mode: $mode. Supported modes are :total, :diagonal, :no_mixed")
    end
end

# Display methods for MultivariateBasis
function Base.show(io::IO, basis::MultivariateBasis)
    basis_type = typeof(basis.univariate_bases[1])
    basis_name = string(basis_type)
    if basis_name == "HermiteBasis"
        basis_name = "HermiteBasis(edge_control=:$(basis.univariate_bases[1].edge_control))"
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

    basis_type = typeof(basis.univariate_bases[1])
    basis_name = string(basis_type)
    if basis_name == "HermiteBasis"
        basis_name = "HermiteBasis(edge_control=:$(basis.univariate_bases[1].edge_control))"
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
