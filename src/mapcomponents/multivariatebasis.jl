struct MultivariateBasis{T<:AbstractPolynomialBasis} <: AbstractMultivariateBasis
    multiindexset::Vector{Int}
    univariatebases::Vector{T}  # Store univariate basis for each dimension
end

function MultivariateBasis(multiindexset::Vector{Int}, ::Type{T}) where T<:AbstractPolynomialBasis
    if T <: LinearizedHermiteBasis
        univariatebases = [T(degree) for degree in multiindexset]
    else
        univariatebases = [T() for _ in multiindexset]
    end
    return MultivariateBasis{T}(multiindexset, univariatebases)
end

# Backwards-compatible constructor that accepts a basis instance
MultivariateBasis(multiindexset::Vector{Int}, basistype::AbstractPolynomialBasis) = MultivariateBasis(multiindexset, typeof(basistype))

# Multivariate basis function Psi(alpha::Vector{<:Real}, z::Vector{<:Real})
function Psi(alpha::Vector{<:Real}, z::Vector{<:Real}, univariatebases::Vector{T}) where T<:AbstractPolynomialBasis
    @assert length(alpha) == length(z) "Dimension mismatch: alpha and z must have same length"
    return prod(basisfunction(ub, αᵢ, zᵢ) for (αᵢ, zᵢ, ub) in zip(alpha, z, univariatebases))
end

# Evaluate MultivariateBasis at point z
function evaluate(mvb::MultivariateBasis{T}, z::Vector{<:Real}) where T<:AbstractPolynomialBasis
    @assert length(mvb.multiindexset) == length(z) "Dimension mismatch: multiindexset and z must have same length"
    alpha = Real.(mvb.multiindexset)
    return Psi(alpha, z, mvb.univariatebases)
end

# Multivariate function f(Ψ::Vector{MultivariateBasis}, coefficients::Vector{<:Real})
function f(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}) where T<:AbstractPolynomialBasis
    @assert length(Ψ) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * evaluate(mvb, z) for (coeff, mvb) in zip(coefficients, Ψ))
end

# Alternative interface matching the exact specification
function f(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}) where T<:AbstractPolynomialBasis
    return (z::Vector{<:Real}) -> f(Ψ, coefficients, z)
end

# Gradient of MultivariateBasis w.r.t. z
function gradient_z(mvb::MultivariateBasis{T}, z::Vector{<:Real}) where T<:AbstractPolynomialBasis
    return [partial_derivative_z(mvb.univariatebases, mvb.multiindexset, z, j) for j in 1:length(z)]
end

# Partial derivative of multivariate basis w.r.t. z_j
function partial_derivative_z(bases::Vector{T}, α::Vector{Int}, z::Vector{<:Real}, j::Int) where T<:AbstractPolynomialBasis
    @assert 1 <= j <= length(z) "Index j must be within bounds of z"
    @assert length(bases) == length(z) "Dimension mismatch"

    # Compute the product of all terms except the j-th, times the derivative of the j-th term
    result = basisfunction_derivative(bases[j], α[j], z[j])
    for (i, (αᵢ, zᵢ)) in enumerate(zip(α, z))
        if i != j
            result *= basisfunction(bases[i], αᵢ, zᵢ)
        end
    end
    return result
end

partial_derivative_z(mvb::MultivariateBasis{T}, z::Vector{<:Real}, j::Int) where T<:AbstractPolynomialBasis = partial_derivative_z(mvb.univariatebases, mvb.multiindexset, z, j)

# Partial derivative of f w.r.t. z_j
function partial_derivative_z(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}, j::Int) where T<:AbstractPolynomialBasis
    @assert length(Ψ) == length(coefficients) "Number of basis functions must equal number of coefficients"
    return sum(coeff * partial_derivative_z(mvb, z, j) for (coeff, mvb) in zip(coefficients, Ψ))
end

# Gradient of f w.r.t. z
function gradient_z(Ψ::Vector{MultivariateBasis{T}}, coefficients::Vector{<:Real}, z::Vector{<:Real}) where T<:AbstractPolynomialBasis
    return [partial_derivative_z(Ψ, coefficients, z, j) for j in 1:length(z)]
end

# Derivative of f w.r.t. coefficients (this is just the basis function values)
function gradient_coefficients(Ψ::Vector{MultivariateBasis{T}}, z::Vector{<:Real}) where T<:AbstractPolynomialBasis
    return [evaluate(mvb, z) for mvb in Ψ]
end

# Return multi-indices for a polynomial of degree p in k dimensions.
function multivariate_indices(p::Int, k::Int; mode::Symbol = :total)
    @assert p >= 0 "Degree p must be non-negative"
    @assert k >= 1 "Dimension k must be at least 1"

    if mode == :total
        # total-order multi-indices
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

function basistype(mvb::MultivariateBasis{T}) where T<:AbstractPolynomialBasis
    return T
end

# Display methods for MultivariateBasis
function Base.show(io::IO, basis::MultivariateBasis{T}) where T<:AbstractPolynomialBasis
    basis_name = string(basistype(basis))


    degree = sum(basis.multiindexset)
    dimension = length(basis.multiindexset)

    print(io, "MultivariateBasis(")
    print(io, "$(basis.multiindexset), ")
    print(io, "degree=$degree, ")
    print(io, "dim=$dimension, ")
    print(io, "basis=$basis_name)")
end

function Base.show(io::IO, ::MIME"text/plain", basis::MultivariateBasis{T}) where T<:AbstractPolynomialBasis
    basis_name = string(basistype(basis))

    degree = sum(basis.multiindexset)
    dimension = length(basis.multiindexset)

    println(io, "MultivariateBasis:")
    println(io, "  Multi-index: $(basis.multiindexset)")
    println(io, "  Total degree: $degree")
    println(io, "  Dimension: $dimension")
    println(io, "  Basis type: $basis_name")

    # Show individual polynomial degrees for each dimension
    if dimension > 1
        println(io, "  Individual degrees: $(basis.multiindexset)")
    end
end
