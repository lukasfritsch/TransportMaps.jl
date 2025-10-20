# Multivariate index set generation and reduced margin of multi-index sets

"""
    multivariate_indices(p::Int, k::Int; mode::Symbol=:total)

Generate multi-index sets Λ for multivariate polynomial bases.

# Arguments
- `p::Int`: Maximum total degree of the polynomials.
- `k::Int`: Dimension of the input space.
- `mode::Symbol`: Type of multi-index set to generate. Supported modes are:
    - `:total`: Total-order multi-indices (default).
    - `:diagonal`: Diagonal multi-indices for a fixed coordinate k.
    - `:no_mixed`: No-mixed multi-indices.

# Returns
- `Vector{Vector{Int}}`: A vector of multi-indices, where each multi-index is represented as a vector of integers.

"""
function multivariate_indices(p::Int, k::Int; mode::Symbol=:total)
    @assert p >= 0 "Degree p must be non-negative"
    @assert k >= 1 "Dimension k must be at least 1"

    if mode == :total
        # total-order multi-indices
        No = Int64(factorial(p + k) / factorial(p) / factorial(k))

        idx = vcat(zeros(Int64, 1, k), Matrix(I, k, k), zeros(Int64, No - k - 1, k))

        pᵢ = ones(Int64, k, No)

        for kk in 2:No
            for i in 1:k
                pᵢ[i, kk] = sum(pᵢ[i:k, kk-1])
            end
        end

        P = k + 1
        for kk in 2:p
            L = P
            for j in 1:k, m in (L-pᵢ[j, kk]+1):L
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

"""
    reduced_margin(Λ::Vector{<:Vector{Int}})

Compute the reduced margin of a multi-index set Λ that ensures downward closure.

# Arguments
- `Λ::Vector{<:Vector{Int}}`: A vector of multi-indices, where each multi-index is represented as a vector of integers.

# Returns
- `Vector{Vector{Int}}`: The reduced margin of Λ, consisting of multi-indices that ensure downward closure.
"""
function reduced_margin(Λ::Vector{<:Vector{Int}})
    # Handle empty input
    if isempty(Λ)
        return Vector{Vector{Int}}()
    end

    d = length(Λ[1])

    # Convert to Set of tuples for O(1) membership tests
    present = Set(tuple(α...) for α in Λ)

    # Generate candidates: all multi-indices one step above elements in Λ
    candidates = Set{NTuple{d, Int}}()
    for β in Λ
        for i in 1:d
            α = copy(β)
            α[i] += 1
            push!(candidates, tuple(α...))
        end
    end

    # A multi-index α is in the reduced margin if:
    # 1. α ∉ Λ (not already in the set)
    # 2. For all j where α[j] > 0: (α - e_j) ∈ Λ (all neighbors are in Λ)
    reduced = Vector{Vector{Int}}()
    for α_tuple in candidates
        # Skip if already in Λ
        α_tuple in present && continue

        α = collect(α_tuple)
        is_reduced = true

        # Check all non-zero coordinates
        for j in 1:d
            if α[j] > 0
                # Check if neighbor (α - e_j) is in Λ
                neighbor = ntuple(i -> i == j ? α[i] - 1 : α[i], d)
                if neighbor ∉ present
                    is_reduced = false
                    break
                end
            end
        end

        is_reduced && push!(reduced, α)
    end

    return reduced
end
