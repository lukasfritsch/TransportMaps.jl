# MVBasis struct for multi-indices
struct MultivariateBasis <: AbstractMultivariateBasis
    multi_index::Vector{Int}
    basis_type::AbstractPolynomialBasis
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
