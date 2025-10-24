struct LaplaceMap <: AbstractLinearMap
    μ::Vector{Float64}  # Mean vector
    Σ::Matrix{Float64}  # Covariance matrix
    Chol::Matrix{Float64}  # Cholesky decomposition of the covariance matrix

    function LaplaceMap(samples::Matrix{Float64})
        μ = mean(samples, dims=1)[:]
        Σ = cov(samples, corrected=true)
        # Cholesky decomposition (Σ = L * L')
        Chol = cholesky(Σ).L
        return new(μ, Σ, Chol)
    end
end

function evaluate(L::LaplaceMap, x::Vector{Float64})
    @assert length(x) == length(L.μ) "Input vector must have the same length as dimensions in the map"
    return L.Chol \ (x .- L.μ)
end


function evaluate(L::LaplaceMap, X::Matrix{Float64})
    @assert size(X, 2) == length(L.μ) "Input data must have the same number of columns as dimensions in the map"
    return (X .- L.μ') * inv(L.Chol)'
end

function inverse(L::LaplaceMap, y::Vector{Float64})
    @assert length(y) == length(L.μ) "Input vector must have the same length as dimensions in the map"
    return L.Chol * y .+ L.μ
end

function inverse(L::LaplaceMap, Y::Matrix{Float64})
    @assert size(Y, 2) == length(L.μ) "Input data must have the same number of columns as dimensions in the map"
    return Y * L.Chol' .+ L.μ'
end

numberdimensions(L::LaplaceMap) = length(L.μ)

function Base.show(io::IO, L::LaplaceMap)
    print(io, "LaplaceMap($(numberdimensions(L))-dimensional")
    print(io, " μ: ", L.μ, ", ")
    print(io, " Σ: ", L.Σ, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", L::LaplaceMap)
    println(io, "LaplaceMap with $(numberdimensions(L)) dimensions")
    println(io, "  μ: ", L.μ)
    println(io, "  Σ: ", L.Σ)
end
