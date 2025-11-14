# Implement a linear transport map (scaling by mean and standard deviation)
struct LinearMap <: AbstractLinearMap
    μ::Vector{Float64}
    σ::Vector{Float64}

    # Constructor that computes mean and std from samples
    function LinearMap(samples::Matrix{Float64})
        μ = mean(samples, dims=1)[:]
        σ = std(samples, dims=1)[:]
        return new(μ, σ)
    end

    # Identity map
    function LinearMap()
        return new(zeros(Float64, 0), ones(Float64, 0))
    end

    # Identity map with specified dimension
    function LinearMap(dim::Int)
        return new(zeros(Float64, dim), ones(Float64, dim))
    end

    function LinearMap(μ::Vector{Float64}, σ::Vector{Float64})
        return new(μ, σ)
    end
end

function setparameters!(map::LinearMap, μ::Vector{Float64}, σ::Vector{Float64})
    map.μ .= μ
    map.σ .= σ
end

function evaluate(L::LinearMap, x::Vector{Float64})
    if numberdimensions(L) == 0
        return x  # Identity map if no dimensions are defined
    else
        @assert length(x) == length(L.μ) "Input vector must have the same length as dimensions in the map"
        return (x .- L.μ) ./ L.σ
    end
end

# Scale and shift input data
function evaluate(L::LinearMap, X::Matrix{Float64})
    if numberdimensions(L) == 0
        return X  # Identity map if no dimensions are defined
    else
        @assert size(X, 2) == length(L.μ) "Input data must have the same number of columns as dimensions in the map"
        return (X .- L.μ') ./ L.σ'
    end
end

function inverse(L::LinearMap, y::Vector{Float64})
    if numberdimensions(L) == 0
        return y # Identity map if no dimensions are defined
    else
        @assert length(y) == length(L.μ) "Input vector must have the same length as dimensions in the map"
        return (y .* L.σ) .+ L.μ
    end
end

# Inverse operation: scale back to original space
function inverse(L::LinearMap, Y::Matrix{Float64})
    if numberdimensions(L) == 0
        return Y  # Identity map if no dimensions are defined
    else
        @assert size(Y, 2) == length(L.μ) "Input data must have the same number of columns as dimensions in the map"
        return (Y .* L.σ') .+ L.μ'
    end
end

function jacobian(L::LinearMap)
    return prod(L.σ)
end

numberdimensions(L::LinearMap) = length(L.μ)

function Base.show(io::IO, L::LinearMap)
    print(io, "LinearMap($(numberdimensions(L))-dimensional, ")
    print(io, "μ: ", L.μ, ", ")
    print(io, "σ: ", L.σ, ")")
end

function Base.show(io::IO, ::MIME"text/plain", L::LinearMap)
    println(io, "LinearMap with $(numberdimensions(L)) dimensions")
    println(io, "  μ: ", L.μ)
    println(io, "  σ: ", L.σ)
end
