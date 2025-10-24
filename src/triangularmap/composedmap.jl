# Map composed of linear and polynomial map
struct ComposedMap{T<:AbstractLinearMap} <: AbstractComposedMap
    linearmap::T
    polynomialmap::PolynomialMap

    function ComposedMap(lm::LinearMap, pm::PolynomialMap)
        if numberdimensions(lm) == 0
            T = typeof(lm)
            return new{T}(lm, pm)
        else
            @assert numberdimensions(lm) == numberdimensions(pm) "Linear map and polynomial map must have the same number of dimensions"
            T = typeof(lm)
            return new{T}(lm, pm)
        end
    end
end

function evaluate(C::ComposedMap{LinearMap}, x::Vector{Float64}) where T<:AbstractLinearMap
    y = evaluate(C.linearmap, x)
    return evaluate(C.polynomialmap, y)
end

function evaluate(C::ComposedMap{LinearMap}, X::Matrix{Float64}) where T<:AbstractLinearMap
    Y = evaluate(C.linearmap, X)
    return evaluate(C.polynomialmap, Y)
end

function inverse(C::ComposedMap{LinearMap}, z::Vector{Float64}) where T<:AbstractLinearMap
    y = inverse(C.polynomialmap, z)
    return inverse(C.linearmap, y)
end

function inverse(C::ComposedMap{LinearMap}, Z::Matrix{Float64}) where T<:AbstractLinearMap
    Y = inverse(C.polynomialmap, Z)
    return inverse(C.linearmap, Y)
end

# Specifically, for ComposedMap with LinearMap
function pullback(C::ComposedMap{LinearMap}, x::Vector{Float64})
    y = evaluate(C.linearmap, x)
    return pullback(C.polynomialmap, y) ./ prod(C.linearmap.σ)  # Adjust for linear map scaling
end

# Specifically, for ComposedMap with LinearMap
function pullback(C::ComposedMap{LinearMap}, X::Matrix{Float64})
    Y = evaluate(C.linearmap, X)
    return pullback(C.polynomialmap, Y) ./ prod(C.linearmap.σ)  # Adjust for linear map scaling
end

numberdimensions(C::ComposedMap{T}) where T<:AbstractLinearMap = numberdimensions(C.linearmap)

function Base.show(io::IO, C::ComposedMap{T}) where T<:AbstractLinearMap
    println(io, "ComposedMap{$(T)} with $(numberdimensions(C)) dimensions:")
    println(io, " linearmap: ", C.linearmap)
    println(io, " polynomialmap: ", C.polynomialmap)
end
