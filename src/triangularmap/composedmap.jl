# Map composed of linear and polynomial map
struct ComposedMap <: AbstractComposedMap
    linearmap::LinearMap
    polynomialmap::PolynomialMap

    function ComposedMap(lm::LinearMap, pm::PolynomialMap)
        @assert numberdimensions(lm) == numberdimensions(pm) "Linear map and polynomial map must have the same number of dimensions"
        return new(lm, pm)
    end
end

function evaluate(C::ComposedMap, x::Vector{Float64})
    y = evaluate(C.linearmap, x)
    return evaluate(C.polynomialmap, y)
end

function evaluate(C::ComposedMap, X::Matrix{Float64})
    Y = evaluate(C.linearmap, X)
    return evaluate(C.polynomialmap, Y)
end

function inverse(C::ComposedMap, z::Vector{Float64})
    y = inverse(C.polynomialmap, z)
    return inverse(C.linearmap, y)
end

function inverse(C::ComposedMap, Z::Matrix{Float64})
    Y = inverse(C.polynomialmap, Z)
    return inverse(C.linearmap, Y)
end

function pullback(C::ComposedMap, x::Vector{Float64})
    y = evaluate(C.linearmap, x)
    return pullback(C.polynomialmap, y) ./ prod(C.linearmap.σ)  # Adjust for linear map scaling
end

function pullback(C::ComposedMap, X::Matrix{Float64})
    Y = evaluate(C.linearmap, X)
    return pullback(C.polynomialmap, Y) ./ prod(C.linearmap.σ)  # Adjust for linear map scaling
end

numberdimensions(C::ComposedMap) = numberdimensions(C.linearmap)

function Base.show(io::IO, C::ComposedMap)
    println(io, "ComposedMap with $(numberdimensions(C)) dimensions:")
    println(io, " linearmap: ", C.linearmap)
    println(io, " polynomialmap: ", C.polynomialmap)
end
