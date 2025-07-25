module TransportMaps

using LinearAlgebra
using QuadGK
using Random
using StatsFuns

# Abstract type definitions
abstract type AbstractBasisFunction end
abstract type AbstractPolynomialBasis <: AbstractBasisFunction end

# Export abstract types
export AbstractBasisFunction
export AbstractPolynomialBasis

# Export functions/methods
export Psi
export evaluate
export f
export gradient_coefficients
export gradient_x
export hermite_derivative
export hermite_polynomial
export partial_derivative_x

# Export structs/types
export HermiteBasis
export MVBasis

# Export fruit functionality
export AbstractFruit
export Banana
export ripeness
export name
export is_implemented
export implemented_fruits
export all_fruits

# Include files
include("mapcomponents/hermite.jl")
include("mapcomponents/fruit.jl")

end
