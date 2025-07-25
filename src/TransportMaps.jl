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
export compute_Mk
export evaluate
export f
export gradient_coefficients
export gradient_x
export hermite_derivative
export hermite_polynomial
export partial_derivative_x
export softplus
export shifted_elu

# Export structs/types
export HermiteBasis
export MVBasis

# Include files
include("mapcomponents/multivariatebasis.jl")
include("mapcomponents/hermite.jl")
include("mapcomponents/map.jl")
include("mapcomponents/rectifier.jl")

end
