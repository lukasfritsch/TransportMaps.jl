module TransportMaps

using Distributions
using FastGaussQuadrature
using LinearAlgebra
using QuadGK
using QuasiMonteCarlo
using Random
using StatsFuns

# Abstract type definitions
abstract type AbstractBasisFunction end
abstract type AbstractPolynomialBasis <: AbstractBasisFunction end
abstract type AbstractMapComponent end
abstract type AbstractTriangularMap end
abstract type AbstractMultivariateBasis end
abstract type AbstractRectifierFunction end
abstract type AbstractQuadratureWeights end

# Export abstract types
export AbstractBasisFunction
export AbstractPolynomialBasis
export AbstractMapComponent
export AbstractTransportMap
export AbstractRectifierFunction
export AbstractQuadratureWeights

# Export functions/methods
export Psi
export evaluate
export f
export gradient_coefficients
export gradient_x
export hermite_derivative
export hermite_polynomial
export jacobian
export partial_derivative_x

# Export structs/types
export HermiteBasis
export MultivariateBasis
export PolynomialMapComponent
export PolynomialMap
export Softplus
export ShiftedELU
export GaussHermiteWeights
export MonteCarloWeights
export LatinHypercubeWeights

# Include files
include("mapcomponents/multivariatebasis.jl")
include("mapcomponents/hermite.jl")
include("mapcomponents/mapcomponent.jl")
include("mapcomponents/rectifier.jl")
include("map/triangularmap.jl")
include("util/quadrature.jl")

end
