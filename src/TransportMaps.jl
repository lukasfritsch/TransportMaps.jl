module TransportMaps

using Distributions
using FastGaussQuadrature
using ForwardDiff
using LinearAlgebra
using Optim
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
# Basis functions and evaluation
export Psi
export evaluate
export f
export hermite_polynomial
export hermite_derivative
export multivariate_indices

# Map operations
export gradient_coefficients
export gradient_x
export jacobian
export inverse
export inverse_jacobian
export partial_derivative_x
export partial_derivative_xk
export pullback
export pushforward

# Coefficient utilities
export setcoefficients!
export getcoefficients
export numbercoefficients
export numberdimensions

# Quadrature and optimization
export gaussquadrature
export optimize!
export variance_diagnostic

# Utilities
export hybridrootfinder

# Export structs/types
export IdentityRectifier
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
include("mapcomponents/hermitebasis.jl")
include("mapcomponents/polynomialmapcomponent.jl")
include("mapcomponents/rectifier.jl")
include("triangularmap/polynomialmap.jl")
include("triangularmap/optimization.jl")

include("util/gaussquadrature.jl")
include("util/hybridrootfinder.jl")
include("util/quadraturepoints.jl")

end
