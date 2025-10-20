module TransportMaps

using Distributions
using FastGaussQuadrature
using ForwardDiff
using LinearAlgebra
using Optim
using QuasiMonteCarlo
using Random
using StatsFuns
using Statistics

# Abstract type definitions
abstract type AbstractBasisFunction end
abstract type AbstractPolynomialBasis <: AbstractBasisFunction end
abstract type AbstractMapComponent end
abstract type AbstractTransportMap end
abstract type AbstractTriangularMap <: AbstractTransportMap end
abstract type AbstractMultivariateBasis end
abstract type AbstractRectifierFunction end
abstract type AbstractQuadratureWeights end
abstract type AbstractMapDensity end
abstract type AbstractComposedMap end

# Export abstract types
export AbstractBasisFunction
export AbstractPolynomialBasis
export AbstractMapComponent
export AbstractTransportMap
export AbstractTriangularMap
export AbstractRectifierFunction
export AbstractQuadratureWeights
export AbstractMapDensity
export AbstractComposedMap

# Export functions/methods
# Basis functions and evaluation
export Psi
export basisfunction
export basisfunction_derivative
export basistype
export evaluate
export edge_controlled_hermite_polynomial
export edge_controlled_hermite_derivative
export f
export hermite_polynomial
export hermite_derivative
export multivariate_indices

# Map operations
export DiagonalMap
export NoMixedMap
export gradient
export gradient_coefficients
export gradient_z
export gradient_zk
export jacobian
export inverse
export inverse_jacobian
export partial_derivative_z
export partial_derivative_zk
export pullback
export pushforward

# Coefficient utilities
export setcoefficients!
export getcoefficients
export numbercoefficients
export numberdimensions
export getmultiindexsets

# Quadrature and optimization
export gaussquadrature
export kldivergence
export kldivergence_gradient
export optimize!
export variance_diagnostic
export reduced_margin
export adaptive_optimization
export AdaptiveTransportMap

# Conditional densities and samples
export conditional_density
export conditional_sample
export multivariate_conditional_density
export multivariate_conditional_sample

# Utilities
export hybridrootfinder

# Export structs/types
export IdentityRectifier
export HermiteBasis
export LinearizedHermiteBasis
export CubicSplineHermiteBasis
export GaussianWeightedHermiteBasis
export RadialBasis

export MultivariateBasis
export LinearMap
export ComposedMap
export PolynomialMapComponent
export PolynomialMap
export Softplus
export ShiftedELU
export GaussHermiteWeights
export MonteCarloWeights
export LatinHypercubeWeights
export MapTargetDensity
export MapReferenceDensity
export SparseSmolyakWeights

# Include files
include("util/mapdensity.jl")

include("mapcomponents/univariatebases/hermitebasis.jl")
include("mapcomponents/univariatebases/linearizedhermitebasis.jl")
include("mapcomponents/univariatebases/cubicsplinehermitebasis.jl")
include("mapcomponents/univariatebases/gaussianweighthermitebasis.jl")

include("mapcomponents/multivariatebasis.jl")
include("mapcomponents/multivariateindices.jl")
include("mapcomponents/polynomialmapcomponent.jl")
include("mapcomponents/rectifier.jl")

include("triangularmap/polynomialmap.jl")
include("triangularmap/conditionaldensities.jl")
include("triangularmap/linearmap.jl")
include("triangularmap/composedmap.jl")

include("optimization/mapfromdensity.jl")
include("optimization/mapfromsamples.jl")
include("optimization/adaptivetransportmap.jl")

include("util/finitedifference.jl")
include("util/gaussquadrature.jl")
include("util/hybridrootfinder.jl")
include("util/quadraturepoints.jl")
include("util/smolyak.jl")

end
