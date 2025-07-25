module TransportMaps

# Include existing components
include("mapcomponents/components.jl")

# Include new Hermite polynomial interface
include("hermite.jl")

# Export the main interface functions
export Psi, f, MVBasis, HermiteBasis
export AbstractBasisFunction, AbstractPolynomialBasis
export evaluate, gradient_x, gradient_coefficients, partial_derivative_x
export hermite_polynomial, hermite_derivative

end
