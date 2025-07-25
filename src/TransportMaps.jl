module TransportMaps

using DataFrames
using LinearAlgebra
using QuadGK
using Random

# Todo : add definitions of abstract types

# Types

# Structs
export BasisFunction
export MultiIndex
export PolynomialFunction
export VectorPolynomialMap

# Methods
export hermite_poly
export evaluate_basis
export evaluate
export df_dxk
export compute_Mk
export dMk_dxk

include("mapcomponents/components.jl")
include("mapcomponents/optimization.jl")

end
