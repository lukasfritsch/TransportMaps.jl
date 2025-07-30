# API Reference

```@meta
CurrentModule = TransportMaps
```

This page provides a comprehensive reference for all exported functions and types in TransportMaps.jl.

## Transport Maps

### `PolynomialMap`

```julia
PolynomialMap(d::Int, order::Int, rectifier::AbstractRectifierFunction)
```

Constructs a triangular polynomial transport map.

**Arguments:**
- `d::Int`: Dimension of the map
- `order::Int`: Maximum polynomial degree
- `rectifier::AbstractRectifierFunction`: Rectifier function to use

**Example:**
```julia
M = PolynomialMap(2, 3, Softplus())
```

## Map Components

### `PolynomialMapComponent`

Individual polynomial components that make up triangular maps. These are typically not constructed directly by users.

### `HermiteBasis`

Hermite polynomial basis functions used in map components.

### `MultivariateBasis`

Multivariate polynomial basis construction utilities.

## Rectifier Functions

Rectifier functions control the monotonicity and behavior of transport map components.

### `IdentityRectifier`

```julia
IdentityRectifier()
```

Identity rectifier that applies no transformation (linear behavior).

### `Softplus`

```julia
Softplus()
```

Softplus rectifier: `softplus(x) = log(1 + exp(x))`. Ensures positive output and smooth behavior.

### `ShiftedELU`

```julia
ShiftedELU()
```

Shifted Exponential Linear Unit rectifier. Provides smooth behavior with improved gradient properties.

## Quadrature and Weights

### `GaussHermiteWeights`

```julia
GaussHermiteWeights(n::Int, d::Int)
```

Gauss-Hermite quadrature points and weights for integration.

**Arguments:**
- `n::Int`: Number of quadrature points per dimension
- `d::Int`: Number of dimensions

### `MonteCarloWeights`

```julia
MonteCarloWeights(n::Int, d::Int)
```

Monte Carlo integration weights using random sampling.

### `LatinHypercubeWeights`

```julia
LatinHypercubeWeights(n::Int, d::Int)
```

Latin hypercube sampling for more uniform coverage than Monte Carlo.

## Core Functions

### Map Evaluation and Operations

#### `evaluate`

```julia
evaluate(M::PolynomialMap, x::Vector)
```

Evaluate the transport map at point `x`.

#### `gradient_x`

```julia
gradient_x(M::PolynomialMap, x::Vector)
```

Compute the gradient of the map with respect to input `x`.

#### `jacobian`

```julia
jacobian(M::PolynomialMap, x::Vector)
```

Compute the Jacobian matrix of the map at point `x`.

#### `inverse`

```julia
inverse(M::PolynomialMap, y::Vector)
```

Compute the inverse map: find `x` such that `M(x) = y`.

#### `pushforward`

```julia
pushforward(M::PolynomialMap, density::Function, x::Vector)
```

Transform a density function through the map.

#### `pullback`

```julia
pullback(M::PolynomialMap, density::Function, y::Vector)
```

Pull back a density function through the inverse map.

### Coefficient Management

#### `setcoefficients!`

```julia
setcoefficients!(M::PolynomialMap, coeffs::Vector)
```

Set the coefficients of the transport map.

#### `getcoefficients`

```julia
getcoefficients(M::PolynomialMap) -> Vector
```

Get the current coefficients of the transport map.

#### `numbercoefficients`

```julia
numbercoefficients(M::PolynomialMap) -> Int
```

Return the total number of coefficients in the map.

#### `numberdimensions`

```julia
numberdimensions(M::PolynomialMap) -> Int
```

Return the dimension of the transport map.

### Optimization

#### `optimize!`

```julia
optimize!(M::PolynomialMap, target_density::Function, quadrature::AbstractQuadratureWeights)
```

Optimize the coefficients of the transport map to approximate the target density.

**Arguments:**
- `M`: Transport map to optimize
- `target_density`: Target probability density function
- `quadrature`: Quadrature scheme for integration

**Returns:** Optimization result object

#### `variance_diagnostic`

```julia
variance_diagnostic(M::PolynomialMap, target_density::Function, samples::Matrix) -> Float64
```

Compute a diagnostic measure of map quality. Values close to 1.0 indicate good approximation.

### Basis Functions

#### `Psi`

Evaluate basis functions (typically used internally).

#### `hermite_polynomial`

```julia
hermite_polynomial(n::Int, x::Float64) -> Float64
```

Evaluate the n-th Hermite polynomial at point x.

#### `hermite_derivative`

```julia
hermite_derivative(n::Int, x::Float64) -> Float64
```

Evaluate the derivative of the n-th Hermite polynomial at point x.

#### `multivariate_indices`

Generate multivariate polynomial indices for basis construction.

### Utilities

#### `gaussquadrature`

```julia
gaussquadrature(n::Int) -> (Vector, Vector)
```

Generate Gauss-Hermite quadrature points and weights.

#### `hybridrootfinder`

Numerical root finding utilities used for map inversion.

## Type Hierarchy

```
AbstractBasisFunction
├── AbstractPolynomialBasis
│   └── HermiteBasis
└── MultivariateBasis

AbstractMapComponent
└── PolynomialMapComponent

AbstractTriangularMap
└── PolynomialMap

AbstractRectifierFunction
├── IdentityRectifier
├── Softplus
└── ShiftedELU

AbstractQuadratureWeights
├── GaussHermiteWeights
├── MonteCarloWeights
└── LatinHypercubeWeights
```
