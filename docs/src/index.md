# TransportMaps.jl

```@meta
CurrentModule = TransportMaps
```

This is an implementation of triangular transport maps in Julia based on the description in [marzouk2016](@cite).
For a comprehensive introduction to transport maps, see [ramgraber2025](@cite). The theoretical foundations for monotone triangular transport maps are detailed in [marzouk2016](@cite), [baptista2023](@cite). For practical applications in structural health monitoring and Bayesian inference, see [grashorn2024](@cite).

## What are Transport Maps?

Transport maps are smooth, invertible functions that can transform one probability distribution into another [marzouk2016](@cite).
The mathematical foundation builds on the Rosenblatt transformation [rosenblatt1952](@cite) and the Knothe-Rosenblatt rearrangement [knothe1957](@cite).
They are particularly useful for:

- **Sampling**: Generate samples from complex distributions by transforming samples from simple distributions
- **Variational inference**: Approximate complex posterior distributions in Bayesian updating problems [grashorn2024](@cite)
- **Density estimation**: Learn the structure of complex probability distributions

## Key Features

- **Polynomial Maps**: Triangular polynomial transport maps
- **Adaptive Construction**: Automatic selection of polynomial terms for efficient approximation
- **Multiple Rectifiers**: Support for different activation functions (Softplus, ShiftedELU, Identity)
- **Quadrature Integration**: Multiple quadrature schemes for map optimization
- **Optimization**: Built-in optimization routines for fitting maps to target densities
- **Multithreaded evaluation** for processing multiple points efficiently

## Installation

```julia
using Pkg
Pkg.add("https://github.com/lukasfritsch/TransportMaps.jl")
```

## Quick Start Example

Here's a simple example showing how to construct a transport map for a "banana" distribution:

```julia
using TransportMaps
using Distributions

# Create a 2D polynomial map with degree 2 and Softplus rectifier
M = PolynomialMap(2, 2, Normal(), Softplus())

# Set up quadrature for optimization
quadrature = GaussHermiteWeights(3, 2)

# Define target density (banana distribution)
target_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(target_density, :auto_diff)

# Optimize the map coefficients
result = optimize!(M, target, quadrature)

# Generate samples by mapping standard Gaussian samples
samples_z = randn(1000, 2)
# Matrix input automatically uses multithreading for better performance
mapped_samples = evaluate(M, samples_z)

# Evaluate map quality
variance_diag = variance_diagnostic(M, target, samples_z)
```

## Package Architecture

The package is organized around several key components:

### Map Components

- **`PolynomialMapComponent`**: Individual polynomial components of triangular maps
- **`HermiteBasis`**: Hermite polynomial basis functions
- **`MultivariateBasis`**: Multivariate polynomial basis construction

### Transport Maps

- **`PolynomialMap`**: Main triangular polynomial transport map implementation

### Rectifier Functions

- **`IdentityRectifier`**: No transformation (linear)
- **`Softplus`**: Smooth positive transformation
- **`ShiftedELU`**: Exponential linear unit variant

### Quadrature and Optimization

- **`GaussHermiteWeights`**: Gauss-Hermite quadrature points and weights
- **`MonteCarloWeights`**: Monte Carlo integration
- **`LatinHypercubeWeights`**: Latin hypercube sampling

## API Reference

```@contents
Pages = ["api.md"]
```

## Authors

- **Lukas Fritsch**, Institute for Risk and Reliability, Leibniz University Hannover
- **Jan Grashorn**, Chair for Engineering Materials and Building Preservation, Helmut-Schmidt-University Hamburg

## Related Implementation

- [ATM](https://github.com/baptistar/ATM): Matlab code for adaptive transport maps [baptista2023](@cite)
- [MParT](https://github.com/MeasureTransport/MParT): C++-based library for transport maps [parno2022](@cite)
- [TransportBasedInference.jl](https://github.com/mleprovost/TransportBasedInference.jl): Julia implementation of adaptive transport maps (ATM) and Kalman filters
- [SequentialMeasureTransport.jl](https://github.com/benjione/SequentialMeasureTransport.jl): Julia implementation of transport maps from sum-of-squares densities [zanger2024](@cite)
- [Triangular-Transport-Toolbox](https://github.com/MaxRamgraber/Triangular-Transport-Toolbox): Python code for the triangular transport tutorial paper [ramgraber2025](@cite)
