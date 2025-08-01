# TransportMaps.jl Documentation

```@meta
CurrentModule = TransportMaps
```

## Overview

This implementation is based on the adaptive transport map framework. For a comprehensive introduction to transport maps, see [marzouk2016](@cite) and [ramgraber2025](@cite). The theoretical foundations for monotone triangular transport maps are detailed in [baptista2023](@cite). For practical applications in structural health monitoring and Bayesian inference, see [grashorn2024](@cite). Related software implementations can be found in [parno2022](@cite).l is a pure Julia implementation of transport maps for probability density transformation and sampling. This package provides tools for constructing polynomial transport maps that can transform samples from simple reference distributions (e.g., standard Gaussian) to complex target distributions.

## What are Transport Maps?

Transport maps are smooth, invertible functions that can transform one probability distribution into another [marzouk2016](@cite). They are particularly useful for:

- **Sampling**: Generate samples from complex distributions by transforming samples from simple distributions
- **Density estimation**: Learn the structure of complex probability distributions
- **Uncertainty quantification**: Efficient sampling and analysis of complex parameter spaces [grashorn2024](@cite)
- **Variational inference**: Approximate complex posterior distributions

The mathematical foundation builds on the Rosenblatt transformation [rosenblatt1952](@cite) and the Knothe-Rosenblatt rearrangement [knothe1957](@cite), with modern developments enabling practical implementation for high-dimensional problems [baptista2023](@cite).

## Key Features

- **Polynomial Maps**: Triangular polynomial transport maps with various basis functions
- **Adaptive Construction**: Automatic selection of polynomial terms for efficient approximation
- **Multiple Rectifiers**: Support for different activation functions (Softplus, ShiftedELU, Identity)
- **Quadrature Integration**: Multiple quadrature schemes for map optimization
- **Optimization**: Built-in optimization routines for fitting maps to target densities

## Installation

```julia
using Pkg
Pkg.add("TransportMaps")
```

## Quick Start Example

Here's a simple example showing how to construct a transport map for a "banana" distribution:

```julia
using TransportMaps
using Distributions

# Create a 2D polynomial map with degree 2 and Softplus rectifier
M = PolynomialMap(2, 2, Softplus())

# Set up quadrature for optimization
quadrature = GaussHermiteWeights(3, 2)

# Define target density (banana distribution)
target_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

# Optimize the map coefficients
result = optimize!(M, target_density, quadrature)

# Generate samples by mapping standard Gaussian samples
samples_z = randn(1000, 2)
mapped_samples = reduce(vcat, [evaluate(M, x)' for x in eachrow(samples_z)])

# Evaluate map quality
variance_diag = variance_diagnostic(M, target_density, samples_z)
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

## Examples

For more detailed examples, see the Examples section in the sidebar.

## References

This implementation is based on the adaptive transport map framework. For theoretical background, see:

1. Youssef Marzouk and Tarek Moselhy. "Bayesian inference with optimal maps." Journal of Computational Physics 231.23 (2012): 7815-7850.
2. Matthew Parno and Youssef Marzouk. "Transport map accelerated Markov chain Monte Carlo." SIAM Journal on Scientific Computing 40.2 (2018): A1340-A1358.

## Contributing

Contributions are welcome! Please see the GitHub repository for issues and contribution guidelines.