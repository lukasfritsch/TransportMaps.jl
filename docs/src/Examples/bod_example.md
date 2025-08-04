```@meta
EditURL = "../../literate/bod_example.jl"
```

# Biochemical Oxygen Demand (BOD) Example

This example demonstrates Bayesian parameter estimation for a biochemical oxygen
demand model using transport maps. The problem comes from environmental engineering
and was originally presented in [sullivan2010](@cite) and later used as a benchmark
in transport map applications [marzouk2016](@cite).

The model describes the evolution of biochemical oxygen demand (BOD) in a river system
using an exponential growth model with two uncertain parameters controlling growth
and decay rates.

````@example bod_example
using TransportMaps
using Plots
using Distributions
using SpecialFunctions
````

## The Forward Model

The BOD model is given by:
```math
\mathcal{B}(t) = A(1-\exp(-Bt))+\mathcal{E}
```
where the parameters A and B are functions of the uncertain inputs θ₁ and θ₂:
```math
\begin{aligned}
A &= 0.4 + 0.4\left(1 + \text{erf}\left(\frac{\theta_1}{\sqrt{2}} \right)\right) \\
B &= 0.01 + 0.15\left(1 + \text{erf}\left(\frac{\theta_2}{\sqrt{2}} \right)\right)
\end{aligned}
```
and ℰ ~ 𝒩(0, 10⁻³) represents measurement noise.

````@example bod_example
function forward_model(t, x)
    A = 0.4 + 0.4 * (1 + erf(x[1] / sqrt(2)))
    B = 0.01 + 0.15 * (1 + erf(x[2] / sqrt(2)))
    return A * (1 - exp(-B * t))
end
#hide nothing
````

## Experimental Data

We have BOD measurements at five time points:

````@example bod_example
t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1e-3)
#hide nothing
````

Let's visualize the data along with model predictions for different parameter values:

````@example bod_example
s = scatter(t, D, label="Data", xlabel="Time (t)", ylabel="Biochemical Oxygen Demand (D)",
    size=(600, 400), legend=:topleft)
# Plot model output for some parameter values
t_values = range(0, 5, length=100)
for x₁ in [-0.5, 0, 0.5]
    for x₂ in [-0.5, 0, 0.5]
        plot!(t_values, [forward_model(ti, [x₁, x₂]) for ti in t_values],
              label="(x₁ = $x₁, x₂ = $x₂)", linestyle=:dash)
    end
end
savefig("realizations-bod.svg"); nothing # hide
````

![BOD Realizations](realizations-bod.svg)

## Bayesian Inference Setup

We define the posterior distribution using a standard normal prior on both parameters
and a Gaussian likelihood for the observations:
```math
\pi(\mathbf{y}|\boldsymbol{\theta}) = \prod_{t=1}^{5} \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(y_t - \mathcal{B}(t))^2\right)
```

````@example bod_example
function posterior(x)
    # Calculate the likelihood
    likelihood = prod([pdf(Normal(forward_model(t[k], x), σ), D[k]) for k in 1:5])
    # Calculate the prior
    prior = pdf(Normal(0,1), x[1]) * pdf(Normal(0,1), x[2])
    return prior * likelihood
end

target = MapTargetDensity(posterior, :auto_diff)
````

## Creating and Optimizing the Transport Map

We use a 2-dimensional polynomial transport map with degree 3 and Softplus rectifier:

````@example bod_example
M = PolynomialMap(2, 3, :normal, Softplus())
````

Set up Gauss-Hermite quadrature for optimization:

````@example bod_example
quadrature = GaussHermiteWeights(10, M)
````

Optimize the map coefficients:

````@example bod_example
@time res = optimize!(M, target, quadrature)
println("Optimization result: ", res)
````

## Generating Posterior Samples

Generate samples from the standard normal distribution and map them to the posterior:

````@example bod_example
samples_z = randn(1000, 2)
````

Map the samples through our transport map:

````@example bod_example
mapped_samples = reduce(vcat, [evaluate(M, z)' for z in eachrow(samples_z)])
````

## Quality Assessment

Compute the variance diagnostic to assess the quality of our approximation:

````@example bod_example
var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)
````

## Visualization

Plot the mapped samples along with contours of the true posterior density:

````@example bod_example
x₁ = range(-0.5, 1.5, length=100)
x₂ = range(-0.5, 3, length=100)

posterior_values = [posterior([x₁, x₂]) for x₂ in x₂, x₁ in x₁]

scatter(mapped_samples[:, 1], mapped_samples[:, 2],
        label="Mapped Samples", alpha=0.5, color=1,
        xlabel="x₁", ylabel="x₂", title="Posterior Density and Mapped Samples")
contour!(x₁, x₂, posterior_values, colormap=:viridis, label="Posterior Density")
savefig("samples-bod.svg"); nothing # hide
````

![BOD Samples](samples-bod.svg)

Finally, we can compute the pullback density to visualize how well our transport map approximates the posterior:

````@example bod_example
posterior_pullback = [pullback(M, [x₁, x₂]) for x₂ in x₂, x₁ in x₁]

contour(x₁, x₂, posterior_values./maximum(posterior_values);
    levels = 5, colormap = :viridis, colorbar = false,
    label="Target", xlabel="x₁", ylabel="x₂")
contour!(x₁, x₂, posterior_pullback./maximum(posterior_pullback);
    levels = 5, colormap = :viridis, linestyle=:dash,
    label="Pullback")
savefig("pullback-bod.svg"); nothing # hide
````

![BOD Pullback Density](pullback-bod.svg)

## Model Parameter Interpretation

The posterior samples provide uncertainty quantification for the BOD model parameters:
- **x₁**: Controls the maximum BOD level (parameter A)
- **x₂**: Controls the rate of BOD development (parameter B)

The correlation structure in the posterior reflects the interdependence between
these parameters in explaining the observed data.

## Further Analysis

You can extend this example by:
- Increasing the polynomial degree for higher accuracy
- Using different rectifier functions for improved monotonicity
- Adding more measurement data points
- Comparing with MCMC methods for validation
- Performing forward uncertainty propagation through the model

This example demonstrates the effectiveness of transport maps for Bayesian inference
in environmental modeling applications, providing efficient posterior sampling
for parameter estimation and uncertainty quantification.

