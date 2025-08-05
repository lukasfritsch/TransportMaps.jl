
# Example from
# Sullivan, A. B., Snyder, D. M., & Rounds, S. A. (2010). Controls on biochemical oxygen demand in the upper Klamath River, Oregon. Chemical Geology, 269(1), 12–21. https://doi.org/10.1016/j.chemgeo.2009.08.007

# Also used in marzouk2016 as example for TMs

#=
 The model is a simple exponential growth model for biochemical oxygen demand (BOD) in a river system.
The parameters \(x_1\) and \(x_2\) control the growth rate
and the decay rate, respectively. The model is defined as:
\begin{equation}
\mathcal{B}(t) = A(1-\exp(-Bt))+\mathcal{E},
\end{equation}

where
\begin{split}\begin{aligned}
\mathcal{E} & \sim \mathcal{N}(0,1e-3)\\
A & = \left[0.4 + 0.4\left(1 + \text{erf}\left(\frac{\theta_1}{\sqrt{2}} \right)\right) \right]\\
B & = \left[0.01 + 0.15\left(1 + \text{erf}\left(\frac{\theta_2}{\sqrt{2}} \right)\right) \right]
\end{aligned}\end{split}

=#

using TransportMaps
using Plots
using Distributions
using SpecialFunctions


# Define the model
function forward_model(t, x)
    A = 0.4 + 0.4 * (1 + erf(x[1] / sqrt(2)))
    B = 0.01 + 0.15 * (1 + erf(x[2] / sqrt(2)))
    return A * (1 - exp(-B * t))
end

# Data
t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1e-3)

# Plot the data
s = scatter(t, D, label="Data", xlabel="Time (t)", ylabel="Biochemical Oxygen Demand (D)",
    size=(600, 400), legend=:topleft)

# Plot model output for some parameter values
t_values = range(0, 5, length=100)
for x₁ in [-0.5, 0, 0.5]
    for x₂ in [-0.5, 0, 0.5]
        plot!(t_values, [forward_model(ti, [x₁, x₂]) for ti in t_values], label="(x₁ = $x₁, x₂ = $x₂)", linestyle=:dash)
    end
end

display(s)


#=
We define the posterior distribution using the prior (standard normal)
and the likelihood based on the model output and observed data.
Here, we assume the likelihood is Gaussian with a small variance \sigma^2 and independent observations.

Thus, the likelihood is given by:
\begin{equation}
\pi(\mathbf{y}|\boldsymbol{\theta}) = \prod_{t=1}^{5} \pi(y_t|\boldsymbol{\theta}),
\end{equation}

with
$\pi(\mathbf{y}_t|\boldsymbol{\theta})=\frac{1}{\sqrt{0.002.\pi}}\exp \left(-\frac{1}{0.002} \left(y_t - \mathcal{B}(t)\right)^2 \right), t \in \{1,...,5\}.$

=#

function posterior(x)
    # Calculate the likelihood
    likelihood = prod([pdf(Normal(forward_model(t[k], x), σ), D[k]) for k in 1:5])
    # Calculate the prior
    prior = pdf(Normal(0,1), x[1]) * pdf(Normal(0,1), x[2])
    return prior * likelihood
end

target = MapTargetDensity(posterior, :auto_diff)

# Define TransportMap
M = PolynomialMap(2, 3, :normal, Softplus())

quadrature = GaussHermiteWeights(10, M)

# Optimize the map coefficients
res = optimize!(M, target, quadrature)
println(res)

# Test mapping
samples_z = randn(1000, 2)

# Map the samples
mapped_samples = evaluate(M, samples_z)

# Compute the variance Diagnostic
var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

# Plot mapped samples along with the posterior density
x₁ = range(-.5, 1.5, length=100)
x₂ = range(-.5, 3, length=100)

posterior_values = [posterior([x₁, x₂]) for x₂ in x₂, x₁ in x₁]  # swap x₁ and x₂ order

scatter(mapped_samples[:, 1], mapped_samples[:, 2], label="Mapped Samples", alpha=0.5, color=1)
contour!(x₁, x₂, posterior_values; colormap = :viridis,
label="Posterior Density", xlabel="x₁", ylabel="x₂", title="Posterior Density and Mapped Samples")
