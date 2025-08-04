# # Getting Started with TransportMaps.jl
#
# This guide will help you get started with TransportMaps.jl for constructing and using transport maps.
#
# ## Basic Concepts
#
# ### What is a Transport Map?
#
# A transport map $T$ is a function that transforms samples from a reference distribution (typically standard Gaussian) to a target distribution [marzouk2016](@cite). The key property is that if $X \sim \rho_0$ (reference) and $Y = T(X)$, then $Y \sim \rho_1$ (target).

# ### Triangular Maps

# TransportMaps.jl focuses on **triangular transport maps** [baptista2023](@cite), where:

# ```math
# T(\boldsymbol{x}) =
# \left(\begin{array}{c}
# T_1(x_1) \\
# T_2(x_1, x_2) \\
# T_3(x_1, x_2, x_3) \\
# \vdots \\
# T_n(x_1, x_2 \dots, x_n)
# \end{array}
# \right)
# ```
#
# This structure ensures that the map is invertible and the Jacobian determinant is easy to compute.
# The construction follows the Knothe-Rosenblatt rearrangement [knothe1957](@cite).

# ## First Example: A Simple 2D Transport Map

using TransportMaps
using Distributions
using Random
using Plots
using LinearAlgebra

# Let's create a simple 2D transport map:

# Set random seed for reproducibility
Random.seed!(1234)
nothing # hide

# Create a 2D polynomial map with degree 2
M = PolynomialMap(2, 2, Softplus())

# The map is initially identity (coefficients are zero)
println("Initial coefficients: ", getcoefficients(M))

# ### Defining a Target Distribution
#
# For optimization, you need to define your target probability density.
# Let's start with a simple correlated Gaussian:

# Example: Correlated Gaussian
function correlated_gaussian(x; ρ=0.8)
    Σ = [1.0 ρ; ρ 1.0]
    return pdf(MvNormal(zeros(2), Σ), x)
end
nothing # hide

# Create a MapTargetDensity object for optimization
target_density = MapTargetDensity(correlated_gaussian, :auto_diff)

# ### Setting up Quadrature
#
# Choose an appropriate quadrature scheme for map optimization:

# Gauss-Hermite quadrature (good for Gaussian-like targets)
quadrature = GaussHermiteWeights(5, 2)  # 5 points per dimension, 2D

# Alternative options (commented out):
# quadrature = MonteCarloWeights(1000, 2)  # 1000 samples, 2D
# quadrature = LatinHypercubeWeights(1000, 2)

# ### Optimizing the Map
#
# Fit the transport map to your target distribution:

println("Optimizing the map...")
@time result = optimize!(M, target_density, quadrature)

println("Optimization result: ", result)
println("Final coefficients: ", getcoefficients(M))

# ### Generating Samples
#
# Once optimized, use the map to generate samples:

# Generate reference samples (standard Gaussian)
n_samples = 1000
reference_samples = randn(n_samples, 2)

# Transform to target distribution
target_samples = zeros(n_samples, 2)
for i in 1:n_samples
    target_samples[i, :] = evaluate(M, reference_samples[i, :])
end

# ### Visualizing Results
#
# Let's plot both the reference and target samples:

p1 = scatter(reference_samples[:, 1], reference_samples[:, 2],
            alpha=0.6, title="Reference Samples",
            xlabel="Z₁", ylabel="Z₂", legend=false, aspect_ratio=:equal)

p2 = scatter(target_samples[:, 1], target_samples[:, 2],
            alpha=0.6, title="Target Samples",
            xlabel="X₁", ylabel="X₂", legend=false, aspect_ratio=:equal)

plot(p1, p2, layout=(1,2), size=(800, 400))
#md savefig("samples.svg"); nothing # hide
# ![Transport Map Samples](samples.svg)

# ### Evaluating Map Quality
#
# Check how well your map approximates the target:

# Variance diagnostic (should be close to 1 for good maps)
var_diag = variance_diagnostic(M, target_density, reference_samples)
println("Variance diagnostic: ", var_diag)

# You can also check the Jacobian determinant
sample_point = [0.0, 0.0]
jac = jacobian(M, sample_point)
det_jac = det(jac)
println("Jacobian determinant at origin: ", det_jac)

# ### Working with Different Rectifiers
#
# The rectifier function affects the map's behavior. Let's compare different options:

# ShiftedELU rectifier
M_elu = PolynomialMap(2, 2, ShiftedELU())
result_elu = optimize!(M_elu, target_density, quadrature)
var_diag_elu = variance_diagnostic(M_elu, target_density, reference_samples)

println("Variance diagnostics:")
println("  Softplus: ", var_diag)
println("  ShiftedELU: ", var_diag_elu)

# ## More Complex Example: Banana Distribution
#
# Now let's try a more challenging target - the banana distribution:

# Define banana density
banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)
target_density_banana = MapTargetDensity(banana_density, :auto_diff)

# Create a new map for this target
M_banana = PolynomialMap(2, 2, Softplus())
result_banana = optimize!(M_banana, target_density_banana, quadrature)

# Display optimized map
display(M_banana)

# Generate samples
banana_samples = zeros(n_samples, 2)
for i in 1:n_samples
    banana_samples[i, :] = evaluate(M_banana, reference_samples[i, :])
end

# Visualize the banana distribution
x1_grid = range(-3, 3, length=100)
x2_grid = range(-3, 6, length=100)
posterior_values = [banana_density([x₁, x₂]) for x₂ in x2_grid, x₁ in x1_grid]

scatter(banana_samples[:, 1], banana_samples[:, 2],
        alpha=0.6, title="Banana Distribution Samples",
        xlabel="X₁", ylabel="X₂", legend=false, aspect_ratio=:equal)
contour!(x1_grid, x2_grid, posterior_values, colormap=:viridis, label="Posterior Density")
#md savefig("banana_samples.svg"); nothing # hide
# ![Banana Distribution Samples](banana_samples.svg)

# Check quality
var_diag_banana = variance_diagnostic(M_banana, target_density_banana, reference_samples)
println("Banana distribution variance diagnostic: ", var_diag_banana)
