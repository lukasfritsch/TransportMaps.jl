# # Banana Distribution Example
#
# This example demonstrates how to use TransportMaps.jl to approximate
# a "banana" distribution using polynomial transport maps.
#
# The banana distribution is a common test case in transport map literature [marzouk2016](@cite),
# defined as a standard normal in the first dimension and a normal distribution
# centered at x₁² in the second dimension. This example showcases the effectiveness
# of triangular transport maps for capturing nonlinear dependencies [baptista2023](@cite).

# We start with the necessary packages:

using TransportMaps
using Distributions
using Plots

# ### Creating the Transport Map
#
# We start by creating a 2-dimensional polynomial transport map with degree 2
# and a Softplus rectifier function.

M = PolynomialMap(2, 2, Normal(), Softplus())

# ### Setting up Quadrature
#
# For optimization, we need to specify quadrature weights. Here we use
# Gauss-Hermite quadrature with 3 points per dimension.

quadrature = GaussHermiteWeights(3, 2)

# ### Defining the Target Density
#
# The banana distribution has the density:
# ```math
# p(x) = \phi(x_1) \cdot \phi(x_2 - x_1^2)
# ```
# where $\phi$ is the standard normal PDF.

target_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)
nothing # hide

# Create a MapTargetDensity object for optimization
target = MapTargetDensity(target_density, :auto_diff)

# ### Optimizing the Map
#
# Now we optimize the map coefficients to approximate the target density:

@time res = optimize!(M, target, quadrature)
println("Optimization result: ", res)

# ### Testing the Map
#
# Let's generate some samples from the standard normal distribution
# and map them through our optimized transport map:

samples_z = randn(1000, 2)

# Map the samples through our transport map:
mapped_samples = evaluate(M, samples_z)

# ### Visualizing Results
#
# Let's create a scatter plot of the mapped samples to see how well
# our transport map approximates the banana distribution:

scatter(mapped_samples[:, 1], mapped_samples[:, 2],
           label="Mapped Samples", alpha=0.5, color=2,
           title="Transport Map Approximation of Banana Distribution",
           xlabel="x₁", ylabel="x₂")
#md savefig("samples-banana.svg"); nothing # hide
# ![Banana Samples](samples-banana.svg)

# ### Quality Assessment
#
# We can assess the quality of our approximation using the variance diagnostic:

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

# ### Interpretation
#
# The variance diagnostic provides a measure of how well the transport map
# approximates the target distribution. Lower values indicate better approximation.
#
# The scatter plot should show the characteristic "banana" shape, with samples
# curved according to the relationship x₂ ≈ x₁².

# ### Further Experiments
#
# You can experiment with:
# - Different polynomial degrees (see [baptista2023](@cite) for monotone map theory)
# - Different rectifier functions (`IdentityRectifier()`, `ShiftedELU()`)
# - Different quadrature methods (`MonteCarloWeights`, `LatinHypercubeWeights`)
# - More quadrature points for higher accuracy
