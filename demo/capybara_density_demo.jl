using TransportMaps
using Distributions
using Plots

# Create a 2D polynomial transport map with degree 3 and Softplus rectifier
M = PolynomialMap(2, 3, Softplus())

# Set up quadrature for optimization
quadrature = GaussHermiteWeights(5, 2)

# Create a TargetDensity object for the capybara density
target = TargetDensity(capybara_density, :auto_diff)

# Optimize the map coefficients to approximate the capybara density
println("Optimizing transport map for capybara density...")
@time result = optimize!(M, target, quadrature)
println("Optimization result: ", result)

# Generate samples from standard normal and map them through our optimized transport map
println("Generating and mapping samples...")
samples_z = randn(2000, 2)

# Map the samples through our transport map
mapped_samples = reduce(vcat, [evaluate(M, x)' for x in eachrow(samples_z)])

# Create visualizations
println("Creating visualizations...")

# Plot 1: Mapped samples showing the capybara shape
scatter(mapped_samples[:, 1], mapped_samples[:, 2],
        label="Mapped Samples", alpha=0.5, color=2,
        title="Transport Map Approximation of Capybara Distribution",
        xlabel="x₁", ylabel="x₂", markersize=2)
savefig("capybara_samples.png")
println("Saved capybara_samples.png")

# Plot 2: Contour plot of the original capybara density
x_range = range(-3, 3, length=100)
y_range = range(-2.5, 2, length=100)
density_grid = [capybara_density([x, y]) for y in y_range, x in x_range]

contour(x_range, y_range, density_grid,
        title="Capybara Density Contours",
        xlabel="x₁", ylabel="x₂",
        fill=true, color=:viridis)
savefig("capybara_contours.png")
println("Saved capybara_contours.png")

# Quality assessment using variance diagnostic
var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

println("Demo completed successfully!")