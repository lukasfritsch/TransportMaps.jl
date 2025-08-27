# # Map from Samples Example
#
# This example demonstrates how to use TransportMaps.jl to approximate
# a "banana" distribution using polynomial transport maps when only samples
# from the target distribution are available.
#
# Unlike the density-based approach, this method learns the transport map
# directly from sample data using optimization techniques. This is particularly
# useful when the target density is unknown or difficult to evaluate [marzouk2016](@cite).

# We start with the necessary packages:

using TransportMaps
using Distributions
using LinearAlgebra
using Plots

# ### Generating Target Samples
#
# The banana distribution has the density:
# ```math
# p(x) = \phi(x_1) \cdot \phi(x_2 - x_1^2)
# ```
# where $\phi$ is the standard normal PDF.

banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

# Set up the log-target function for sampling:
num_samples = 500
nothing # hide

# Generate samples using rejection sampling (no external dependencies)
function generate_banana_samples(n_samples::Int)
    samples = Matrix{Float64}(undef, n_samples, 2)

    count = 0
    while count < n_samples
        x1 = randn() * 2
        x2 = randn() * 3 + x1^2

        if rand() < banana_density([x1, x2]) / 0.4
            count += 1
            samples[count, :] = [x1, x2]
        end
    end

    return samples
end

println("Generating samples from banana distribution...")
target_samples = generate_banana_samples(num_samples)
println("Generated $(size(target_samples, 1)) samples")

# ### Creating the Transport Map
#
# We create a 2-dimensional polynomial transport map with degree 2.
# For sample-based optimization, we typically start with lower degrees
# and can increase complexity as needed.

M = PolynomialMap(2, 2, :normal, Softplus())

# ### Optimizing from Samples
#
# The key difference from density-based optimization is that we optimize
# directly from the sample data without requiring the density function. Inside the optimization the map is arranged s.t. the "forward" direction is from the (unknown) target distribution to the standard normal distribution:

res = optimize!(M, target_samples)
println("Optimization result: ", res)

# ### Testing the Map
#
# Let's generate new samples from the banana density and standard normal samples and map them through our optimized transport map to verify the learned distribution:

new_samples = generate_banana_samples(1000)
norm_samples = randn(1000, 2)
# Map the samples through our transport map. Note that `evaluate` now transports from reference to target, i.e. `mapped_samples` should be standard normal samples:
mapped_samples = evaluate(M, new_samples)
# while pushing from the standard normal samples to the target distribution generates new samples from the banana distribution:
mapped_banana_samples = inverse(M, norm_samples)

# ### Visualizing Results
#
# Let's create a scatter plot comparing the original samples with
# the mapped samples to see how well our transport map learned the distribution:

p11 = scatter(new_samples[:, 1], new_samples[:, 2],
            label="Original Samples", alpha=0.5, color=1,
            title="Original Banana Distribution Samples",
            xlabel="x₁", ylabel="x₂")

scatter!(p11, mapped_banana_samples[:, 1], mapped_banana_samples[:, 2],
            label="Mapped Samples", alpha=0.5, color=2,
            title="Transport Map Generated Samples",
            xlabel="x₁", ylabel="x₂")

plot(p11, size=(800, 400))
#md savefig("samples-comparison-target.svg"); nothing # hide
# ![Sample Comparison](samples-comparison-target.svg)

# and the resulting samples in standard normal space:

p12 = scatter(norm_samples[:, 1], norm_samples[:, 2],
            label="Original Samples", alpha=0.5, color=1,
            title="Original Banana Distribution Samples",
            xlabel="x₁", ylabel="x₂")

scatter!(p12, mapped_samples[:, 1], mapped_samples[:, 2],
            label="Mapped Samples", alpha=0.5, color=2,
            title="Transport Map Generated Samples",
            xlabel="x₁", ylabel="x₂")

plot(p12, size=(800, 400))
#md savefig("samples-comparison-reference.svg"); nothing # hide
# ![Sample Comparison](samples-comparison-reference.svg)

# ### Density Comparison
#
# We can also compare the learned density (via pullback) with the true density:

x₁ = range(-3, 3, length=100)
x₂ = range(-2.5, 4.0, length=100)

# True banana density values:
true_density = [banana_density([x1, x2]) for x2 in x₂, x1 in x₁]

# Learned density via pullback through the transport map. Note that "pullback" computes the density of the mapped samples in the standard normal space:
learned_density = [pullback(M, [x1, x2]) for x2 in x₂, x1 in x₁]

# Create contour plots for comparison:
p3 = contour(x₁, x₂, true_density,
            title="True Banana Density",
            xlabel="x₁", ylabel="x₂",
            colormap=:viridis, levels=10)

p4 = contour(x₁, x₂, learned_density,
            title="Learned Density (Pullback)",
            xlabel="x₁", ylabel="x₂",
            colormap=:viridis, levels=10)

plot(p3, p4, layout=(1, 2), size=(800, 400))
#md savefig("density-comparison.svg"); nothing # hide
# ![Density Comparison](density-comparison.svg)

# ### Combined Visualization
#
# Finally, let's create a combined plot showing both the original samples
# and the density contours:

scatter(target_samples[:, 1], target_samples[:, 2],
        label="Original Samples", alpha=0.3, color=1,
        xlabel="x₁", ylabel="x₂",
        title="Banana Distribution: Samples and Learned Density")

contour!(x₁, x₂, learned_density./maximum(learned_density),
        levels=5, colormap=:viridis, alpha=0.8,
        label="Learned Density Contours")

xlims!(-3, 3)
ylims!(-2.5, 4.0)
#md savefig("combined-result.svg"); nothing # hide
# ![Combined Result](combined-result.svg)

# ### Quality Assessment
#
# We can assess the quality of our sample-based approximation by comparing
# statistics of the original and mapped samples:

println("Sample Statistics Comparison:")
println("Original samples - Mean: ", Distributions.mean(target_samples, dims=1))
println("Original samples - Std:  ", Distributions.std(target_samples, dims=1))
println("Mapped samples - Mean:   ", Distributions.mean(mapped_banana_samples, dims=1))
println("Mapped samples - Std:    ", Distributions.std(mapped_banana_samples, dims=1))

# ### Interpretation
#
# The sample-based approach learns the transport map by fitting to the
# empirical distribution of the samples. This method is particularly useful when:
# - The target density is unknown or expensive to evaluate
# - Only sample data is available from experiments or simulations
# - The distribution is complex and difficult to express analytically
#
# The quality of the approximation depends on:
# - The number and quality of the original samples
# - The polynomial degree of the transport map
# - The optimization algorithm and convergence criteria

# ### Further Experiments
#
# You can experiment with:
# - Different polynomial degrees for more complex distributions
# - Different rectifier functions (`Softplus()`, `ShiftedELU()`)
# - More sophisticated MCMC sampling strategies
# - Cross-validation techniques to assess generalization
# - Different sample sizes to study convergence behavior
