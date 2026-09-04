# # Uniform Reference: Map from Samples
#
# This example learns a transport map from samples of a beta distribution to a
# uniform reference distribution. It demonstrates that sample-based map
# construction uses the reference density supplied to `PolynomialMap`, rather
# than assuming a standard normal reference.

using TransportMaps
using Distributions
using Plots
using Random
using Statistics

Random.seed!(72)

# ## Generate target samples
#
# We use a `Beta(2, 2)` target because its distribution is visibly non-uniform
# while having the same bounded support as the reference distribution.

target = Beta(2, 2)
target_samples = reshape(rand(target, 300), :, 1)

# ## Construct and optimize the map
#
# `Uniform(0, 1)` is the map's reference density. A cubic map is sufficient for
# this simple one-dimensional example. The bounded reference automatically uses
# a constrained optimizer that keeps all mapped training samples strictly inside
# the support of the uniform density.

M = PolynomialMap(
    1,
    3,
    Uniform(0, 1),
    Softplus(),
    ShiftedLegendreBasis(),
)
result = optimize!(M, target_samples)
#md result.optimization_results[1]

# The forward direction maps target samples into reference space:

reference_samples = evaluate(M, target_samples)
@assert all(0 .< reference_samples .< 1)

println("Mapped sample range: ", extrema(reference_samples))
println("Mapped sample mean:  ", mean(reference_samples))
println("Mapped sample std:   ", std(reference_samples))

# For comparison, a `Uniform(0, 1)` random variable has mean `0.5` and standard
# deviation `1 / sqrt(12)`, approximately `0.289`.
#
# The inverse direction turns fresh reference samples into approximate target
# samples:

uniform_samples = reshape(rand(Uniform(0, 1), 1000), :, 1)
generated_samples = inverse(M, uniform_samples)

# ## Visualize the result

x = range(0, 1, length = 200)

target_plot = histogram(
    target_samples[:, 1];
    bins = 25,
    normalize = :pdf,
    alpha = 0.45,
    label = "Training samples",
    xlabel = "x",
    ylabel = "density",
    title = "Target space",
)
histogram!(
    target_plot,
    generated_samples[:, 1];
    bins = 25,
    normalize = :pdf,
    alpha = 0.45,
    label = "Generated with inverse(M, z)",
)
plot!(target_plot, x, pdf.(target, x); linewidth = 2, label = "Beta(2, 2)")

reference_plot = histogram(
    reference_samples[:, 1];
    bins = 20,
    normalize = :pdf,
    alpha = 0.6,
    label = "evaluate(M, x)",
    xlabel = "z",
    ylabel = "density",
    xlims = (0, 1),
    title = "Uniform reference space",
)
plot!(reference_plot, [0, 1], [1, 1]; linewidth = 2, label = "Uniform(0, 1)")

plot(target_plot, reference_plot; layout = (1, 2), size = (900, 350))
#md savefig("uniform-reference-mapfromsamples.svg"); nothing # hide
# ![The learned map in target and reference space](uniform-reference-mapfromsamples.svg)

# The left panel checks the generative direction of the map, while the right
# panel shows that the fitted forward map approximately uniformizes the target
# samples.
