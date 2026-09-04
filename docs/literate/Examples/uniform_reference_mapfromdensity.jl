# # Uniform Reference: Map from Density
#
# This example constructs a map from a uniform reference distribution when the
# target density is available. Unlike sample-based construction, the forward map
# directly transports reference samples into target space.

using TransportMaps
using Distributions
using Plots
using Random
using Statistics

Random.seed!(72)

# ## Define the target and reference
#
# We transport `Uniform(0, 1)` to a standard normal target. The exact transport
# is the normal quantile function, which gives us a useful comparison for the
# learned polynomial map.

reference = Uniform(0, 1)
target_distribution = Normal()
target = MapTargetDensity(x -> logpdf(target_distribution, x[1]))

# ## Construct and optimize the map
#
# The quadrature is constructed from `M`, ensuring that its points and weights
# represent the map's configured reference distribution.

M = PolynomialMap(
    1,
    5,
    reference,
    Softplus(),
    ShiftedLegendreBasis(),
)
quadrature = GaussLegendreWeights(5, M)

# `δ` controls the small stability perturbation used when evaluating the target
# density. Its default is `1e-9`; set `δ = 0` to optimize the exact KL objective.

result = optimize!(M, target, quadrature; δ = 1.0e-9)
#md result

# ## Check the transported samples

reference_samples = reshape(rand(reference, 2000), :, 1)
target_samples = evaluate(M, reference_samples)

println("KL divergence:       ", result.minimum)
println("Mapped sample mean:  ", mean(target_samples))
println("Mapped sample std:   ", std(target_samples))

# The mapped samples should have approximately zero mean and unit standard
# deviation.

# ## Visualize the result

x = range(-3.5, 3.5, length = 300)
sample_plot = histogram(
    target_samples[:, 1];
    bins = 35,
    normalize = :pdf,
    alpha = 0.55,
    label = "evaluate(M, z)",
    xlabel = "x",
    ylabel = "density",
    title = "Mapped target samples",
)
plot!(sample_plot, x, pdf.(target_distribution, x); linewidth = 2, label = "Normal(0, 1)")

# Since the exact one-dimensional map is known, we also compare the learned map
# with the normal quantile function away from its infinite endpoints.

z = range(0.005, 0.995, length = 250)
learned_quantiles = vec(evaluate(M, reshape(collect(z), :, 1)))
quantile_plot = plot(
    z,
    learned_quantiles;
    linewidth = 2,
    label = "Learned map",
    xlabel = "z",
    ylabel = "M(z)",
    title = "Uniform-to-normal transport",
)
plot!(
    quantile_plot,
    z,
    quantile.(target_distribution, z);
    linestyle = :dash,
    linewidth = 2,
    label = "Exact quantile",
)

plot(sample_plot, quantile_plot; layout = (1, 2), size = (900, 350))
#md savefig("uniform-reference-mapfromdensity.svg"); nothing # hide
# ![A density-based transport from a uniform reference](uniform-reference-mapfromdensity.svg)
