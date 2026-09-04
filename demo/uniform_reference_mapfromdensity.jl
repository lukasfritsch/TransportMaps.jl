using TransportMaps
using Distributions
using Plots
using Random
using Statistics

Random.seed!(72)

reference = Uniform(0, 1)
target_distribution = Normal()
target = MapTargetDensity(x -> logpdf(target_distribution, x[1]))

M = PolynomialMap(
    1,
    5,
    reference,
    Softplus(),
    ShiftedLegendreBasis(),
)
quadrature = GaussLegendreWeights(5, M)

result = optimize!(M, target, quadrature; δ = 1.0e-9)

reference_samples = reshape(rand(reference, 2000), :, 1)
target_samples = evaluate(M, reference_samples)

println("KL divergence:       ", result.minimum)
println("Mapped sample mean:  ", mean(target_samples))
println("Mapped sample std:   ", std(target_samples))

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

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
