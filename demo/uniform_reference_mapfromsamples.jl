using TransportMaps
using Distributions
using Plots
using Random
using Statistics

Random.seed!(72)

target = Beta(2, 2)
target_samples = reshape(rand(target, 300), :, 1)

M = PolynomialMap(
    1,
    3,
    Uniform(0, 1),
    Softplus(),
    ShiftedLegendreBasis(),
)
result = optimize!(M, target_samples)

reference_samples = evaluate(M, target_samples)
@assert all(0 .< reference_samples .< 1)

println("Mapped sample range: ", extrema(reference_samples))
println("Mapped sample mean:  ", mean(reference_samples))
println("Mapped sample std:   ", std(reference_samples))

uniform_samples = reshape(rand(Uniform(0, 1), 1000), :, 1)
generated_samples = inverse(M, uniform_samples)

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

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
