using TransportMaps
using Distributions
using LinearAlgebra
using Plots

banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

num_samples = 500

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

M = PolynomialMap(2, 2, :normal, Softplus())

res = optimize!(M, target_samples)

new_samples = generate_banana_samples(1000)
norm_samples = randn(1000, 2)

mapped_samples = evaluate(M, new_samples)

mapped_banana_samples = inverse(M, norm_samples)

p11 = scatter(new_samples[:, 1], new_samples[:, 2],
            label="Original Samples", alpha=0.5, color=1,
            title="Original Banana Distribution Samples",
            xlabel="x₁", ylabel="x₂")

scatter!(p11, mapped_banana_samples[:, 1], mapped_banana_samples[:, 2],
            label="Mapped Samples", alpha=0.5, color=2,
            title="Transport Map Generated Samples",
            xlabel="x₁", ylabel="x₂")

plot(p11, size=(800, 400))

p12 = scatter(norm_samples[:, 1], norm_samples[:, 2],
            label="Original Samples", alpha=0.5, color=1,
            title="Original Banana Distribution Samples",
            xlabel="x₁", ylabel="x₂")

scatter!(p12, mapped_samples[:, 1], mapped_samples[:, 2],
            label="Mapped Samples", alpha=0.5, color=2,
            title="Transport Map Generated Samples",
            xlabel="x₁", ylabel="x₂")

plot(p12, size=(800, 400))

x₁ = range(-3, 3, length=100)
x₂ = range(-2.5, 4.0, length=100)

true_density = [banana_density([x1, x2]) for x2 in x₂, x1 in x₁]

learned_density = [pullback(M, [x1, x2]) for x2 in x₂, x1 in x₁]

p3 = contour(x₁, x₂, true_density,
            title="True Banana Density",
            xlabel="x₁", ylabel="x₂",
            colormap=:viridis, levels=10)

p4 = contour(x₁, x₂, learned_density,
            title="Learned Density (Pullback)",
            xlabel="x₁", ylabel="x₂",
            colormap=:viridis, levels=10)

plot(p3, p4, layout=(1, 2), size=(800, 400))

scatter(target_samples[:, 1], target_samples[:, 2],
        label="Original Samples", alpha=0.3, color=1,
        xlabel="x₁", ylabel="x₂",
        title="Banana Distribution: Samples and Learned Density")

contour!(x₁, x₂, learned_density./maximum(learned_density),
        levels=5, colormap=:viridis, alpha=0.8,
        label="Learned Density Contours")

xlims!(-3, 3)
ylims!(-2.5, 4.0)

println("Sample Statistics Comparison:")
println("Original samples - Mean: ", Distributions.mean(target_samples, dims=1))
println("Original samples - Std:  ", Distributions.std(target_samples, dims=1))
println("Mapped samples - Mean:   ", Distributions.mean(mapped_banana_samples, dims=1))
println("Mapped samples - Std:    ", Distributions.std(mapped_banana_samples, dims=1))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
