using TransportMaps
using UncertaintyQuantification
using Distributions
using LinearAlgebra
using Plots

banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

num_samples = 5000
logtarget(df) = logpdf(Normal(), df.x1) .+ logpdf(Normal(), df.x2 - df.x1.^2)
tmcmc = TransitionalMarkovChainMonteCarlo(RandomVariable.(Uniform(-15, 15), [:x1, :x2]), num_samples, 10)
samples, evidence = bayesianupdating(logtarget, tmcmc)
samples_banana = Matrix(unique(samples))

M = PolynomialMap(2, 1)

# Optimize the map coefficients
@time res = optimize!(M, target_samples)

z = reduce(vcat, [TransportMaps.evaluate(M, x)' for x in eachrow(target_samples)])

x₁ = range(1.5, 2.5, length=100)
x₂ = range(0.5, 2.0, length=100)

posterior_values = [target_density([x₁, x₂]) for x₂ in x₂, x₁ in x₁]

scatter(target_samples[:, 1], target_samples[:, 2],
        label="Mapped Samples", alpha=0.5, color=2,
        xlabel="x₁", ylabel="x₂", title="Posterior Density and Mapped Samples")
contour!(x₁, x₂, posterior_values, colormap=:viridis, label="Posterior Density")

posterior_pullback = [pullback(M, [x₁, x₂]) for x₂ in x₂, x₁ in x₁]

contour(x₁, x₂, posterior_values./maximum(posterior_values);
    levels = 5, colormap = :viridis, colorbar = false,
    alpha=0.5, xlabel="x₁", ylabel="x₂")
contour!(x₁, x₂, posterior_pullback./maximum(posterior_pullback);
    levels = 5, colormap = :viridis, linestyle=:dash,
    alpha=0.5)
xlims!(1.5, 2.5)
ylims!(0.5, 1.5)
# scatter!(samples_banana[:, 1], samples_banana[:, 2],
        # label="Mapped Samples", alpha=0.1, color=2)