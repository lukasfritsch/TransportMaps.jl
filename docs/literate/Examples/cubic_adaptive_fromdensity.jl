# # Cubic: Adaptive Transport Map from Density

# This example demonstrates the adaptive map construction from
# a given density $\pi(x)$. The target is given by the unnormalized joint density
# ```math
# p(x) = \phi(x_1) \cdot \phi(x_2 - x_1^3).
# ```

# This density is similar to the banana density, however, it has a cubic instead of a
# quadratic term in the second term of the product.

using TransportMaps
using Distributions
using LinearAlgebra
using Plots

#md using Random # hide
#md Random.seed!(123)
#md nothing # hide

# Define the target density and `MapTargetDensity` object
target_density(x) = logpdf(Normal(0, 0.5), x[1]) + logpdf(Normal(0, 0.1), x[2] - x[1]^3)

target = MapTargetDensity(target_density, :ad)

# As quadrature method, we use Smolyak integration
quadrature = SparseSmolyakWeights(3, 2)

# Then we perform adaptive map construction
T, hist = optimize_adaptive_transportmap(target, quadrature, 10;
    validation = LatinHypercubeWeights(100, 2))

# Generate samples and compare response
samples_z = randn(2000, 2)
mapped_samples = evaluate(T, samples_z)

# Compare computed map with the target pdf
x1 = -2:0.01:2
x2 = -2:0.01:2

pdf_val = [pdf(target, [x₁, x₂]) for x₂ in x2, x₁ in x1]

s = scatter(mapped_samples[:, 1], mapped_samples[:, 2],
    label="Mapped Samples", alpha=0.5, color=1,
    xlabel="x₁", ylabel="x₂", title="Target Density and Mapped Samples")

contour!(x1, x2, pdf_val)
#md savefig("cubic-density.svg"); nothing # hide
# ![Cubic density: TM samples and density contour](cubic-density.svg)

# Convergence plots
convergence_kl = plot(hist.train_objectives, label="Train Objective", xlabel="Iteration",
    ylabel="KL divergence", title="Objective Value vs Iteration", marker=:o)
plot!(convergence_kl, hist.test_objectives, label="Test Objective", marker=:o)
yaxis!(:log10)

grad_norms = maximum.(hist.gradients[2:end])

convergence_grad = plot(2:length(hist.gradients), grad_norms; xlabel="Iteration", ylabel="Maximum Gradient",
    label=nothing, marker=:o, title="Gradient")
yaxis!(:log10)
xlims!(xlims(convergence_kl))

plot(convergence_kl, convergence_grad, layout=(2, 1))
#md savefig("cubic-density-convergence.svg"); nothing # hide
# ![Cubic density: Convergence](cubic-density-convergence.svg)

# Visualize the multi-index sets of both map components
ind_atm = getmultiindexsets(T[1])
MIS1 = scatter(ind_atm[:, 1], zeros(length(ind_atm)), ms=30, legend=false)
plot!(xlims=(-0.5, maximum(ind_atm[:, 1]) + 0.5), ylims=(-0.5, 0.5),
    aspect_ratio=1, xlabel="Multi-index α₁", ylabel="", title="Multi-indices Component 1")
xticks!(0:maximum(ind_atm[:, 1]))
yticks!(0:0)

ind_atm = getmultiindexsets(T[2])
MIS2 = scatter(ind_atm[:, 1], ind_atm[:, 2], ms=30, legend=false)
plot!(xlims=(-0.5, maximum(ind_atm[:, 1]) + 0.5), ylims=(-0.5, maximum(ind_atm[:, 2]) + 0.5),
    aspect_ratio=1, xlabel="Multi-index α₁", ylabel="Multi-index α₂", title="Multi-indices Component 2")
xticks!(0:maximum(ind_atm[:, 1]))
yticks!(0:maximum(ind_atm[:, 2]))

plot(MIS1, MIS2, layout=(2, 1))
#md savefig("cubic-density-terms.svg"); nothing # hide
# ![Cubic density: Terms in the multi-index sets of the two map components](cubic-density-terms.svg)
