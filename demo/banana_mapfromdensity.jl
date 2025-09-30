using TransportMaps
using Distributions
using Plots

M = PolynomialMap(2, 2, Normal(), Softplus())

quadrature = SparseSmolyakWeights(2, 2)

target_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

target = MapTargetDensity(target_density, :auto_diff)

res = optimize!(M, target, quadrature)
println("Optimization result: ", res)

samples_z = randn(1000, 2)

mapped_samples = evaluate(M, samples_z)

scatter(mapped_samples[:, 1], mapped_samples[:, 2],
           label="Mapped Samples", alpha=0.5, color=2,
           title="Transport Map Approximation of Banana Distribution",
           xlabel="x₁", ylabel="x₂")

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
