using TransportMaps
using Distributions
using Plots

M = PolynomialMap(2, 2)
quadrature = GaussHermiteWeights(3, 2)

banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(banana_density, :auto_diff)

# Optimize the map coefficients
@time res = optimize!(M, target, quadrature)
println(res)

# Test mapping
samples_z = randn(1000, 2)

# Map the samples
mapped_samples = evaluate(M, samples_z)

s = scatter(mapped_samples[:, 1], mapped_samples[:, 2], label="Mapped Samples", alpha=0.5, color=2)
display(s)

var_diag = variance_diagnostic(M, target, samples_z)
println("Variance Diagnostic: ", var_diag)
