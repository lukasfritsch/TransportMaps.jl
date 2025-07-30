using TransportMaps
using Distributions
using Plots

M = PolynomialMap(2, 2, Softplus())

quadrature = GaussHermiteWeights(3, 2)
target_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

# Optimize the map coefficients
@time res = optimize!(M, target_density, quadrature)
println(res)

# Test mapping
samples_z = randn(1000, 2)

# Map the samples
mapped_samples = reduce(vcat, [evaluate(M, x)' for x in eachrow(samples_z)])


s = scatter(mapped_samples[:, 1], mapped_samples[:, 2], label="Mapped Samples", alpha=0.5, color=2)
display(s)
# savefig(s, "mapped_samples.png")


var_diag = variance_diagnostic(M, target_density, samples_z)
println("Variance Diagnostic: ", var_diag)
