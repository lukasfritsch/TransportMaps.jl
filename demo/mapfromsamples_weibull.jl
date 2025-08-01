using TransportMaps
using Distributions
using Plots

wb = [Weibull(1.0, 2.0), Weibull(1.0, 2.0)]
wbsamples = hcat(rand.(wb, 1000)...)

M = PolynomialMap(2, 2, Softplus())
quadrature = GaussHermiteWeights(3, 2)

target = TargetDensity(x -> pdf(Normal(),x[1]).*pdf(Normal(),x[2]),:auto_diff)

# Optimize the map coefficients
@time res = optimize!(M, target, wbsamples)