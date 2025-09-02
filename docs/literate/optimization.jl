
# # Optimization of the Map Coefficients
#
# A crucial step in constructing transport maps is the optimization of the map coefficients, which determine how well the map represents the target distribution. This process can be approached in two distinct ways:
#
# 1. **Map from Density**: Here, the transport map is constructed directly from a (possibly unnormalized) target density function. This approach requires access to the target density and typically uses quadrature schemes to approximate integrals. See [Banana: Map from Density](@ref) for an example.
#
# 2. **Map from Samples**: In this strategy, the transport map is built from a set of samples drawn from the target distribution. This is useful when the density is unknown or intractable, but samples are available. See [Banana: Map from Samples](@ref) for details.
#
# Both strategies utilize the same optimization routine, provided by [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/), and support a wide range of optimizers and options (such as convergence criteria and printing preferences).
#
# The main difference is in the inputs:
# - For **map-from-density**, you provide a target density function and quadrature points.
# - For **map-from-samples**, you only provide the sample array; no density or quadrature is needed.
#
# In both cases, you can specify the optimizer and its options. For a full overview of available options, see the [Optim.jl configuration documentation](https://julianlsolvers.github.io/Optim.jl/stable/user/config/).

# ### Map-from-density: trying different optimizers

# In the following, we demonstrate how to optimize the coefficients of a transport map using several different optimizers provided by `Optim.jl`. This allows us to compare their performance and convergence behavior on a simple example with a known target density.

# First we load the packages:
using TransportMaps
using Optim
using Distributions
using Plots

# Then, define the target density and quadrature scheme:
banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)
target = MapTargetDensity(banana_density, :auto_diff)
quadrature = GaussHermiteWeights(10, 2)
nothing #hide

# Set optimization options to print the trace every 20 iterations:
opts_trace = Optim.Options(iterations = 200, show_trace = true, show_every = 20, store_trace = true)

# We will try the following optimizers from `Optim.jl`:

# LBFGS (default, quasi-Newton)
M_lbfgs = PolynomialMap(2, 2, Normal(), Softplus(), LinearizedHermiteBasis())
res_lbfgs = optimize!(M_lbfgs, target, quadrature; optimizer = LBFGS(), options = opts_trace)
println(res_lbfgs)

# BFGS (quasi-Newton)
M_bfgs = PolynomialMap(2, 2, Normal(), Softplus(), LinearizedHermiteBasis())
res_bfgs = optimize!(M_bfgs, target, quadrature; optimizer = BFGS(), options = opts_trace)
println(res_bfgs)

# Conjugate Gradient
M_cg = PolynomialMap(2, 2, Normal(), Softplus(), LinearizedHermiteBasis())
res_cg = optimize!(M_cg, target, quadrature; optimizer = ConjugateGradient(), options = opts_trace)
println(res_cg)

# Gradient Descent
M_gd = PolynomialMap(2, 2, Normal(), Softplus(), LinearizedHermiteBasis())
res_gd = optimize!(M_gd, target, quadrature; optimizer = GradientDescent(), options = opts_trace)
println(res_gd)

# Nelder-Mead (derivative-free)
M_nm = PolynomialMap(2, 2, Normal(), Softplus(), LinearizedHermiteBasis())
res_nm = optimize!(M_nm, target, quadrature; optimizer = NelderMead(), options = opts_trace)
println(res_nm)

# Finally, we can compare the results by means of variance diagnostic:
samples_z = randn(1000, 2)
v_lbfgs = variance_diagnostic(M_lbfgs, target, samples_z)
v_bfgs = variance_diagnostic(M_bfgs, target, samples_z)
v_cg = variance_diagnostic(M_cg, target, samples_z)
v_gd = variance_diagnostic(M_gd, target, samples_z)
v_nm = variance_diagnostic(M_nm, target, samples_z)

println("Variance diagnostic LBFGS:             ", v_lbfgs)
println("Variance diagnostic BFGS:              ", v_bfgs)
println("Variance diagnostic CG:                ", v_cg)
println("Variance diagnostic GradientDescent:   ", v_gd)
println("Variance diagnostic NelderMead:        ", v_nm)

# We can visualize the convergence of all optimizers:
plot([res_lbfgs.trace[i].iteration for i in 1:length(res_lbfgs.trace)], lw=2,
     [res_lbfgs.trace[i].g_norm for i in 1:length(res_lbfgs.trace)], label="LBFGS")
plot!([res_bfgs.trace[i].iteration for i in 1:length(res_bfgs.trace)], lw=2,
     [res_bfgs.trace[i].g_norm for i in 1:length(res_bfgs.trace)], label="BFGS")
plot!([res_cg.trace[i].iteration for i in 1:length(res_cg.trace)], lw=2,
     [res_cg.trace[i].g_norm for i in 1:length(res_cg.trace)], label="CG")
plot!([res_gd.trace[i].iteration for i in 1:length(res_gd.trace)], lw=2,
     [res_gd.trace[i].g_norm for i in 1:length(res_gd.trace)], label="GradientDescent")
plot!([res_nm.trace[i].iteration for i in 1:length(res_nm.trace)], lw=2,
     [res_nm.trace[i].g_norm for i in 1:length(res_nm.trace)], label="NelderMead")
plot!(xaxis=:log, yaxis=:log, xlabel="Iteration", ylabel="Gradient norm",
    title="Convergence of different optimizers", xlims=(1, 200),
    legend=:bottomright)
#md savefig("optimization-conv.svg"); nothing # hide
# ![Optimization Convergence](optimization-conv.svg)

# It becomes clear, that LBFGS and BFGS are the most efficient optimizers in this case, while Nelder-Mead struggles to keep up.

# ### Map-from-samples: trying different optimizers
