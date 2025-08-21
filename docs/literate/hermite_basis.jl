# # Basis Functions
#
# This manual describes the different one-dimensional basis families currently
# implemented in TransportMaps.jl. We focus on variants of the (probabilists')
# Hermite polynomials and recently proposed edge-controlled versions
# [baptista2023](@cite), [ramgraber2025](@cite).

# ## Probabilistic Hermite Basis
#
# The probabilistic Hermite polynomials form an orthonormal basis with respect to
# the standard normal distribution. They satisfy the recurrence relation
# $\operatorname{He}_{n+1}(z)=z \operatorname{He}_n(z)-n \operatorname{He}_{n-1}(z)$ with
# $\operatorname{He}_0(z)=1$ and $\operatorname{He}_1(z)=z$.
#
# Construct the (unmodified) Hermite basis via `HermiteBasis(:none)`:
using Plots
using TransportMaps


basis = HermiteBasis(:none)
z = -7:0.1:7

p1 = plot()
for degree in 1:4
    plot!(p1, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
plot!(p1, xlabel="z", ylabel="Basis function", title="Standard Hermite Basis")
#md savefig("hermite_basis_standard.svg"); nothing # hide
# ![Standard Hermite Basis](hermite_basis_standard.svg)

# ## Linearized Hermite Basis
#
# Linearized Hermite polynomials (edge-linearized basis) were introduced in
# [baptista2023](@cite) to control growth for large |z| by replacing the
# polynomial with a tangent line outside data-dependent bounds $z^l,z^u$:
#
# ```math
# \mathcal{H}^{\mathrm{Lin}}_j(z)=\frac{1}{\sqrt{Z_{\alpha_j}}}
# \begin{cases}
# \mathrm{He}_j(z^l)+\mathrm{He}'_j(z^l)(z-z^l), & z< z^l \\
# \mathrm{He}_j(z), & z^l\le z \le z^u \\
# \mathrm{He}_j(z^u)+\mathrm{He}'_j(z^u)(z-z^u), & z> z^u
# \end{cases}
# ```
#
# The bounds are chosen here as the 0.01 and 0.99 empirical quantiles of a
# (reference) sample. The normalization constant $Z_{\alpha_j}$ follows the
# definition in the paper: $Z_{\alpha_j}=\alpha_j!$ for $j<k$ and
# $Z_{\alpha_k}=(\alpha_k+1)!$.
samples = randn(1_000); nothing # hide
basis = LinearizedHermiteBasis(samples, 4, 1)
println("Linearization bounds: ", basis.bounds_linearization)

p2 = plot()
for degree in 1:4
    plot!(p2, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
plot!(p2, xlabel="z", ylabel="Basis function", title="Linearized Hermite Basis")
#md savefig("hermite_basis_linearized.svg"); nothing # hide
# ![Linearized Hermite Basis](hermite_basis_linearized.svg)

# ## Edge-Controlled (Weighted) Hermite Basis: Gaussian Weight
#
# Edge control modifies each Hermite polynomial with a decaying weight to reduce
# growth in the tails [ramgraber2025](@cite). Using a Gaussian weight gives:
#
# ```math
# \mathcal{H}_j^{\text{Gauss}}(z)=\mathrm{He}_j(z)\exp\left(-\tfrac{z^2}{4}\right).
# ```
basis = HermiteBasis(:gaussian)

p3 = plot()
for degree in 1:4
    plot!(p3, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
plot!(p3, xlabel="z", ylabel="Basis function", title="Gaussian-Weighted Hermite Basis")
#md savefig("hermite_basis_gaussian.svg"); nothing # hide
# ![Gaussian Weighted Hermite Basis](hermite_basis_gaussian.svg)

# ## Edge-Controlled Hermite Basis: Cubic Spline Weight
#
# A cubic spline weight smoothly damps the polynomials outside a radius $r$, also introduced in [ramgraber2025](@cite).
# In this implementation, we define $r$ based on the 0.01 and 0.99 quantile values $z^l, z^u$:
#
# ```math
# \mathcal{H}_j^{\mathrm{Cub}}(z)=\operatorname{He}_j(z)\left(2 u^3-3 u^2+1\right),\qquad u=\min\!\left(1,\frac{|z|}{r}\right),\; r=2\max(|z^l|,|z^u|).
# ```
basis = CubicSplineHermiteBasis(samples)

p4 = plot()
for degree in 1:4
    plot!(p4, z, map(x -> basisfunction(basis, degree, x), z), label="degree $degree")
end
plot!(p4, xlabel="z", ylabel="Basis function", title="Cubic Spline Weighted Hermite Basis")
p4
#md savefig("hermite_basis_cubic.svg"); nothing # hide
# ![Cubic Spline Weighted Hermite Basis](hermite_basis_cubic.svg)

# ## Summary
#
# We showcased four basis variants:
# - Standard (orthonormal) Hermite
# - Linearized Hermite (piecewise linear tails)
# - Gaussian-weighted Hermite (exponential damping)
# - Cubic-spline-weighted Hermite (compact-style smooth damping)
#
# These can be supplied when constructing polynomial transport maps to tune
# stability, tail behavior, and sparsity characteristics.
