struct RadialBasis <: AbstractPolynomialBasis
    centers::Vector{Float64}   # μ_i
    scales::Vector{Float64}    # σ_i (per-center)
    kernel::Symbol

    function RadialBasis(centers::Vector{<:Real}, scales::Vector{<:Real}; kernel::Symbol = :gaussian)
        if length(centers) != length(scales)
            throw(ArgumentError("centers and scales must have the same length"))
        end
        if !(kernel in (:gaussian,))
            throw(ArgumentError("unsupported kernel; only :gaussian is implemented"))
        end
        return new(Float64.(centers), Float64.(scales), kernel)
    end
end

# Convenience constructor: equally spaced centers and scales in [-range_radius, range_radius]
function RadialBasis(num_rbfs::Int; range_radius::Real=3.0, kernel::Symbol=:gaussian)
    @assert num_rbfs >= 1 "num_rbfs must be at least 1"
    centers = collect(range(-range_radius, range_radius, length=num_rbfs))
    # scales: average distance to neighbours (constant for equally spaced)
    if num_rbfs == 1
        scales = [range_radius]
    else
        d = centers[2] - centers[1]
        scales = fill(d, num_rbfs)
    end
    return RadialBasis(centers, scales; kernel=kernel)
end

# Default placeholder constructor: empty centers/scales. The real number of RBFs
# will be set by the map/polynomial component (e.g., degree+1) when needed.
RadialBasis() = RadialBasis(Float64[], Float64[])

# Construct centers and scales from empirical samples using quantiles
function RadialBasis(samples::Vector{<:Real}, num_rbfs::Int; kernel::Symbol=:gaussian)
    @assert num_rbfs >= 1 "num_rbfs must be at least 1"
    # choose centers at quantiles q_{i/(j+1)}, i=1..j
    qs = [(i) / (num_rbfs + 1) for i in 1:num_rbfs]
    centers = [quantile(samples, q) for q in qs]

    # compute per-center scales sigma_i by averaging distances to neighbouring centers
    scales = Vector{Float64}(undef, num_rbfs)
    for i in 1:num_rbfs
        if num_rbfs == 1
            # fallback to sample std if only one center
            scales[i] = std(samples)
        elseif i == 1
            right = centers[2] - centers[1]
            scales[i] = right
        elseif i == num_rbfs
            left = centers[end] - centers[end-1]
            scales[i] = left
        else
            left = centers[i] - centers[i-1]
            right = centers[i+1] - centers[i]
            scales[i] = 0.5 * (left + right)
        end
        # guard against zero scale
        if scales[i] <= 0.0 || !isfinite(scales[i])
            scales[i] = maximum([eps(Float64), std(samples)])
        end
    end

    return RadialBasis(centers, scales; kernel=kernel)
end

# Construct centers and scales from an analytical univariate distribution (use quantile function)
function RadialBasis(density::Distributions.UnivariateDistribution, num_rbfs::Int; kernel::Symbol=:gaussian)
    @assert num_rbfs >= 1 "num_rbfs must be at least 1"
    qs = [(i) / (num_rbfs + 1) for i in 1:num_rbfs]
    centers = [quantile(density, q) for q in qs]

    # estimate scales from distribution variance and neighbor distances
    scales = Vector{Float64}(undef, num_rbfs)
    # helper to safely get a std-like scale from the distribution
    function _density_std(d::Distributions.UnivariateDistribution)
        try
            return sqrt(var(d))
        catch
            return 1.0
        end
    end

    for i in 1:num_rbfs
        if num_rbfs == 1
            scales[i] = _density_std(density)
        elseif i == 1
            scales[i] = centers[2] - centers[1]
        elseif i == num_rbfs
            scales[i] = centers[end] - centers[end-1]
        else
            scales[i] = 0.5 * ((centers[i] - centers[i-1]) + (centers[i+1] - centers[i]))
        end
        if scales[i] <= 0.0 || !isfinite(scales[i])
            scales[i] = maximum([eps(Float64), _density_std(density)])
        end
    end

    return RadialBasis(centers, scales; kernel=kernel)
end

## Kernel and derivative (Gaussian on local coordinate)
@inline function _xloc(z::Real, mu::Real, sigma::Real)
    return (z - mu) / (sqrt(2pi) * sigma)
end

@inline function _rbf_gaussian(xloc::Real)
    return exp(-0.5 * xloc^2)
end

@inline function _rbf_gaussian_derivative(xloc::Real, sigma::Real)
    # derivative w.r.t. z: d/dz exp(-0.5 xloc^2) = -xloc * exp(-0.5 xloc^2) * d xloc/dz
    # where d xloc/dz = 1 / (sqrt(2pi) * sigma)
    dxloc_dz = 1.0 / (sqrt(2pi) * sigma)
    return -xloc * _rbf_gaussian(xloc) * dxloc_dz
end

@inline function basisfunction(basis::RadialBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    idx = n + 1
    if idx < 1 || idx > length(basis.centers)
        throw(ArgumentError("RadialBasis index αᵢ=$αᵢ out of bounds (centers length=$(length(basis.centers)))"))
    end
    mu = basis.centers[idx]
    sigma = basis.scales[idx]
    xloc = _xloc(zᵢ, mu, sigma)
    return _rbf_gaussian(xloc)
end

@inline function basisfunction_derivative(basis::RadialBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    idx = n + 1
    if idx < 1 || idx > length(basis.centers)
        throw(ArgumentError("RadialBasis index αᵢ=$αᵢ out of bounds (centers length=$(length(basis.centers)))"))
    end
    mu = basis.centers[idx]
    sigma = basis.scales[idx]
    xloc = _xloc(zᵢ, mu, sigma)
    return _rbf_gaussian_derivative(xloc, sigma)
end

# Display methods
function Base.show(io::IO, basis::RadialBasis)
    print(io, "RadialBasis(kernel=:$(basis.kernel), centers=$(length(basis.centers)))")
end

function Base.show(io::IO, ::MIME"text/plain", basis::RadialBasis)
    println(io, "RadialBasis:")
    println(io, "  Kernel: $(basis.kernel)")
    println(io, "  Number of centers: $(length(basis.centers))")
    println(io, "  Centers: $(basis.centers)")
    println(io, "  Scales: $(basis.scales)")
end
