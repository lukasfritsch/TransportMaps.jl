struct HermiteBasis <: AbstractPolynomialBasis
    edge_control::Symbol
    bounds_linearization::Vector{Float64}
    normalization::Vector{Float64}

    function HermiteBasis(edge_control::Symbol=:none; lower_bound_linearization=-Inf, upper_bound_linearization=Inf, normalization=Float64[])
        if !(edge_control in (:none, :gaussian, :cubic, :linearized))
            throw(ArgumentError("edge_control must be :none, :gaussian, :cubic, or :linearized"))
        end
        return new(edge_control, [lower_bound_linearization, upper_bound_linearization], normalization)
    end
end

# Initialize normalization for linearized Hermite basis
LinearizedHermiteBasis(max_degree::Int) = HermiteBasis(:linearized, normalization=ones(max_degree+1))

# Set quantile bounds and normalization for linearized Hermite basis
function LinearizedHermiteBasis(samples::Vector{<:Real}, max_degree::Int, k::Int)
    lower_bound, upper_bound = quantile(samples, 0.01), quantile(samples, 0.99)
    normalization = Vector{Float64}(undef, max_degree+1)
    for n in 0:max_degree
        normalization[n+1] = n == k ? factorial(n+1) : factorial(n)
    end
    return HermiteBasis(:linearized, lower_bound_linearization=lower_bound, upper_bound_linearization=upper_bound, normalization=normalization)
end

# Construct LinearizedHermiteBasis from an analytical univariate distribution
function LinearizedHermiteBasis(density::Distributions.UnivariateDistribution, max_degree::Int, k::Int)
    lower_bound, upper_bound = quantile(density, 0.01), quantile(density, 0.99)
    normalization = Vector{Float64}(undef, max_degree+1)
    for n in 0:max_degree
        normalization[n+1] = n == k ? factorial(n+1) : factorial(n)
    end
    return HermiteBasis(:linearized, lower_bound_linearization=lower_bound, upper_bound_linearization=upper_bound, normalization=normalization)
end

# Gaussian-weighted Hermite basis
GaussianWeightHermiteBasis() = HermiteBasis(:gaussian)

function CubicSplineHermiteBasis(samples::Vector{<:Real})
    lower_bound, upper_bound = quantile(samples, 0.01), quantile(samples, 0.99)
    return HermiteBasis(:cubic, lower_bound_linearization=lower_bound, upper_bound_linearization=upper_bound)
end

# Construct CubicSplineHermiteBasis from an analytical univariate distribution
function CubicSplineHermiteBasis(density::Distributions.UnivariateDistribution)
    lower_bound, upper_bound = quantile(density, 0.01), quantile(density, 0.99)
    return HermiteBasis(:cubic, lower_bound_linearization=lower_bound, upper_bound_linearization=upper_bound)
end

# Univariate probabilist's Hermite polynomials
@inline function hermite_polynomial(n::Int, z::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return z
    else
        H_nm2 = 1.0
        H_nm1 = z
        for k in 2:n
            H_n = z * H_nm1 - (k - 1) * H_nm2
            H_nm2, H_nm1 = H_nm1, H_n
        end
        return H_nm1
    end
end

# Derivative of univariate Hermite polynomial
@inline function hermite_derivative(n::Int, z::Real)
    n == 0 ? 0.0 : n * hermite_polynomial(n - 1, z)
end

# Linearized Hermite polynomial (unnormalized)
function _linearized_hermite(n::Int, z::Real, bounds_linearization::Vector{Float64})
    lower = bounds_linearization[1]
    upper = bounds_linearization[2]
    if z < lower
        φ = hermite_polynomial(n, lower)
        dφ = hermite_derivative(n, lower)
        return φ + dφ * (z - lower)
    elseif z > upper
        φ = hermite_polynomial(n, upper)
        dφ = hermite_derivative(n, upper)
        return φ + dφ * (z - upper)
    else
        return hermite_polynomial(n, z)
    end
end

# Derivative of linearized Hermite polynomial
function _linearized_hermite_derivative(n::Int, z::Real, bounds_linearization::Vector{Float64})
    lower = bounds_linearization[1]
    upper = bounds_linearization[2]
    if z < lower
        dφ = hermite_derivative(n, lower)
        return dφ
    elseif z > upper
        dφ = hermite_derivative(n, upper)
        return dφ
    else
        return hermite_derivative(n, z)
    end
end

# Normalized edge-controlled Hermite polynomial
@inline function edge_controlled_hermite_polynomial(n::Int, z::Real, edge_control::Symbol, bounds_linearization=fill(-Inf, 2), normalization=Float64[])
    if edge_control == :linearized && isfinite(bounds_linearization[1]) && isfinite(bounds_linearization[2]) && !isempty(normalization)
        norm = normalization[n+1]
        return _linearized_hermite(n, z, bounds_linearization) / sqrt(norm)
    elseif edge_control == :gaussian
        weight = exp(-0.25 * z^2)
        return hermite_polynomial(n, z) * weight
    elseif edge_control == :cubic
        r = 2 * maximum(abs.(bounds_linearization))
        m = min(1.0, abs(z) / r)
        weight = 2 * m^3 - 3 * m^2 + 1
        return hermite_polynomial(n, z) * weight
    else
        return hermite_polynomial(n, z)
    end
end

# Derivative of the univariate Hermite polynomial with edge control
function edge_controlled_hermite_derivative(n::Int, z::Real, edge_control::Symbol, bounds_linearization=fill(-Inf, 2))
    if edge_control == :linearized && isfinite(bounds_linearization[1]) && isfinite(bounds_linearization[2])
        return _linearized_hermite_derivative(n, z, bounds_linearization)
    elseif edge_control == :gaussian
        return exp(-0.25 * z .^ 2) .* (n * hermite_polynomial(n - 1, z) - z / 2 .* hermite_polynomial(n, z))
    elseif edge_control == :cubic
        r = 2 * maximum(abs.(bounds_linearization))
        f(z) = begin
            m = min(1.0, abs.(z) / r)
            return 2 * m^3 - 3 * m^2 + 1
        end
        ∂f(z) = begin
            if abs.(z) < r
                m = abs.(z) / r
                return (6 / r) * (m .^ 2 - m) .* sign.(z)
            else
                return 0.0
            end
        end
        return hermite_derivative(n, z) .* f.(z) .+ hermite_polynomial(n, z) .* ∂f.(z)
    else
        return hermite_derivative(n, z)
    end
end

@inline function basisfunction(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
    return edge_controlled_hermite_polynomial(Int(αᵢ), zᵢ, basis.edge_control, basis.bounds_linearization, basis.normalization)
end

@inline function basisfunction_derivative(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
    n = Int(αᵢ)
    if basis.edge_control == :linearized
        norm = basis.normalization[n+1]
        return edge_controlled_hermite_derivative(n, zᵢ, :linearized, basis.bounds_linearization) / sqrt(norm)
    else
        return edge_controlled_hermite_derivative(n, zᵢ, basis.edge_control)
    end
end

# Display methods for HermiteBasis
function Base.show(io::IO, basis::HermiteBasis)
    print(io, "HermiteBasis(edge_control=:$(basis.edge_control)")
    if basis.edge_control == :linearized && isfinite(basis.bounds_linearization[1]) && isfinite(basis.bounds_linearization[2])
        print(io, ", bounds_linearization=$(basis.bounds_linearization)")
    end
    print(io, ")")
end
