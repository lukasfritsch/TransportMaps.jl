"""
    MapTargetDensity

Wrapper for target density functions used in transport map optimization.
Stores the log-density function and its gradient, with support for automatic
differentiation, finite differences, or analytical gradients.

# Fields
- `logdensity::F`: Function computing log-density `log π(x)`
- `gradient_type::Symbol`: Type of gradient computation (`:analytical`, `:auto_diff`, or `:finite_difference`)
- `grad_logdensity::G`: Function computing gradient `∇ log π(x)`

# Constructors
- `MapTargetDensity(logdensity, grad_logdensity)`: Provide both log-density and analytical gradient.
- `MapTargetDensity(logdensity, :analytical, grad_logdensity)`: Explicitly specify analytical gradient.
- `MapTargetDensity(logdensity, :auto_diff)`: Use automatic differentiation for gradient.
- `MapTargetDensity(logdensity, :finite_difference)`: Use finite differences for gradient.
"""
struct MapTargetDensity{F,G} <: AbstractMapDensity
    logdensity::F
    gradient_type::Symbol
    grad_logdensity::G

    function MapTargetDensity(logdensity::F, grad_logdensity::G) where {F,G}
        return new{F,G}(logdensity, :analytical, grad_logdensity)
    end

    function MapTargetDensity(logdensity::F, gradient_type::Symbol, grad_logdensity::G) where {F,G}

        if gradient_type ∉ [:analytical, :analytic]
            throw(ArgumentError("gradient_type must be :analytical (or :analytic) when providing a custom gradient function."))
        end

        return new{F,G}(logdensity, :analytical, grad_logdensity)
    end

    function MapTargetDensity(logdensity::F, gradient_type::Symbol) where {F}
        # Check gradient type and set gradient function accordingly
        if gradient_type ∈ [:auto_diff, :autodiff, :ad, :automatic, :forward_diff, :forwarddiff]
            grad_logdensity = x -> ForwardDiff.gradient(logdensity, x)
            gradient_type = :auto_diff
        elseif gradient_type ∈ [:finite_difference, :finitedifference, :finite_diff, :finitediff, :fd, :numerical, :numeric]
            grad_logdensity = x -> central_difference_gradient(logdensity, x)
            gradient_type = :finite_difference
        else
            throw(ArgumentError("gradient_type must be either :auto_diff (or :autodiff, :ad,
            :automatic, :forward_diff) or :finite_difference (or :fd, :finite_diff, :finitediff, :numerical)."))
        end

        return new{F,typeof(grad_logdensity)}(logdensity, gradient_type, grad_logdensity)
    end
end

"""
    MapReferenceDensity

Wrapper for reference density (typically standard Gaussian) used in transport maps.
The reference density defines the space from which samples are drawn and mapped
to the target distribution.

# Fields
- `logdensity::F`: Function computing log-density `log ρ(z)`
- `gradient_type::Symbol`: Type of gradient computation (always `:auto_diff`)
- `grad_logdensity::G`: Function computing gradient `∇ log ρ(z)`
- `densitytype::Distributions.UnivariateDistribution`: Univariate density type (e.g., `Normal()`)

# Constructors
- `MapReferenceDensity()`: Use standard normal distribution (default).
- `MapReferenceDensity(densitytype::Distributions.UnivariateDistribution)`: Specify reference distribution.
"""
struct MapReferenceDensity{F,G} <: AbstractMapDensity
    logdensity::F
    gradient_type::Symbol
    grad_logdensity::G
    densitytype::Distributions.UnivariateDistribution

    # base constructor for reference density, directly using automatic differentiation
    function MapReferenceDensity(densitytype::Distributions.UnivariateDistribution=Normal())
        density = x -> sum(map(Base.Fix1(logpdf, densitytype), x))
        gradient = x -> ForwardDiff.gradient(density, x)
        return new{typeof(density),typeof(gradient)}(density, :auto_diff, gradient, densitytype)
    end

    function MapReferenceDensity(densitytype::Distributions.Uniform)
        error("Not implemented yet")
    end
end

"""
    logpdf(density::AbstractMapDensity, x)

Evaluate the log-density at point(s) `x`.

# Arguments
- `density::AbstractMapDensity`: Target or reference density
- `x`: Point (vector) or multiple points (matrix, rows are samples) at which to evaluate

# Returns
- Scalar log-density value for vector input, or vector of log-densities for matrix input
"""
logpdf(density::AbstractMapDensity, x::Vector{<:Real}) = density.logdensity(x)

logpdf(density::AbstractMapDensity, x::Real) = logpdf(density, [x])

function logpdf(density::AbstractMapDensity, X::Matrix{<:Real})
    n = size(X, 1)
    logdensities = zeros(Float64, n)

    Threads.@threads for i in 1:n
        logdensities[i] = density.logdensity(view(X, i, :))
    end
    return logdensities
end

"""
    grad_logpdf(density::AbstractMapDensity, x)

Evaluate the gradient of log-density at point(s) `x`.

# Arguments
- `density::AbstractMapDensity`: Target or reference density
- `x`: Point (vector) or multiple points (matrix, rows are samples) at which to evaluate

# Returns
- Gradient vector for vector input, or matrix of gradients (one per row) for matrix input
"""
grad_logpdf(density::AbstractMapDensity, x::Vector{<:Real}) = density.grad_logdensity(x)

function grad_logpdf(density::AbstractMapDensity, X::Matrix{<:Real})
    n, d = size(X)
    log_gradients = zeros(Float64, n, d)

    Threads.@threads for i in 1:n
        log_gradients[i, :] .= density.grad_logdensity(view(X, i, :))
    end
    return log_gradients
end

"""
    pdf(density::AbstractMapDensity, x)

Evaluate the probability density at point(s) `x`.

# Arguments
- `density::AbstractMapDensity`: Target or reference density
- `x`: Point (vector) or multiple points (matrix, rows are samples) at which to evaluate

# Returns
- Scalar density value for vector input, or vector of densities for matrix input

# Note
This computes `exp(logpdf(density, x))`. For numerical stability, prefer using `logpdf` when possible.
"""
pdf(density::AbstractMapDensity, x::Vector{<:Real}) = exp(density.logdensity(x))

pdf(density::AbstractMapDensity, x::Real) = pdf(density, [x])

function pdf(density::AbstractMapDensity, X::Matrix{<:Real})
    n = size(X, 1)
    densities = zeros(Float64, n)

    Threads.@threads for i in 1:n
        densities[i] = exp(density.logdensity(view(X, i, :)))
    end
    return densities
end

function Base.show(io::IO, target::MapTargetDensity)
    print(io, "MapTargetDensity(gradient_type=:$(target.gradient_type)) ")
end

function Base.show(io::IO, ref::MapReferenceDensity)
    print(io, "MapReferenceDensity(density=$(ref.densitytype)")
end
