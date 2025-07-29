# Implementation of various quadrature rules for numerical integration
# Todo: Add more flexible reference density (so far: only Gaussian)
# Todo: Add more quadrature rules

struct GaussHermiteWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function GaussHermiteWeights(numberpoints::Int64, dimension::Int64)
        points, weights = gausshermite_weights(numberpoints, dimension)
        return new(points, weights)
    end
end

function gausshermite_weights(numberpoints::Int64, dimension::Int64)
    # Tensor product Gauss-Hermite quadrature
    x1d, w1d = gausshermite(numberpoints; normalize=true)

    # Generate tensor product indices
    indices = collect(Iterators.product(ntuple(_ -> 1:numberpoints, dimension)...))

    # Allocate arrays for points and weights
    points = Matrix{Float64}(undef, dimension, length(indices))
    weights = Vector{Float64}(undef, length(indices))

    for (k, idx) in enumerate(indices)
        points[:, k] = [x1d[i] for i in idx]
        weights[k] = prod(w1d[i] for i in idx)
    end

    return points, weights
end

struct MonteCarloWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function MonteCarloWeights(numberpoints::Int64, dimension::Int64)
        points, weights = montecarlo_weights(numberpoints, dimension)
        return new(points, weights)
    end
end

function montecarlo_weights(numberpoints::Int64, dimension::Int64)
    points = randn(dimension, numberpoints)
    weights = 1/numberpoints*ones(numberpoints)
    return points, weights
end

struct LatinHypercubeWeights <: AbstractQuadratureWeights
    points::Matrix{Float64}
    weights::Vector{Float64}

    function LatinHypercubeWeights(n::Int64, d::Int64)
        points, weights = latinhypercube_weights(n, d)
        return new(points, weights)
    end
end

function latinhypercube_weights(numberpoints::Int64, dimension::Int64)
    points = [quantile(Normal(), u) for u in QuasiMonteCarlo.sample(numberpoints, dimension, LatinHypercubeSample())]
    weights = 1/numberpoints*ones(numberpoints)
    return points, weights
end
