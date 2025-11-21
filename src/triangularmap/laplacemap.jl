struct LaplaceMap <: AbstractLinearMap
    mode::Vector{Float64}  # Mode / mean vector
    chol::Matrix{Float64}  # Cholesky decomposition of the covariance matrix

    # Construct a LaplaceMap from sample data
    function LaplaceMap(samples::Matrix{Float64})
        mode = mean(samples, dims=1)[:]
        Σ = cov(samples, corrected=true)
        # Cholesky decomposition (Σ = L * L')
        chol = cholesky(Σ).L
        return new(mode, chol)
    end

    # Compute a Laplace approximation given a target density and initial guess x0
    function LaplaceMap(
        density::MapTargetDensity,
        x0::Vector{Float64};
        hessian_type::Symbol = :auto_diff,      # type of Hessian computation (:auto_diff or :finite_difference)
        optimizer::Optim.AbstractOptimizer = LBFGS(),
        options::Optim.Options = Optim.Options()
    )

        # objective: f(x) = -log(π(x))
        function obj(x)
            return -logpdf(density, x)
        end

        # gradient: f'(x) = - ∇(log(π(x))) (chain rule)
        function grad!(storage, x)
            storage .= -grad_logpdf(density, x)
            return storage
        end

        # Optimize to find the mode
        res = optimize(obj, grad!, x0, optimizer, options)

        if Optim.converged(res)
            mode = Optim.minimizer(res)
        else
            error("LaplaceMap optimization did not converge.")
        end

        # Compute Hessian at mode
        if density.gradient_type ∈ [:auto_diff, :analytical] && hessian_type ∈ [:auto_diff, :autodiff, :ad, :automatic, :forward_diff, :forwarddiff]
            H = ForwardDiff.hessian(obj, mode)
        else
            H = central_difference_hessian(obj, mode)
        end

        # Make matrix Hermitian to avoid numerical issues
        Σ = Hermitian(inv(H))
        chol = cholesky(Σ).L

        return new(mode, chol)
    end
end

function evaluate(L::LaplaceMap, x::Vector{Float64})
    @assert length(x) == length(L.mode) "Input vector must have the same length as dimensions in the map"
    return L.chol \ (x .- L.mode)
end


function evaluate(L::LaplaceMap, X::Matrix{Float64})
    @assert size(X, 2) == length(L.mode) "Input data must have the same number of columns as dimensions in the map"
    return (X .- L.mode') * inv(L.chol)'
end

function inverse(L::LaplaceMap, y::Vector{Float64})
    @assert length(y) == length(L.mode) "Input vector must have the same length as dimensions in the map"
    return L.chol * y .+ L.mode
end

function inverse(L::LaplaceMap, Y::Matrix{Float64})
    @assert size(Y, 2) == length(L.mode) "Input data must have the same number of columns as dimensions in the map"
    return Y * L.chol' .+ L.mode'
end

function jacobian(L::LaplaceMap)
    return abs(det(L.chol))
end

numberdimensions(L::LaplaceMap) = length(L.mode)

cov(L::LaplaceMap) = L.chol * L.chol'

mean(L::LaplaceMap) = L.mode

mode(L::LaplaceMap) = L.mode

MvNormal(L::LaplaceMap) = Distributions.MvNormal(mean(L), cov(L))

# Make LaplaceMap callable: L(x) instead of evaluate(L, x)
Base.@propagate_inbounds (L::LaplaceMap)(x::AbstractVector{<:Real}) = evaluate(L, x)
Base.@propagate_inbounds (L::LaplaceMap)(X::AbstractMatrix{<:Real}) = evaluate(L, X)

function Base.show(io::IO, L::LaplaceMap)
    print(io, "LaplaceMap($(numberdimensions(L))-dimensional")
    print(io, " mode: ", mode(L), ", ")
    print(io, " Σ: ", cov(L), ")")
end

function Base.show(io::IO, mime::MIME"text/plain", L::LaplaceMap)
    println(io, "LaplaceMap with $(numberdimensions(L)) dimensions")
    println(io, "  mode: ", mode(L))
    println(io, "  Σ: ", cov(L))
end
