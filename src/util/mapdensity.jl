struct MapTargetDensity{F,G} <: AbstractMapDensity
    density::F #! should be changed to logdensity for numerical stability
    gradient_type::Symbol
    grad_density::G #! should be changed to logdensity gradient for numerical stability

    #! take logdensity instead of density for numerical stability
    function MapTargetDensity(logdensity::F, grad_density::G) where {F,G}
        return new{F,G}(logdensity, :analytical, grad_density)
    end

    function MapTargetDensity(logdensity::F, gradient_type::Symbol, grad_density::G) where {F,G}

        if gradient_type ∉ [:analytical, :analytic]
            throw(ArgumentError("gradient_type must be :analytical (or :analytic) when providing a custom gradient function."))
        end

        return new{F,G}(logdensity, gradient_type, grad_density)
    end

    function MapTargetDensity(logdensity::F, gradient_type::Symbol) where {F}
        # Check gradient type and set gradient function accordingly
        if gradient_type ∈ [:auto_diff, :autodiff, :ad, :automatic, :forward_diff, :forwarddiff]
            grad_density = x -> ForwardDiff.gradient(logdensity, x)
            gradient_type = :auto_diff
        elseif gradient_type ∈ [:finite_difference, :finitedifference, :finite_diff, :finitediff, :fd, :numerical, :numeric]
            grad_density = x -> central_difference_gradient(logdensity, x)
            gradient_type = :finite_difference
        else
            throw(ArgumentError("gradient_type must be either :auto_diff (or :autodiff, :ad,
            :automatic, :forward_diff) or :finite_difference (or :fd, :finite_diff, :finitediff, :numerical)."))
        end

        return new{F,typeof(grad_density)}(logdensity, gradient_type, grad_density)
    end
end

#! maybe also replace density with logdensity here for numerical stability
struct MapReferenceDensity{F, G} <: AbstractMapDensity
    density::F
    gradient_type::Symbol
    grad_density::G
    densitytype::Distributions.UnivariateDistribution

    # base constructor for reference density, directly using automatic differentiation
    function MapReferenceDensity(densitytype::Distributions.UnivariateDistribution=Normal())
        density = x -> prod(map(Base.Fix1(pdf, densitytype), x))
        gradient = x -> ForwardDiff.gradient(density, x)
        return new{typeof(density), typeof(gradient)}(density, :auto_diff, gradient, densitytype)
    end

    function MapReferenceDensity(densitytype::Distributions.Uniform)
        error("Not implemented yet")
    end
end

gradient(density::AbstractMapDensity, x::AbstractArray{<:Real}) = density.grad_density(x)

function Base.show(io::IO, target::MapTargetDensity)
    print(io, "MapTargetDensity(gradient_type=:$(target.gradient_type)) ")
end

function Base.show(io::IO, ref::MapReferenceDensity)
    print(io, "MapReferenceDensity(density=$(ref.densitytype)")
end
