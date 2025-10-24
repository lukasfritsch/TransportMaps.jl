struct MapTargetDensity <: AbstractMapDensity
    density::Function
    gradient_type::Symbol
    grad_density::Function

    MapTargetDensity(density::Function, grad_density::Function) = new(density, :analytical, grad_density)

    function MapTargetDensity(density::Function, gradient_type::Symbol, grad_density::Function)

        if gradient_type != :analytical
            throw(ArgumentError("gradient_type must be :analytical"))
        end
        return new(density, gradient_type, grad_density)
    end

    function MapTargetDensity(density::Function, gradient_type::Symbol)
        if gradient_type != :auto_diff && gradient_type != :finite_difference
            throw(ArgumentError("gradient_type must be either :auto_diff, or :finite_difference"))
        end
        # Define automatic differentiation gradient
        if gradient_type == :auto_diff
            grad_density = x -> ForwardDiff.gradient(density, x)
            # Define finite difference gradient
        elseif gradient_type == :finite_difference
            grad_density = x -> central_difference_gradient(density, x)
        end

        return new(density, gradient_type, grad_density)
    end
end

struct MapReferenceDensity <: AbstractMapDensity
    density::Function
    gradient_type::Symbol
    grad_density::Function
    densitytype::Distributions.UnivariateDistribution

    # base constructor for reference density, directly using automatic differentiation
    function MapReferenceDensity(densitytype::Distributions.UnivariateDistribution=Normal())
        density = x -> prod(map(Base.Fix1(pdf, densitytype), x))
        return new(density, :auto_diff, x -> ForwardDiff.gradient(density, x), densitytype)
    end

    function MapReferenceDensity(densitytype::Distributions.Uniform)
        error("Not implemented yet")
    end
end

gradient(density::AbstractMapDensity, x::AbstractArray{<:Real}) = density.grad_density(x)

function Base.show(io:: IO, target::MapTargetDensity)
    print(io, "MapTargetDensity(density=$(target.density), gradient_type=$(target.gradient_type))")
end

function Base.show(io:: IO, ref::MapReferenceDensity)
    print(io, "MapReferenceDensity(density=$(ref.densitytype)")
end
