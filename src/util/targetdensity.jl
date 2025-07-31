struct TargetDensity <: AbstractTargetDensity
    density::Function
    gradient_type::Symbol
    grad_density::Function

    TargetDensity(density::Function, grad_density::Function) = new(density, :analytical, grad_density)

    function TargetDensity(density:: Function, gradient_type::Symbol, grad_density::Function)

        if gradient_type != :analytical
            throw(ArgumentError("gradient_type must be :analytical"))
        end
        return new(density, gradient_type, grad_density)
    end

    function TargetDensity(density::Function, gradient_type::Symbol)
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

gradient(target_density::TargetDensity, x::AbstractArray{<:Real}) = target_density.grad_density(x)

# Pretty printing for TargetDensity
function Base.show(io::IO, target::TargetDensity)
    print(io, "TargetDensity(gradient_type=:$(target.gradient_type))")
end
