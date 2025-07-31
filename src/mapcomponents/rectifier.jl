struct Softplus <: AbstractRectifierFunction
end

function (r::Softplus)(ξ)
    return log1p(exp(ξ))  # log(1 + exp(ξ)) for numerical stability
end

struct ShiftedELU <: AbstractRectifierFunction
end

# Shifted exponential linear unit (ELU)
function (r::ShiftedELU)(ξ)
    return ξ <= 0 ? exp(ξ) : ξ + 1
end

struct IdentityRectifier <: AbstractRectifierFunction
end

function (r::IdentityRectifier)(ξ)
    return ξ
end

# Display methods for Softplus
function Base.show(io::IO, ::Softplus)
    print(io, "Softplus()")
end

function Base.show(io::IO, ::MIME"text/plain", ::Softplus)
    println(io, "Softplus:")
    println(io, "  Function: log(1 + exp(ξ))")
    println(io, "  Domain: ℝ")
    println(io, "  Range: (0, ∞)")
    println(io, "  Properties: Smooth approximation to ReLU, always positive")
    println(io, "  Derivative: σ(ξ) = 1/(1 + exp(-ξ)) (sigmoid)")
end

# Display methods for ShiftedELU
function Base.show(io::IO, ::ShiftedELU)
    print(io, "ShiftedELU()")
end

function Base.show(io::IO, ::MIME"text/plain", ::ShiftedELU)
    println(io, "ShiftedELU:")
    println(io, "  Function: ξ ≤ 0 ? exp(ξ) : ξ + 1")
    println(io, "  Domain: ℝ")
    println(io, "  Range: (0, ∞)")
    println(io, "  Properties: Exponential for negative inputs, linear + 1 for positive")
    println(io, "  Continuous and differentiable everywhere")
end

# Display methods for IdentityRectifier
function Base.show(io::IO, ::IdentityRectifier)
    print(io, "IdentityRectifier()")
end

function Base.show(io::IO, ::MIME"text/plain", ::IdentityRectifier)
    println(io, "IdentityRectifier:")
    println(io, "  Function: ξ")
    println(io, "  Domain: ℝ")
    println(io, "  Range: ℝ")
    println(io, "  Properties: No transformation, passes input unchanged")
    println(io, "  Warning: May result in non-monotonic transport maps")
end
