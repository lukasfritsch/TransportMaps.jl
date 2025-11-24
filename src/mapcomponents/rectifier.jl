struct Softplus <: AbstractRectifierFunction
    β::Float64
    Softplus(β::Float64=1.0) = new(β)
end

function (r::Softplus)(ξ)
    return 1 / r.β   * log1p(exp(r.β * ξ))
end

# Derivative of Softplus: d/dξ Softplus(ξ) = σ(βξ) = 1 / (1 + exp(-βξ))
function derivative(r::Softplus, ξ)
    return 1.0 / (1.0 + exp(-r.β * ξ))  # sigmoid function
end

struct ShiftedELU <: AbstractRectifierFunction
end

# Shifted exponential linear unit (ELU)
function (r::ShiftedELU)(ξ)
    return ξ <= 0 ? exp(ξ) : ξ + 1
end

# Derivative of ShiftedELU
function derivative(r::ShiftedELU, ξ)
    return ξ <= 0 ? exp(ξ) : 1.0
end

struct IdentityRectifier <: AbstractRectifierFunction
end

function (r::IdentityRectifier)(ξ)
    return ξ
end

# Derivative of IdentityRectifier
function derivative(r::IdentityRectifier, ξ)
    return 1.0
end

struct Exponential <: AbstractRectifierFunction
end

function (r::Exponential)(ξ)
    return exp.(ξ)
end

function derivative(r::Exponential, ξ)
    return exp.(ξ)
end

# Display methods for Softplus
function Base.show(io::IO, s::Softplus)
    print(io, "Softplus(β=$(s.β))")
end

function Base.show(io::IO, ::MIME"text/plain", s::Softplus)
    println(io, "Softplus:")
    println(io, "  Function: log(1 + exp(βξ)) / β")
    println(io, "  Parameter β: $(s.β)")
    println(io, "  Domain: ℝ")
    println(io, "  Range: (0, ∞)")
    println(io, "  Properties: Smooth approximation to ReLU, always positive")
    println(io, "  Derivative: σ(βξ) = 1/(1 + exp(-βξ)) (sigmoid)")
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
