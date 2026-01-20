"""
    Softplus(β::Float64=1.0)

Smooth rectifier function: `g(ξ) = log(1 + exp(βξ)) / β`. Ensures monotonicity
with smooth approximation to ReLU. Higher β values create sharper transitions.
"""
struct Softplus <: AbstractRectifierFunction
    β::Float64
    Softplus(β::Float64=1.0) = new(β)
end

"""
    (r::Softplus)(ξ)

Evaluate the Softplus rectifier at `ξ`.
"""
function (r::Softplus)(ξ)
    return 1 / r.β   * log1p(exp(r.β * ξ))
end

"""
    derivative(r::Softplus, ξ)

Compute the derivative of Softplus: `g'(ξ) = 1/(1 + exp(-βξ))` (sigmoid function).
"""
function derivative(r::Softplus, ξ)
    return 1.0 / (1.0 + exp(-r.β * ξ))  # sigmoid function
end

"""
    ShiftedELU()

Rectifier combining exponential and linear behavior: `g(ξ) = exp(ξ)` for `ξ ≤ 0`,
`g(ξ) = ξ + 1` for `ξ > 0`. Ensures monotonicity with different behavior for
negative and positive inputs.
"""
struct ShiftedELU <: AbstractRectifierFunction
end

"""
    (r::ShiftedELU)(ξ)

Evaluate the ShiftedELU rectifier at `ξ`.
"""
function (r::ShiftedELU)(ξ)
    return ξ <= 0 ? exp(ξ) : ξ + 1
end

"""
    derivative(r::ShiftedELU, ξ)

Compute the derivative of ShiftedELU: `g'(ξ) = exp(ξ)` for `ξ ≤ 0`, `g'(ξ) = 1` for `ξ > 0`.
"""
function derivative(r::ShiftedELU, ξ)
    return ξ <= 0 ? exp(ξ) : 1.0
end

"""
    IdentityRectifier()

No-op rectifier: `g(ξ) = ξ`. Does not ensure monotonicity! Only use when
partial derivatives are guaranteed to be positive by other means.
"""
struct IdentityRectifier <: AbstractRectifierFunction
end

"""
    (r::IdentityRectifier)(ξ)

Evaluate the identity rectifier at `ξ` (returns `ξ` unchanged).
"""
function (r::IdentityRectifier)(ξ)
    return ξ
end

"""
    derivative(r::IdentityRectifier, ξ)

Compute the derivative of IdentityRectifier: `g'(ξ) = 1`.
"""
function derivative(r::IdentityRectifier, ξ)
    return 1.0
end

"""
    ExpRectifier()

Exponential rectifier: `g(ξ) = exp(ξ)`. Ensures strict positivity and monotonicity.
Can lead to extreme values for large |ξ|.
"""
struct ExpRectifier <: AbstractRectifierFunction
end

"""
    (r::ExpRectifier)(ξ)

Evaluate the exponential rectifier at `ξ`.
"""
function (r::ExpRectifier)(ξ)
    return exp.(ξ)
end

"""
    derivative(r::ExpRectifier, ξ)

Compute the derivative of ExpRectifier: `g'(ξ) = exp(ξ)`.
"""
function derivative(r::ExpRectifier, ξ)
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
