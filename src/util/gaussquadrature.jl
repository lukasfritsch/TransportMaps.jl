# Gauss-Legendre quadrature to evaluate integral of rectifier

"""
    gaussquadrature(fun::Function, n::Int, a::Real, b::Real)

Numerically integrates the function `fun` over the interval `[a, b]` using the Gauss-Legendre quadrature rule with `n` points.

# Arguments
- `fun::Function`: The function to integrate. Should accept a single `Float64` argument.
- `n::Int`: The number of quadrature points to use.
- `a::Real`: The lower bound of the integration interval.
- `b::Real`: The upper bound of the integration interval.

# Returns
- `Float64`: The approximate value of the integral of `fun` over `[a, b]`.

"""
function gaussquadrature(
    fun::Function,
    n::Int,
    a::Real,
    b::Real
)
    # Get Gauss-Legendre points and weights for the interval [-1, 1]
    (points, weights) = gausslegendre(n)

    # Change of variable from [-1, 1] to [a, b], vectorized computation
    scale = 0.5 * (b - a)
    shift = 0.5 * (a + b)
    x_transformed = muladd.(points, scale, shift)  # fused multiply-add for efficiency
    integral_result = dot(weights, fun.(x_transformed)) * scale

    return integral_result
end
