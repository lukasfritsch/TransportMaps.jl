
# Compute Mᵏ according to Eq. (4.13)
function compute_Mk(
    Psi_vec::Vector{MVBasis},
    coefficients::Vector{Float64},
    x::Vector{Float64},
    rectifier::Function,
    k::Int)

    @assert length(Psi_vec) == length(coefficients) "Number of basis functions must equal number of coefficients"
    @assert k > 0 "k must be a positive integer"
    @assert k <= length(x) "k must not exceed the dimension of x"
    @assert length(x) == length(Psi_vec[1].multi_index) "Dimension mismatch: x and multi_index must have same length"


    # f(x₁, ..., x_{k-1}, 0, a)
    x₀ = copy(x)
    x₀[k] = 0.0
    f₀ = f(Psi_vec, coefficients, x₀)

    # Integrand for the integral over \bar{x}
    integrand(x̄) = begin
        x_temp = copy(x)
        x_temp[k] = x̄
        ∂f = partial_derivative_x(Psi_vec, coefficients, x_temp, k)
        return rectifier(∂f)
    end

    ∫g∂f, _ = quadgk(integrand, 0.0, x₀[k])

    return f₀ + ∫g∂f
end
