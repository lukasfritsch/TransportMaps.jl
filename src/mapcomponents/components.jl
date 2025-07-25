# Struct to represent a multi-index α
struct MultiIndex
    indices::Vector{Int}
end

# Struct for basis functions Ψ_α
struct BasisFunction
    multiindex::MultiIndex
end

# Evaluate univariate Hermite polynomial of given degree at x
function hermite_poly(n::Int, x::Float64)
    if n == 0
        return 1.0
    elseif n == 1
        return 2.0 * x
    else
        Hn_minus_two = 1.0
        Hn_minus_one = 2.0 * x
        for k in 2:n
            Hn = 2.0 * x * Hn_minus_one - 2.0 * (k - 1) * Hn_minus_two
            Hn_minus_two, Hn_minus_one = Hn_minus_one, Hn
        end
        return Hn_minus_one
    end
end

# Evaluate tensor product Ψ_α(x₁, ..., xₖ)
function evaluate_basis(bf::BasisFunction, x::Vector{Float64})
    @assert length(bf.multiindex.indices) == length(x)
    prod(hermite_poly(degree, xi) for (degree, xi) in zip(bf.multiindex.indices, x))
end

# Struct for the function f(x₁,...,xₖ, a)
struct PolynomialFunction
    basis_functions::Vector{BasisFunction}
    coefficients::Vector{Float64}
end

# Evaluate f at x
function evaluate(pf::PolynomialFunction, x::Vector{Float64})
    @assert length(pf.basis_functions) == length(pf.coefficients)
    sum(a * evaluate_basis(bf, x) for (a, bf) in zip(pf.coefficients, pf.basis_functions))
end

# Struct for the full vector-valued map M(x)
struct VectorPolynomialMap
    components::Vector{PolynomialFunction}  # one for each dimension of the output
end

# Evaluate M(x)
function evaluate(vpm::VectorPolynomialMap, x::Vector{Float64})
    [evaluate(comp, x[1:length(comp.basis_functions[1].multiindex.indices)]) for comp in vpm.components]
end

# Example monotonic function g — identity or softplus
g(x) = exp.(x)             # exponential
# g(x) = log(1 + exp(x))  # softplus

# Compute ∂f/∂xk numerically via central difference, there are some smarter ways of doing this
function df_dxk(pf::PolynomialFunction, x::Vector{Float64}, k::Int; h=1e-6)
    x_forward = copy(x)
    x_backward = copy(x)
    x_forward[k] += h
    x_backward[k] -= h
    (evaluate(pf, x_forward) - evaluate(pf, x_backward)) / (2h)
end

# Compute M^k according to Eq. (4.13)
function compute_Mk(pf::PolynomialFunction, x::Vector{Float64}, k::Int)
    # f(x₁, ..., x_{k-1}, 0, a)
    x0 = copy(x)
    x0[k] = 0.0
    f_at_0 = evaluate(pf, x0)

    # Integrand for the integral over \bar{x}
    integrand(x̄) = begin
        x_temp = copy(x)
        x_temp[k] = x̄
        g(df_dxk(pf, x_temp, k))
    end

    integral, _ = quadgk(integrand, 0.0, x[k])

    return f_at_0 + integral
end

# Compute diagonal derivative element ∂M^k/∂x_k at x
function dMk_dxk(pf::PolynomialFunction, x::Vector{Float64}, k::Int)
    g(df_dxk(pf, x, k))
end
