# Concrete type for Hermite polynomials
struct HermiteBasis <: AbstractPolynomialBasis
    edge_control::Symbol

    function HermiteBasis(edge_control::Symbol=:none)
        if edge_control != :none && edge_control != :gaussian && edge_control != :cubic
            throw(ArgumentError("edge_control must be either :none or :gaussian or :cubic"))
        end
        return new(edge_control)
    end
end

# Univariate probabilist's Hermite polynomials
function hermite_polynomial(n::Int, z::Real)
    if n == 0
        return 1.0
    elseif n == 1
        return z
    else
        H_nm2 = 1.0  # H_{n-2}
        H_nm1 = z    # H_{n-1}
        for k in 2:n
            H_n = z * H_nm1 - (k - 1) * H_nm2
            H_nm2, H_nm1 = H_nm1, H_n
        end
        return H_nm1
    end
end

# Derivative of univariate Hermite polynomial
function hermite_derivative(n::Int, z::Real)
    if n == 0
        return 0.0
    else
        return n * hermite_polynomial(n - 1, z)
    end
end

# Hermite polynomial with edge control
function edge_controlled_hermite_polynomial(n::Int, z::Real, edge_control::Symbol)
    if edge_control == :gaussian
        weight = exp(-.25 * z.^2)
    elseif edge_control == :cubic
        r = 4.0 # Radius for cubic edge control
        m = min(1.0, abs(z)/r)
        weight = 2 * m.^3 - 3 * m.^2 + 1
    else
        weight = 1.0  # No edge control
    end

    return hermite_polynomial(n, z) .* weight
end

# Derivative of the univariate Hermite polynomial with edge control
function edge_controlled_hermite_derivative(n::Int, z::Real, edge_control::Symbol)
    # Gaussian weight
    if edge_control == :gaussian
        return exp(-.25 * z.^2) .* (n * hermite_polynomial(n-1, z) - z/2 .* hermite_polynomial(n, z))

    # Cubic weight
    elseif edge_control == :cubic
        r = 4.0

        f(z) = begin
            m = min(1.0, abs.(z)/r)
            return 2 * m^3 - 3 * m^2 + 1
        end

        ∂f(z) = begin
            if abs.(z) < r
                m = abs.(z) / r
                return (6/r) * (m.^2 - m) .* sign.(z)
            else
                return 0.0
            end
        end

        return hermite_derivative(n, z) .* f.(z) .+ hermite_polynomial(n, z) .* ∂f.(z)

    # No edge control
    else
        return hermite_derivative(n, z)
    end
end

function basisfunction(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
    return edge_controlled_hermite_polynomial(Int(αᵢ), zᵢ, basis.edge_control)
end

function basisfunction_derivative(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
    return edge_controlled_hermite_derivative(Int(αᵢ), zᵢ, basis.edge_control)
end

# Display methods for HermiteBasis
function Base.show(io::IO, basis::HermiteBasis)
    print(io, "HermiteBasis(edge_control=:$(basis.edge_control))")
end
