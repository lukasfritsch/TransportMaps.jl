# Standard Hermite basis (probabilist's Hermite polynomials)
struct HermiteBasis <: AbstractPolynomialBasis

	function HermiteBasis()
		return new()
	end
end

# Univariate probabilist's Hermite polynomials
@inline function hermite_polynomial(n::Int64, z::Float64)
	if n == 0
		return 1.0
	elseif n == 1
		return z
	else
		H_nm2 = 1.0
		H_nm1 = z
		for k in 2:n
			H_n = z * H_nm1 - (k - 1) * H_nm2
			H_nm2, H_nm1 = H_nm1, H_n
		end
		return H_nm1
	end
end

# Derivative of univariate Hermite polynomial
@inline function hermite_derivative(n::Int64, z::Float64)
	n == 0 ? 0.0 : n * hermite_polynomial(n - 1, z)
end

# Basis function for standard Hermite
@inline function basisfunction(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
	return hermite_polynomial(Int(αᵢ), zᵢ)
end

# Derivative for standard Hermite
@inline function basisfunction_derivative(basis::HermiteBasis, αᵢ::Real, zᵢ::Real)
	return hermite_derivative(Int(αᵢ), zᵢ)
end

function Base.show(io::IO, ::HermiteBasis)
	print(io, "HermiteBasis()")
end
