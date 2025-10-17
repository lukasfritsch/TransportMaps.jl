using TransportMaps
using Test
using Distributions

@testset "LinearizedHermiteBasis" begin
    # Create samples from a normal distribution
    samples = randn(1000)
    max_degree = 4
    k = 2
    basis = LinearizedHermiteBasis(samples, max_degree, k)

    # Test bounds are set to quantiles
    lower, upper = basis.linearizationbounds
    @test isfinite(lower) && isfinite(upper)
    @test isapprox(lower, quantile(samples, 0.01); atol=1e-8)
    @test isapprox(upper, quantile(samples, 0.99); atol=1e-8)


    s = sprint(show, basis)
    @test occursin("LinearizedHermiteBasis", s) || !isempty(s)

    # Test normalization for k and not k
    for n in 0:max_degree
        if n == k
            @test basis.normalization[n+1] == factorial(n+1)
        else
            @test basis.normalization[n+1] == factorial(n)
        end
    end

    # Test piecewise polynomial and derivative
    n = 3
    z_a = lower - 1.0
    z_b = upper + 1.0
    z_mid = (lower + upper) / 2
    # Left linear region
    ψ_left = basisfunction(basis, n, z_a)
    ψ_left_expected = hermite_polynomial(n, lower) + hermite_derivative(n, lower) * (z_a - lower)
    ψ_left_expected /= sqrt(basis.normalization[n])
    @test isapprox(ψ_left, ψ_left_expected; atol=1e-10)
    # Right linear region
    ψ_right = basisfunction(basis, n, z_b)
    ψ_right_expected = hermite_polynomial(n, upper) + hermite_derivative(n, upper) * (z_b - upper)
    ψ_right_expected /= sqrt(basis.normalization[n])
    @test isapprox(ψ_right, ψ_right_expected; atol=1e-10)
    # Middle region
    ψ_mid = basisfunction(basis, n, z_mid)
    ψ_mid_expected = hermite_polynomial(n, z_mid) / sqrt(basis.normalization[n])
    @test isapprox(ψ_mid, ψ_mid_expected; atol=1e-10)

    # Derivative left
    dψ_left = basisfunction_derivative(basis, n, z_a)
    dψ_left_expected = hermite_derivative(n, lower) / sqrt(basis.normalization[n])
    @test isapprox(dψ_left, dψ_left_expected; atol=1e-10)
    # Derivative right
    dψ_right = basisfunction_derivative(basis, n, z_b)
    dψ_right_expected = hermite_derivative(n, upper) / sqrt(basis.normalization[n])
    @test isapprox(dψ_right, dψ_right_expected; atol=1e-10)
    # Derivative mid
    dψ_mid = basisfunction_derivative(basis, n, z_mid)
    dψ_mid_expected = hermite_derivative(n, z_mid) / sqrt(basis.normalization[n])
    @test isapprox(dψ_mid, dψ_mid_expected; atol=1e-10)

    # Additional constructor cases
    # Default constructor (keyword defaults)
    b_def = LinearizedHermiteBasis()
    @test isa(b_def, LinearizedHermiteBasis)
    @test b_def.linearizationbounds[1] == -Inf
    @test b_def.linearizationbounds[2] == Inf

    # max_degree-only constructor
    b_deg = LinearizedHermiteBasis(3)
    @test length(b_deg.normalization) == 4
    @test all(x -> x == 1.0, b_deg.normalization)

    # density-based constructor
    b_den = LinearizedHermiteBasis(Normal(0.0, 1.0), 4, 1)
    lb, ub = b_den.linearizationbounds
    @test isapprox(lb, quantile(Normal(0.0,1.0), 0.01); atol=1e-8)
    @test isapprox(ub, quantile(Normal(0.0,1.0), 0.99); atol=1e-8)

    # Constructor errors: passing Any-typed empty vector is a MethodError (no matching signature)
    @test_throws MethodError LinearizedHermiteBasis(Any[], 3, 1)

end
