using TransportMaps
using Test
using Random

@testset "Map from Samples" begin

    Random.seed!(789)

    banana_density = function(x)
        return exp(-0.5 * x[1]^2) * exp(-0.5 * (x[2] - x[1]^2)^2)
    end

    num_samples = 500

    function generate_banana_samples(n_samples::Int)
        samples = Matrix{Float64}(undef, n_samples, 2)

        count = 0
        while count < n_samples
            x1 = randn() * 2
            x2 = randn() * 3 + x1^2

            if rand() < banana_density([x1, x2]) / 0.4
                count += 1
                samples[count, :] = [x1, x2]
            end
        end

        return samples
    end

    samples_banana = generate_banana_samples(num_samples)
    M = PolynomialMap(2, 2)
    result = optimize!(M, samples_banana)

    @test result[1].iterations > 0  # Check that optimization ran
    @test isfinite(result[1].minimum)

    @test result[2].iterations > 0  # Check that optimization ran
    @test isfinite(result[2].minimum)

    samples_new = generate_banana_samples(100)
    M2 = PolynomialMap(2, 2)
    result2 = optimize!(M2, samples_new, test_fraction=0.3)

    @test result2[1].iterations > 0  # Check that optimization ran
    @test isfinite(result2[1].minimum)

    @test result2[2].iterations > 0  # Check that optimization ran
    @test isfinite(result2[2].minimum)

end
