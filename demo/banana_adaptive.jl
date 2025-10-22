using TransportMaps
using Distributions
using LinearAlgebra
using Plots

banana_density(x) = pdf(Normal(), x[1]) * pdf(Normal(), x[2] - x[1]^2)

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

target_samples = generate_banana_samples(num_samples)

L = LinearMap(target_samples)

ATM, results = AdaptiveTransportMap(target_samples, [3, 6], 5, L, Softplus(2.))

C = ComposedMap(L, ATM)

ind_atm = getmultiindexsets(ATM[2])

dim = scatter(ind_atm[:, 1], ind_atm[:, 2], ms=30, legend=false)
plot!(xlims=(-0.5, maximum(ind_atm[:, 1]) + 0.5), ylims=(-0.5, maximum(ind_atm[:, 2]) + 0.5),
    aspect_ratio=1, xlabel="Multi-index α₁", ylabel="Multi-index α₂")
xticks!(0:maximum(ind_atm[:, 1]))
yticks!(0:maximum(ind_atm[:, 2]))

new_samples = generate_banana_samples(1000)
norm_samples = randn(1000, 2)

mapped_banana_samples = inverse(C, norm_samples)

p11 = scatter(new_samples[:, 1], new_samples[:, 2],
    label="Original Samples", alpha=0.5, color=1,
    title="Original Banana Distribution Samples",
    xlabel="x₁", ylabel="x₂")

scatter!(p11, mapped_banana_samples[:, 1], mapped_banana_samples[:, 2],
    label="Mapped Samples", alpha=0.5, color=2,
    title="Transport Map Generated Samples",
    xlabel="x₁", ylabel="x₂")

plot(p11, size=(600, 400))

x₁ = range(-3, 3, length=100)
x₂ = range(-2.5, 4.0, length=100)

true_density = [banana_density([x1, x2]) for x2 in x₂, x1 in x₁]

learned_density = [pullback(C, [x1, x2]) for x2 in x₂, x1 in x₁]

p3 = contour(x₁, x₂, true_density,
    title="True Banana Density",
    xlabel="x₁", ylabel="x₂",
    colormap=:viridis, levels=10)

p4 = contour(x₁, x₂, learned_density,
    title="Learned Density (Pullback)",
    xlabel="x₁", ylabel="x₂",
    colormap=:viridis, levels=10)

plot(p3, p4, layout=(1, 2), size=(800, 400))

best_fold = argmin([res.test_objectives[end] for res in results[2]])
res_best = results[2][best_fold]

max_1 = maximum(res_best.terms[end][:, 1])
max_2 = maximum(res_best.terms[end][:, 2])

p = plot(layout=(2, 3), xlims=(-0.5, max_1 + 0.5), ylims=(-0.5, max_2 + 0.5),
    aspect_ratio=1, xlabel="Multi-index α₁", ylabel="Multi-index α₂", legend=false,)

for (i, term) in enumerate(res_best.terms)
    scatter!(p, term[:, 1], term[:, 2], ms=20, title="Iteration $i", subplot=i)
end

xticks!(0:max_1)
yticks!(0:max_2)
plot!(p, size=(800, 600))

plot(res_best.train_objectives, label="Train Objective", lw=2, xlabel="Iteration",
    ylabel="Objective Value", title="Training and Test Objectives over Iterations")
plot!(res_best.test_objectives, label="Test Objective", lw=2)
plot!(size=(600, 400))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
