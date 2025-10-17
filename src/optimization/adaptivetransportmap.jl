# Placeholder for ATM implementation


function adaptive_optimization(
    component::PolynomialMapComponent,
    samples::Matrix{Float64},
    maximum_cardinality::Int
)

# initialize empty multi-index set
multi_index_set = 0

for card in 1:maximum_cardinality
    println("Optimizing with cardinality $card / $maximum_cardinality")
    optimize!(component, samples, LBFGS(), Optim.Options(g_tol=1e-5))
end

end
