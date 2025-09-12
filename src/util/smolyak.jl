function nested_hermite_levels_quads()
    levels = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    levels[0] = ([0.0], [1.0])
    nodes1, weights1 = gausshermite(3; normalize=true)
    levels[1] = (nodes1, weights1)
    for k in 2:10
        n = 2^k + 1
        try
            nodes_k, weights_k = gausshermite(n; normalize=true)
            levels[k] = (nodes_k, weights_k)
        catch
            # stop extending if the underlying routine fails for very large n
            break
        end
    end
    return levels
end

function get_hermite_rule_quads(level::Int)
    if level == 0
        return ([0.0], [1.0])
    else
        n = min(2^level + 1, 200)
        return gausshermite(n; normalize=true)
    end
end

function generate_multi_indices(d::Int, target_sum::Int)
    if d == 1
        return [[target_sum]]
    end
    indices = Vector{Vector{Int}}()
    for i1 in 0:target_sum
        remaining_sum = target_sum - i1
        sub_indices = generate_multi_indices(d - 1, remaining_sum)
        for sub_idx in sub_indices
            push!(indices, [i1; sub_idx])
        end
    end
    return indices
end

function create_tensor_product(nodes_sets::Vector{Vector{Float64}}, weights_sets::Vector{Vector{Float64}})
    d = length(nodes_sets)
    ranges = [1:length(nodes_sets[i]) for i in 1:d]
    all_combinations = Iterators.product(ranges...)

    tensor_nodes = Vector{Vector{Float64}}()
    tensor_weights = Float64[]

    for combination in all_combinations
        node = [nodes_sets[i][combination[i]] for i in 1:d]
        weight = prod(weights_sets[i][combination[i]] for i in 1:d)
        push!(tensor_nodes, node)
        push!(tensor_weights, weight)
    end

    return tensor_nodes, tensor_weights
end

function combine_duplicate_nodes(nodes::Vector{Vector{Float64}}, weights::Vector{Float64}; tol=1e-12)
    unique_nodes = Vector{Vector{Float64}}()
    combined_weights = Float64[]

    for (node, weight) in zip(nodes, weights)
        found_idx = 0
        for (i, existing_node) in enumerate(unique_nodes)
            if all(abs.(node .- existing_node) .< tol)
                found_idx = i
                break
            end
        end
        if found_idx > 0
            combined_weights[found_idx] += weight
        else
            push!(unique_nodes, copy(node))
            push!(combined_weights, weight)
        end
    end

    return unique_nodes, combined_weights
end

function hermite_smolyak_points(d::Int, level::Int)
    all_nodes = Vector{Vector{Float64}}()
    all_weights = Float64[]

    for total_level in max(0, level - d + 1):level
        multi_indices = generate_multi_indices(d, total_level)
        coeff = (-1)^(level - total_level) * binomial(d - 1, level - total_level)

        for mi in multi_indices
            nodes_1d = Vector{Vector{Float64}}()
            weights_1d = Vector{Vector{Float64}}()
            for j in 1:d
                nodes_j, weights_j = get_hermite_rule_quads(mi[j])
                push!(nodes_1d, nodes_j)
                push!(weights_1d, weights_j)
            end

            tensor_nodes, tensor_weights = create_tensor_product(nodes_1d, weights_1d)

            for (node, weight) in zip(tensor_nodes, tensor_weights)
                push!(all_nodes, node)
                push!(all_weights, coeff * weight)
            end
        end
    end

    unique_nodes, combined_weights = combine_duplicate_nodes(all_nodes, all_weights)

    filtered_nodes = Vector{Vector{Float64}}()
    filtered_weights = Float64[]
    for (node, weight) in zip(unique_nodes, combined_weights)
        if abs(weight) > 1e-14
            push!(filtered_nodes, node)
            push!(filtered_weights, weight)
        end
    end

    n_nodes = length(filtered_nodes)
    if n_nodes == 0
        return zeros(Float64, 0, d), Float64[]
    end

    points = Matrix{Float64}(undef, n_nodes, d)
    for (i, node) in enumerate(filtered_nodes)
        points[i, :] = node
    end

    return points, filtered_weights
end
