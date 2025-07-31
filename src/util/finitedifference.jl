function central_difference_gradient(fun::Function, x::AbstractArray{<:Real}, ε=1e-10)
    n = length(x)
    ∇fun = similar(x)
    for i in 1:n
        e = zeros(eltype(x), n)
        e[i] = ε
        ∇fun[i] = (fun(x + e) - fun(x - e)) / (2ε)
    end

    return ∇fun
end
