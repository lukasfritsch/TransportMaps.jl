function central_difference_gradient(f::F, x::Vector{Float64}, ε::Float64=√eps(Float64)) where {F<:Function}
    n = length(x)
    g = similar(x, Float64)

    # Work buffers to avoid allocations
    xp = copy(x)
    xm = copy(x)

    for i in 1:n
        δ = ε * max(1, abs(x[i]))
        xp[i] = x[i] + δ
        xm[i] = x[i] - δ
        g[i] = (f(xp) - f(xm)) / (2δ)
        xp[i] = x[i]   # reset
        xm[i] = x[i]
    end
    return g
end
