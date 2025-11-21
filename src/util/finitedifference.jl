function central_difference_gradient(f::F, x::Vector{Float64}, ε::Float64=eps(Float64)^(1/3)) where {F<:Function}
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

function central_difference_hessian(f::F, x::Vector{Float64}, ε::Float64=eps(Float64)^(1/4)) where {F<:Function}
    n = length(x)
    H = zeros(Float64, n, n)

    # Work buffers to avoid allocations
    xp = copy(x)
    xm = copy(x)
    xpp = copy(x)
    xmm = copy(x)

    for i in 1:n
        δi = ε *(1 + abs(x[i]))
        for j in i:n
            if i == j
                # Diagonal: second derivative
                δ = δi
                xp[i] = x[i] + δ
                xm[i] = x[i] - δ
                H[i, i] = (f(xp) - 2 * f(x) + f(xm)) / (δ^2)
                # Reset
                xp[i] = x[i]
                xm[i] = x[i]
            else
                # Off-diagonal: mixed partial
                δj = ε * (1 + abs(x[j]))

                xpp[i] = x[i] + δi
                xpp[j] = x[j] + δj

                xmm[i] = x[i] - δi
                xmm[j] = x[j] - δj

                xp[i] = x[i] + δi
                xp[j] = x[j] - δj

                xm[i] = x[i] - δi
                xm[j] = x[j] + δj

                H[i, j] = (f(xpp) - f(xp) - f(xm) + f(xmm)) / (4δi * δj)

                # Reset
                xpp[i] = x[i]
                xpp[j] = x[j]
                xmm[i] = x[i]
                xmm[j] = x[j]
                xp[i] = x[i]
                xp[j] = x[j]
                xm[i] = x[i]
                xm[j] = x[j]
            end
        end
    end

    # Fill lower triangle
    for i in 1:n
        for j in 1:i-1
            H[i, j] = H[j, i]
        end
    end

    return H
end
