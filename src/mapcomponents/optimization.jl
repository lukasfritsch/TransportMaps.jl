function objective(a::Vector{Float64}, basis_functions, xq::Matrix{Float64}, wq::Vector{Float64}, π_tilde)
    pf = PolynomialFunction(basis_functions, a)
    Nq, d = size(xq)
    total = 0.0

    for i in 1:Nq
        xi = xq[i, :]
        Mi = [compute_Mk(pf, xi, k) for k in 1:d]
        log_pi = log(π_tilde(Mi))

        log_detJ = sum([log(dMk_dxk(pf, xi, k)) for k in 1:d])

        total += wq[i] * (-log_pi - log_detJ)
    end

    return total
end