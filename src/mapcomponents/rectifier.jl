
# Softplus rectifier
function softplus(ξ)
    return log1p(exp(ξ))  # log(1 + exp(ξ)) for numerical stability
end

# Shifted exponential linear unit (ELU)
function shifted_elu(ξ)
    return ξ <= 0 ? exp(ξ) : ξ + 1
end
