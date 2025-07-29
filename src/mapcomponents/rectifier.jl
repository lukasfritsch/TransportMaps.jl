struct Softplus <: AbstractRectifierFunction
end

function (r::Softplus)(ξ)
    return log1p(exp(ξ))  # log(1 + exp(ξ)) for numerical stability
end

struct ShiftedELU <: AbstractRectifierFunction
end

# Shifted exponential linear unit (ELU)
function (r::ShiftedELU)(ξ)
    return ξ <= 0 ? exp(ξ) : ξ + 1
end

struct IdentityRectifier <: AbstractRectifierFunction
end

function (r::IdentityRectifier)(ξ)
    return ξ
end
