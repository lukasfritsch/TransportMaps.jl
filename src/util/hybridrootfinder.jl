# Root finder to get inverse of polynomial map components

# Helper function to find bounds for the inverse map
# Based on `inverse` in `IntegratedPositiveFunction.m` from https://github.com/baptistar/ATM
function _inverse_bound(fun::Function)
    # Initial bounds for the root finding
    lower = -1.0
    upper = 1.0

    fa = fun(lower)
    fb = fun(upper)
    # Expand bounds until the root is bracketed
    while fa * fb > 0.0
        delta = 0.5 * (upper - lower)
        if fa > 0
            lower -= delta
        elseif fb < 0
            upper += delta
        end
        fa = fun(lower)
        fb = fun(upper)
    end

    return lower, upper
end


# `hybridRootFindingSolver.m` from https://github.com/baptistar/ATM
function hybridrootfinder(
    f::Function,
    ∂f::Function,
    lower::Real,
    upper::Real;
    xtol::Real=1e-6,
    ftol::Real=1e-6,
    maxiter::Int=10_000
)
    x = (lower + upper) / 2
    fx, dfx = f(x), ∂f(x)

    for _ in 1:maxiter
        # Decide between bisection and Newton step
        use_bisect = (x - upper) * dfx - fx * (x - lower) * dfx > 0.0 ||
                     (abs(2 * fx) > abs(upper - lower) * abs(dfx) && abs(fx) > ftol)
        if use_bisect
            x = lower + 0.5 * (upper - lower)
        else
            x -= fx / dfx
        end

        fx, dfx = f(x), ∂f(x)

        if abs(fx) < ftol || abs(fx / dfx) < xtol
            return x, fx, dfx
        end

        if fx < 0.0
            lower = x
        else
            upper = x
        end
    end

    @warn "Maximum iterations reached in hybridrootfinder"
    return x, fx, dfx
end
