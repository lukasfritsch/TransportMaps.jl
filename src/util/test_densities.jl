# Test density functions for TransportMaps.jl
# These functions provide irregular, complex shapes for testing transport maps

using Distributions
using StatsFuns

"""
    capybara_density(x::AbstractVector{<:Real})

A test density function that creates an irregular capybara-like shape.

The capybara density is designed to have:
- A main oval body (large component)  
- A smaller head region (smaller component)
- Irregular, non-convex overall shape
- Proper normalization (integrates to approximately 1)

This provides a challenging test case for transport map optimization
due to its irregular geometry and multi-modal nature.

# Arguments
- `x::AbstractVector{<:Real}`: 2D point [x₁, x₂]

# Returns
- `Float64`: Density value at point x

# Example
```julia
using TransportMaps

# Create a target density
target = TargetDensity(capybara_density, :auto_diff)

# Use with transport map optimization
M = PolynomialMap(2, 3, Softplus())
quadrature = GaussHermiteWeights(5, 2)
result = optimize!(M, target, quadrature)
```
"""
function capybara_density(x::AbstractVector{<:Real})
    if length(x) != 2
        throw(ArgumentError("capybara_density only defined for 2D inputs"))
    end
    
    x1, x2 = x[1], x[2]
    
    # Main body component (shifted oval)
    # Center at (0.5, 0) with elongated shape
    body_center = [0.5, 0.0]
    body_cov = [1.2 0.3; 0.3 0.8]  # Correlation creates irregular shape
    body_component = pdf(MvNormal(body_center, body_cov), x)
    
    # Head component (smaller, shifted up and left)
    # Center at (-0.8, 0.3) 
    head_center = [-0.8, 0.3]
    head_cov = [0.4 -0.1; -0.1 0.3]  # Smaller with negative correlation
    head_component = pdf(MvNormal(head_center, head_cov), x)
    
    # Legs components (small bumps at bottom)
    leg1_center = [0.2, -1.2]
    leg1_cov = [0.15 0.0; 0.0 0.1]
    leg1_component = pdf(MvNormal(leg1_center, leg1_cov), x)
    
    leg2_center = [0.8, -1.2]
    leg2_cov = [0.15 0.0; 0.0 0.1]
    leg2_component = pdf(MvNormal(leg2_center, leg2_cov), x)
    
    # Tail component (small protrusion)
    tail_center = [1.8, -0.2]
    tail_cov = [0.3 0.1; 0.1 0.2]
    tail_component = pdf(MvNormal(tail_center, tail_cov), x)
    
    # Combine components with weights to create capybara shape
    # Main body dominates, other components add irregular features
    density = 0.6 * body_component + 
              0.25 * head_component + 
              0.05 * leg1_component + 
              0.05 * leg2_component + 
              0.05 * tail_component
    
    return density
end

"""
    irregular_density(x::AbstractVector{<:Real})

A simpler irregular test density with non-convex shape.

This creates a density with multiple modes and irregular boundaries,
useful for testing transport map capabilities on challenging geometries.

# Arguments  
- `x::AbstractVector{<:Real}`: 2D point [x₁, x₂]

# Returns
- `Float64`: Density value at point x
"""
function irregular_density(x::AbstractVector{<:Real})
    if length(x) != 2
        throw(ArgumentError("irregular_density only defined for 2D inputs"))
    end
    
    x1, x2 = x[1], x[2]
    
    # Create irregular shape using combination of functions
    # Main component with polynomial modulation
    main = exp(-0.5 * (x1^2 + x2^2)) * (1 + 0.3 * cos(3 * atan(x2, x1)))
    
    # Secondary mode 
    secondary = 0.3 * exp(-2 * ((x1 + 1.5)^2 + (x2 - 1)^2))
    
    # Ensure non-negative
    density = max(main + secondary, 1e-12)
    
    return density
end