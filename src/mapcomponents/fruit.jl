# Fruit functionality for TransportMaps.jl
# Simple demonstration module implementing "Obst im Haus" (Fruit in the House)

# Abstract type for fruits
abstract type AbstractFruit end

# Concrete fruit types
struct Banana <: AbstractFruit
    ripeness::Float64  # 0.0 = green, 1.0 = fully ripe
end

# Constructor with default ripeness
Banana() = Banana(0.5)

# Fruit evaluation function - could be used as a basis function
function evaluate(fruit::Banana, x::Float64)
    # Simple function that depends on ripeness and input
    # Banana function: ripeness * sin(x) + (1 - ripeness) * cos(x)
    return fruit.ripeness * sin(x) + (1 - fruit.ripeness) * cos(x)
end

# Fruit properties
ripeness(fruit::Banana) = fruit.ripeness
name(fruit::Banana) = "banana"

# Collection to track available fruits (checklist functionality)
const FRUIT_CHECKLIST = Dict{String, Bool}(
    "banana" => true,  # ✓ implemented
)

# Function to check if a fruit is implemented
is_implemented(fruit_name::String) = get(FRUIT_CHECKLIST, fruit_name, false)

# Function to list implemented fruits
implemented_fruits() = [fruit for (fruit, implemented) in FRUIT_CHECKLIST if implemented]

# Function to list all fruits in checklist
all_fruits() = collect(keys(FRUIT_CHECKLIST))