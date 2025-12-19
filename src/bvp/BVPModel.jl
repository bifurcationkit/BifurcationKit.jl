# BVPModel - Mathematical BVP Formulation
#
# This module defines the mathematical structure of a Boundary Value Problem
# without any discretization. The BVPModel captures:
#   - Vector field F(u, p)
#   - Boundary conditions g(u(0), u(1), p) = 0

using DocStringExtensions

"""
$(TYPEDEF)

Mathematical formulation of a Boundary Value Problem (no discretization).

## Mathematical Definition

    u'(t) = F(u(t), p),        t ∈ [0, 1]
    g(u(0), u(1), p) = 0       boundary conditions

## Fields
$(TYPEDFIELDS)

## Example

```julia
# Simple harmonic oscillator BVP
F(u, p) = [u[2], -p.ω² * u[1]]
g(u0, u1, p) = u0 .- u1  # Periodic BC

model = BVPModel(F, g; n=2)
```
"""
struct BVPModel{TF, Tg, Tphase}
    "Vector field: F(u, p) → ℝⁿ"
    F::TF
    
    "Boundary conditions: g(u(0), u(1), p) → ℝⁿᵇ (must equal zero)"
    g::Tg
    
    "Phase/integral constraint (optional): phase(u, p) → ℝ"
    phase::Tphase
    
    "State dimension n"
    n::Int
end

# ============================================================================
# Constructors
# ============================================================================

"""
$(TYPEDSIGNATURES)

Create a BVP model with vector field `F` and boundary conditions `g`.

## Arguments
- `F`: Vector field `F(u, p) → ℝⁿ`
- `g`: Boundary condition `g(u0, u1, p) → ℝⁿᵇ` (should return zero at solution)

## Keyword Arguments
- `n::Int = 0`: State dimension (can be inferred later)
- `phase = nothing`: Optional phase/integral constraint

## Example
```julia
F(u, p) = [u[2], -u[1]]
g(u0, u1, p) = u0 .- u1

model = BVPModel(F, g; n=2)
```
"""
function BVPModel(F, g; n::Int=0, phase=nothing)
    BVPModel(F, g, phase, n)
end

"""
$(TYPEDSIGNATURES)

Create a BVP model for periodic orbits.

The boundary condition is automatically set to `u(0) = u(1)`.

## Arguments
- `F`: Vector field `F(u, p) → ℝⁿ`

## Keyword Arguments
- `n::Int = 0`: State dimension
- `phase = nothing`: Optional phase constraint

## Example
```julia
# Stuart-Landau oscillator
F(u, p) = [p.μ*u[1] - u[2] - u[1]*(u[1]^2 + u[2]^2),
           u[1] + p.μ*u[2] - u[2]*(u[1]^2 + u[2]^2)]

model = PeriodicOrbitModel(F; n=2)
```
"""
function PeriodicOrbitModel(F; n::Int=0, phase=nothing)
    g_periodic(u0, u1, p) = u0 .- u1
    BVPModel(F, g_periodic, phase, n)
end

# ============================================================================
# Getters
# ============================================================================

"""State dimension of the model."""
state_dimension(model::BVPModel) = model.n

"""Check if the model has a phase constraint."""
has_phase_constraint(model::BVPModel) = !isnothing(model.phase)

"""Evaluate the vector field."""
evaluate_F(model::BVPModel, u, p) = model.F(u, p)

"""Evaluate the boundary condition."""
evaluate_g(model::BVPModel, u0, u1, p) = model.g(u0, u1, p)

"""Evaluate the phase constraint (if present)."""
function evaluate_phase(model::BVPModel, u, p, T)
    isnothing(model.phase) && return zero(eltype(u))
    # Support both (u, p) and (u, p, T) signatures
    try
        return model.phase(u, p, T)
    catch
        return model.phase(u, p)
    end
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, model::BVPModel)
    println(io, "┌─ BVPModel")
    println(io, "├─ State dimension n : ", model.n == 0 ? "unspecified" : model.n)
    println(io, "├─ Vector field F    : ", typeof(model.F).name.name)
    println(io, "├─ Boundary cond g   : ", typeof(model.g).name.name)
    print(io,   "└─ Phase constraint  : ", has_phase_constraint(model) ? "yes" : "none")
end

# ============================================================================
# Utilities
# ============================================================================

"""
$(TYPEDSIGNATURES)

Create a copy of the model with updated state dimension.
"""
function set_dimension(model::BVPModel, n::Int)
    BVPModel(model.F, model.g, model.phase, n)
end
