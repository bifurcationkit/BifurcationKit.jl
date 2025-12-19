# Discretizers - Numerical discretization methods for BVPs
#
# This module defines lightweight configuration structs for each
# discretization method. They contain no problem-specific data,
# only method parameters.

using DocStringExtensions

"""
$(TYPEDEF)

Abstract type for all BVP discretization methods.
"""
abstract type AbstractDiscretizer end

# ============================================================================
# Shooting
# ============================================================================

"""
$(TYPEDEF)

Shooting method discretization.

Solves the BVP by:
1. Guessing initial conditions at M shooting points
2. Integrating the ODE between points
3. Matching conditions at boundaries

## Fields
$(TYPEDFIELDS)

## Example
```julia
using OrdinaryDiffEq
disc = Shooting(M=4, alg=Tsit5())
```
"""
struct Shooting{Talg} <: AbstractDiscretizer
    "Number of shooting intervals"
    M::Int
    
    "ODE solver algorithm (from OrdinaryDiffEq.jl)"
    alg::Talg
    
    "Use parallel integration for multiple shooting"
    parallel::Bool
end

"""
$(TYPEDSIGNATURES)

Create a shooting discretizer.

## Keyword Arguments
- `M::Int = 1`: Number of shooting intervals (M=1 is simple shooting)
- `alg = nothing`: ODE solver algorithm
- `parallel::Bool = false`: Use parallel integration
"""
Shooting(; M::Int=1, alg=nothing, parallel::Bool=false) = Shooting(M, alg, parallel)

# ============================================================================
# Trapezoid (Finite Difference)
# ============================================================================

"""
$(TYPEDEF)

Trapezoidal rule discretization.

Discretizes the BVP using:
    uᵢ₊₁ - uᵢ = (h/2)(F(uᵢ) + F(uᵢ₊₁))

where h = T/(M-1) is the time step.

## Fields
$(TYPEDFIELDS)

## Example
```julia
disc = Trap(M=100)
```
"""
struct Trap{Tjac} <: AbstractDiscretizer
    "Number of time slices"
    M::Int
    
    "Jacobian computation method (:auto, :dense, :sparse, :matrixfree)"
    jacobian::Tjac
end

"""
$(TYPEDSIGNATURES)

Create a trapezoidal discretizer.

## Keyword Arguments
- `M::Int = 100`: Number of time slices
- `jacobian = :auto`: Jacobian computation method
"""
Trap(; M::Int=100, jacobian=:auto) = Trap(M, jacobian)

# ============================================================================
# Collocation
# ============================================================================

"""
$(TYPEDEF)

Orthogonal collocation discretization.

Uses piecewise polynomials of degree `m` on `Ntst` mesh intervals,
with collocation at Gauss-Legendre points.

## Fields
$(TYPEDFIELDS)

## Example
```julia
disc = Collocation(Ntst=20, m=4)
```
"""
struct Collocation{Tjac} <: AbstractDiscretizer
    "Number of mesh intervals"
    Ntst::Int
    
    "Polynomial degree"
    m::Int
    
    "Jacobian computation method"
    jacobian::Tjac
    
    "Enable mesh adaptation"
    meshadapt::Bool
    
    "Mesh adaptation parameter (max/min step ratio)"
    K::Float64
end

"""
$(TYPEDSIGNATURES)

Create a collocation discretizer.

## Keyword Arguments
- `Ntst::Int = 20`: Number of mesh intervals
- `m::Int = 4`: Polynomial degree
- `jacobian = :auto`: Jacobian computation method
- `meshadapt::Bool = false`: Enable mesh adaptation
- `K::Float64 = 100.0`: Mesh adaptation parameter
"""
Collocation(; Ntst::Int=20, m::Int=4, jacobian=:auto, meshadapt::Bool=false, K::Float64=100.0) = 
    Collocation(Ntst, m, jacobian, meshadapt, K)

# ============================================================================
# Common Interface
# ============================================================================

"""Number of time points/intervals in the discretization."""
function mesh_size end

mesh_size(d::Shooting) = d.M
mesh_size(d::Trap) = d.M
mesh_size(d::Collocation) = d.Ntst * d.m + 1

"""Dimension of the discretized solution vector (excluding period)."""
function solution_dim end

solution_dim(d::Shooting, n::Int) = n * d.M
solution_dim(d::Trap, n::Int) = n * d.M
solution_dim(d::Collocation, n::Int) = n * (d.Ntst * d.m + 1)

"""Total dimension including period/parameter."""
total_dim(d::AbstractDiscretizer, n::Int) = solution_dim(d, n) + 1

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, d::Shooting)
    println(io, "┌─ Shooting Discretizer")
    println(io, "├─ Intervals M : ", d.M)
    println(io, "├─ ODE solver  : ", isnothing(d.alg) ? "default" : typeof(d.alg).name.name)
    print(io,   "└─ Parallel    : ", d.parallel)
end

function Base.show(io::IO, d::Trap)
    println(io, "┌─ Trapezoid Discretizer")
    println(io, "├─ Time slices M : ", d.M)
    print(io,   "└─ Jacobian      : ", d.jacobian)
end

function Base.show(io::IO, d::Collocation)
    println(io, "┌─ Collocation Discretizer")
    println(io, "├─ Mesh intervals Ntst : ", d.Ntst)
    println(io, "├─ Polynomial degree m : ", d.m)
    println(io, "├─ Total points        : ", mesh_size(d))
    println(io, "├─ Jacobian            : ", d.jacobian)
    print(io,   "└─ Mesh adaptation     : ", d.meshadapt)
end
