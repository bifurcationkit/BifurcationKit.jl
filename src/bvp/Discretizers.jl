# Discretizers - Numerical discretization methods for BVPs
#
# This module defines lightweight configuration structs for each
# discretization method. They contain no problem-specific data,
# only method parameters.

using DocStringExtensions
import ..BifurcationKit: AutoDiffDense, TimeMesh, can_adapt

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shooting
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

## Constructor
- `M::Int = 1`: Number of shooting intervals (M=1 is simple shooting)
- `alg = nothing`: ODE solver algorithm
- `parallel::Bool = false`: Use parallel integration
```
"""
Base.@kwdef struct Shooting{Talg} <: AbstractDiscretizer
    "Number of shooting intervals"
    M::Int

    "ODE solver algorithm (from OrdinaryDiffEq.jl)"
    alg::Talg

    "Use parallel integration for multiple shooting"
    parallel::Bool
end

is_parallel(sh::Shooting) = sh.parallel

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trapezoid (Finite Difference)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
disc = Trapeze(M=100)
```
"""
struct Trapeze{Tjac, Tmesh} <: AbstractDiscretizer
    "Number of time slices"
    M::Int

    "Time mesh over M-1 intervals (normalized to sum to 1)"
    mesh::Tmesh

    "Jacobian computation method (:auto, :dense, :sparse, :matrixfree)"
    jacobian::Tjac
end

"""
$(TYPEDSIGNATURES)

Create a trapezoidal discretizer.

## Keyword Arguments
- `M::Int = 100`: Number of time slices
- `mesh = nothing`: Optional normalized step vector over `M-1` intervals
- `jacobian = :auto`: Jacobian computation method
"""
function Trapeze(; M::Int=100, mesh=TimeMesh(M - 1), jacobian = AutoDiffDense())
    @assert M >= 2 "Trapeze requires at least M=2 time slices"

    msh = if isnothing(mesh)
        TimeMesh(M - 1)
    elseif mesh isa TimeMesh
        _validate_mesh(mesh, M - 1)
        mesh
    else
        _mesh_from_steps(mesh, M - 1)
    end

    return Trapeze(M, msh, jacobian)
end

function _validate_mesh(mesh::TimeMesh{Ti}, n_intervals::Int) where {Ti <: Int}
    @assert length(mesh) == n_intervals "Expected $n_intervals intervals, got $(length(mesh))"
    return mesh
end

function _validate_mesh(mesh::TimeMesh, n_intervals::Int)
    @assert length(mesh) == n_intervals "Expected $n_intervals intervals, got $(length(mesh))"
    @assert all(mesh.ds .> 0) "Mesh steps must be strictly positive"
    return mesh
end

function _mesh_from_steps(mesh_steps, n_intervals::Int)
    @assert length(mesh_steps) == n_intervals "Expected $n_intervals intervals, got $(length(mesh_steps))"
    steps = float.(collect(mesh_steps))
    @assert all(steps .> 0) "Mesh steps must be strictly positive"
    steps ./= sum(steps)
    return TimeMesh(steps)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Collocation
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
Base.@kwdef struct Collocation <: AbstractDiscretizer
    "Number of mesh intervals."
    Ntst::Int = 0

    "Polynomial degree."
    m::Int = 0

    "Enable mesh adaptation."
    meshadapt::Bool = false

    "Mesh adaptation parameter (max/min step ratio)."
    K::Float64 = 100.

    "Update mesh update_every_step continuation step."
    update_every_step::Int = 1

    verbose_mesh_adapt::Bool = false
end

meshadapt(coll::Collocation) = coll.meshadapt
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Common Interface
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""Number of time points/intervals in the discretization."""
function mesh_size end

mesh_size(d::Shooting) = d.M
mesh_size(d::Trapeze) = d.M
mesh_size(d::Collocation) = d.Ntst * d.m + 1

"""Dimension of the discretized solution vector (excluding period)."""
function solution_dim end

solution_dim(d::Shooting, n::Int) = n * d.M
solution_dim(d::Trapeze, n::Int) = n * d.M
solution_dim(d::Collocation, n::Int) = n * (d.Ntst * d.m + 1)

"""Total dimension including period/parameter."""
total_dim(d::AbstractDiscretizer, n::Int) = solution_dim(d, n) + 1
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Display
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function Base.show(io::IO, d::Shooting)
    println(io, "┌─ Shooting Discretizer")
    println(io, "├─ Intervals M : ", d.M)
    println(io, "├─ ODE solver  : ", isnothing(d.alg) ? "default" : typeof(d.alg).name.name)
    print(io,   "└─ Parallel    : ", d.parallel)
end

function Base.show(io::IO, d::Trapeze)
    println(io, "┌─ Trapezoid Discretizer")
    println(io, "├─ Time slices M : ", d.M)
    println(io, "├─ Adaptive mesh : ", can_adapt(d.mesh))
    print(io,   "└─ Jacobian      : ", d.jacobian)
end

function Base.show(io::IO, d::Collocation)
    println(io, "┌─ Collocation Discretizer")
    println(io, "├─ Mesh intervals Ntst : ", d.Ntst)
    println(io, "├─ Polynomial degree m : ", d.m)
    println(io, "├─ Total points        : ", mesh_size(d))
    print(io,   "└─ Mesh adaptation     : ", d.meshadapt)
end
