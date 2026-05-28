# DiscretizedBVP - Discretized Boundary Value Problem
#
# This module defines the DiscretizedBVP struct which combines a BVPModel
# with a discretization method to create a callable functional.

using DocStringExtensions

"""
$(TYPEDEF)

A discretized BVP ready for Newton iteration and continuation.

Combines a mathematical model (`BVPModel`) with a discretization method
to produce a callable functional F(X, p) = 0.

## Fields
$(TYPEDFIELDS)

## Usage

The DiscretizedBVP is callable:
```julia
model = BVPModel(F, g; n=2)
disc = Trapeze(M=100)
bvp = discretize(model, disc)

# Evaluate residual
res = bvp_residual(bvp, X, params)

# Use with BVPBifProblem
prob = BVPBifProblem(bvp, X0, params, (@optic _.μ))
```
"""
struct DiscretizedBVP{Tmodel<:BVPModel, Tdisc<:AbstractDiscretizer, Tcache}
    "Mathematical BVP model"
    model::Tmodel

    "Discretization method"
    discretizer::Tdisc

    "Pre-allocated workspace (for performance)"
    cache::Tcache
end

# ============================================================================
# Interface: bvp_residual and bvp_jacobian
# ============================================================================
# These functions must be implemented for each discretizer type.
# The implementations are in the discretizer-specific files:
    #   - trapeze/residual.jl, trapeze/jacobian.jl
#   - shooting/residual.jl, shooting/jacobian.jl
#   - collocation/residual.jl, collocation/jacobian.jl

"""
$(TYPEDSIGNATURES)

Compute the residual F(X, p) for the discretized BVP.
Must be implemented for each discretizer type.
"""
function bvp_residual end

"""
$(TYPEDSIGNATURES)

Compute the Jacobian ∂F/∂X with specified jacobian type.
Specialized implementations can be provided for each discretizer type.
"""
function bvp_jacobian end

# Default implementation for AutoDiffDense - uses ForwardDiff
function bvp_jacobian(d_bvp::DiscretizedBVP, ::BK.AutoDiffDense, x, p)
    FD.jacobian(z -> bvp_residual(d_bvp, z, p), x)
end
# ============================================================================
# Getters
# ============================================================================

"""State dimension."""
state_dimension(bvp::DiscretizedBVP) = state_dimension(bvp.model)

"""Total dimension of the discretized problem."""
Base.length(bvp::DiscretizedBVP) = total_dim(bvp.discretizer, state_dimension(bvp))

"""Get the underlying model."""
get_model(bvp::DiscretizedBVP) = bvp.model

"""Get the discretizer."""
get_discretizer(bvp::DiscretizedBVP) = bvp.discretizer

"""Get the cache."""
get_cache(bvp::DiscretizedBVP) = bvp.cache

record_from_solution(bvp::DiscretizedBVP) = record_from_solution(get_model(bvp))
plot_solution(bvp::DiscretizedBVP) = plot_solution(get_model(bvp))

# ============================================================================
# Display
# ============================================================================
function Base.show(io::IO, bvp::DiscretizedBVP)
    println(io, "┌─ DiscretizedBVP")
    println(io, "├─ State dimension : ", state_dimension(bvp))
    println(io, "├─ Total unknowns  : ", length(bvp))
    println(io, "├─ Model           : BVPModel")
    print(io,   "└─ Discretizer     : ", typeof(bvp.discretizer).name.name)
end
# ============================================================================
function get_time_slices(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, u::AbstractVector) where {Tmodel}
    coll = d_bvp.cache.po_coll
    N, m, Ntst = size(coll)
    BK.get_time_slices(u, N, m, Ntst)
end

function get_time_slices(d_bvp::DiscretizedBVP{Tmodel, <: Shooting}, u::AbstractVector) where {Tmodel}
    sh = d_bvp.cache
    N = state_dimension(d_bvp)
    M = mesh_size(get_discretizer(d_bvp))
    reshape(u, N, M)
end

function get_time_slices(d_bvp::DiscretizedBVP{Tmodel, <: Trapeze}, u::AbstractVector) where {Tmodel}
    N = state_dimension(d_bvp)
    M = mesh_size(get_discretizer(d_bvp))
    reshape(u, N, M)
end
# ============================================================================
function get_solution_bvp(d_bvp::DiscretizedBVP{Tmodel, <: Shooting}, u::AbstractVector, params) where {Tmodel}
    sh = d_bvp.cache
    N = state_dimension(d_bvp)
    disc = get_discretizer(d_bvp)
    um = get_time_slices(d_bvp, u)
    t0, tf = get_time_interval(get_model(d_bvp))
    return BK._get_shooting_solution(sh, um, tf - t0, params) # TODO must be a SolBVP
end

function get_solution_bvp(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, u::AbstractVector, params) where {Tmodel}
    t0, tf = get_time_interval(get_model(d_bvp))
    T = tf - t0
    coll = d_bvp.cache.po_coll
    ts = BK.get_times(coll)
    um = get_time_slices(d_bvp, u)
    return BK.SolPeriodicOrbit(t = ts .* T, u = um) # TODO must be a SolBVP
end

function get_solution_bvp(d_bvp::DiscretizedBVP{Tmodel, <: Trapeze}, u::AbstractVector, params) where {Tmodel}
    t0, tf = get_time_interval(get_model(d_bvp))
    T = tf - t0
    disc = get_discretizer(d_bvp)
    ts = pushfirst!(cumsum(collect(disc.mesh)), zero(T))
    um = get_time_slices(d_bvp, u)
    return BK.SolPeriodicOrbit(t = t0 .+ T .* ts, u = um)
end