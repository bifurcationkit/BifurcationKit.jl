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
struct BVPSavedSolutionAndState{T1, T2, T3, T4}
    mesh::T1
    sol::T2
    _mesh::T3
    ϕ::T4
end

"""
$(TYPEDEF)

Structure to encode the solution associated to a functional like `::Collocation` or `::Shooting`. In the particular case of `::Collocation`, this allows to use the collocation polynomials to interpolate the solution. Hence, if `sol::BVPInterpolation`, then one can call

    sol = BVPInterpolation(prob_coll, x)
    sol(t)

on any time `t`.

## Fields
$(TYPEDFIELDS)
"""
struct BVPInterpolation{Tpb, Tx, Tp}
    pb::Tpb
    x::Tx
    pars::Tp
end
BVPInterpolation(prob, x) = BVPInterpolation(prob, x, nothing)
getx(interp::BVPInterpolation) = interp.x
getprob(interp::BVPInterpolation) = interp.pb

struct DiscretizedBVP{Tmodel <: BVPModel, Tdisc <: AbstractDiscretizer, Tcache} <: AbstractDiscretizedBVP
    "Mathematical BVP model"
    model::Tmodel

    "Discretization method"
    discretizer::Tdisc

    "Pre-allocated workspace (for performance)"
    cache::Tcache
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Getters
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

"""Get the mesh cache."""
get_mesh_cache(bvp::DiscretizedBVP) = get_cache(bvp).mesh_cache

"""Get the collocation cache."""
get_coll_cache(bvp::DiscretizedBVP) = get_cache(bvp).coll_cache

BK.record_from_solution(bvp::DiscretizedBVP) = BK.record_from_solution(get_model(bvp))
BK.plot_solution(bvp::DiscretizedBVP) = BK.plot_solution(get_model(bvp))

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Display
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function Base.show(io::IO, bvp::DiscretizedBVP)
    println(io, "┌─ DiscretizedBVP")
    println(io, "├─ State dimension : ", state_dimension(bvp))
    println(io, "├─ Total unknowns  : ", length(bvp))
    println(io, "├─ Model           : BVPModel")
    print(io,   "└─ Discretizer     : ", typeof(bvp.discretizer).name.name)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DiscretizedPO - Discretized Periodic Orbit Problem
#
# A periodic orbit is a BVP with a periodic boundary condition and an
# additional unknown: the period T. The section provides the phase condition.
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Wraps a `DiscretizedBVP` together with a Poincaré section to form a periodic orbit problem.
The period `T` is appended as the last component of the solution vector `X`.

## Fields
$(TYPEDFIELDS)
"""
struct DiscretizedPO{Tmodel <: BVPModel, Tdisc <: AbstractDiscretizer, Tcache, Ts} <: AbstractDiscretizedPO
    "The underlying discretized BVP (without phase condition)"
    d_bvp::DiscretizedBVP{Tmodel, Tdisc, Tcache}

    "Phase condition (section)"
    section::Ts
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Getters for DiscretizedPO — delegate to d_bvp
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""State dimension of the underlying BVP."""
state_dimension(d_po::DiscretizedPO) = state_dimension(d_po.d_bvp)

"""Get the underlying BVPModel."""
get_model(d_po::DiscretizedPO) = get_model(d_po.d_bvp)

"""Get the discretizer."""
get_discretizer(d_po::DiscretizedPO) = get_discretizer(d_po.d_bvp)

"""Get the cache."""
get_cache(d_po::DiscretizedPO) = get_cache(d_po.d_bvp)

"""Get the mesh cache."""
get_mesh_cache(d_po::DiscretizedPO) = get_mesh_cache(d_po.d_bvp)

"""Get the collocation cache."""
get_coll_cache(d_po::DiscretizedPO) = get_coll_cache(d_po.d_bvp)

"""Total length of the solution vector: state unknowns + 1 (period T)."""
Base.length(d_po::DiscretizedPO) = length(d_po.d_bvp) + 1

"""Get the section (phase condition)."""
get_section(d_po::DiscretizedPO) = d_po.section

get_time_slices(d_po::DiscretizedPO, u::AbstractVector) = get_time_slices(d_po.d_bvp, u)

"""Get the time interval of the underlying model."""
get_time_interval(d_bvp::DiscretizedBVP) = get_time_interval(get_model(d_bvp))
get_time_interval(d_po::DiscretizedPO) = get_time_interval(d_po.d_bvp)

BK.record_from_solution(d_po::DiscretizedPO) = BK.record_from_solution(d_po.d_bvp)
BK.plot_solution(d_po::DiscretizedPO) = BK.plot_solution(d_po.d_bvp)

function Base.show(io::IO, d_po::DiscretizedPO)
    println(io, "┌─ DiscretizedPO")
    println(io, "├─ State dimension : ", state_dimension(d_po))
    println(io, "├─ Total unknowns  : ", length(d_po))
    println(io, "├─ Model           : ", typeof(get_model(d_po)).name.name)
    println(io, "├─ Discretizer     : ", typeof(get_discretizer(d_po)).name.name)
    print(io,   "└─ Section         : ", typeof(get_section(d_po)).name.name)
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function get_time_slices(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, u::AbstractVector) where {Tmodel}
    N = state_dimension(d_bvp)
    m = get_m(get_discretizer(d_bvp))
    Ntst = get_ntst(get_discretizer(d_bvp))
    BK.get_time_slices(u, N, m, Ntst)
end

function get_time_slices(d_bvp::DiscretizedBVP{Tmodel, <: Shooting}, u::AbstractVector) where {Tmodel}
    sh = d_bvp.cache
    N = state_dimension(d_bvp)
    M = mesh_size(get_discretizer(d_bvp))
    reshape(@view(u[1:N*M]), N, M)
end

function get_time_slices(d_bvp::DiscretizedBVP{Tmodel, <: Trapeze}, u::AbstractVector) where {Tmodel}
    N = state_dimension(d_bvp)
    M = mesh_size(get_discretizer(d_bvp))
    reshape(@view(u[1:N*M]), N, M)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function get_solution_bvp(d_bvp::DiscretizedBVP{Tmodel, <: Shooting}, u::AbstractVector, params) where {Tmodel}
    sh = d_bvp.cache
    N = state_dimension(d_bvp)
    disc = get_discretizer(d_bvp)
    um = get_time_slices(d_bvp, u)
    t0, tf = get_time_interval(get_model(d_bvp))
    return BK._get_shooting_solution(sh, um, tf - t0, params)
end

function get_solution_bvp(d_bvp::DiscretizedBVP{Tmodel, <: Trapeze}, u::AbstractVector, params) where {Tmodel}
    t0, tf = get_time_interval(get_model(d_bvp))
    T = tf - t0
    disc = get_discretizer(d_bvp)
    ts = pushfirst!(cumsum(collect(disc.mesh)), zero(T))
    um = get_time_slices(d_bvp, u)
    return BK.BVPSolution(t = t0 .+ T .* ts, u = um)
end

function get_solution_bvp(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, u::AbstractVector, params) where {Tmodel}
    t0, tf = get_time_interval(get_model(d_bvp))
    T = tf - t0
    ts = get_times(d_bvp.cache.mesh_cache)
    um = get_time_slices(d_bvp, u)
    return BK.BVPSolution(t = ts .* T, u = um)
end

function get_solution_bvp(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, x::Tx, params) where {Tmodel, Tx <: BVPSavedSolutionAndState}
    t0, tf = get_time_interval(get_model(d_bvp))
    T = tf - t0
    ts = get_times(d_bvp.cache.mesh_cache)
    mesh = x.mesh
    u = BK.saved_solution(x)
    um = get_time_slices(d_bvp, u)
    return BK.BVPSolution(t = mesh .* T, u = um)
end