# BVP Bifurcation Problem
#
# This file defines the BVPBifProblem struct which wraps a DiscretizedBVP
# for use with BifurcationKit's continuation and bifurcation algorithms.

import BifurcationKit: AbstractBifurcationProblem, 
                       save_solution_default, update_default,
                       plot_solution, record_from_solution, save_solution, update!,
                       _getvectortype, getu0, getparams, getlens, getparam, setparam, _get,
                       is_symmetric, re_make, set,
                       residual, jacobian, isinplace,
                       AutoDiffDense

using Accessors: Accessors

"""
$(TYPEDEF)

Structure to hold a BVP bifurcation problem for computing periodic orbits.

## Fields

$(TYPEDFIELDS)

## Methods

- `re_make(pb; kwargs...)` modify a bifurcation problem
- `getu0(pb)` calls `pb.u0`
- `getparams(pb)` calls `pb.params`
- `getlens(pb)` calls `pb.lens`
- `getparam(pb)` calls `get(pb.params, pb.lens)`
- `setparam(pb, p0)` calls `set(pb.params, pb.lens, p0)`
- `record_from_solution(pb)` calls `pb.recordFromSolution`
- `plot_solution(pb)` calls `pb.plotSolution`
- `residual(pb, x, p)` computes the BVP residual
- `jacobian(pb, x, p)` computes the BVP jacobian

## Constructors

- `BVPBifProblem(d_bvp, u0, params, lens; kwargs...)` where `d_bvp` is a `DiscretizedBVP`
"""
struct BVPBifProblem{Tbvp<:DiscretizedBVP, Tjac, Tu, Tp, Tl, Tplot, Trec, Tgets, Tupdate} <: AbstractBifurcationProblem
    "The discretized BVP"
    d_bvp::Tbvp
    "The jacobian (type or function)"
    jacobian::Tjac
    "Initial guess"
    u0::Tu
    "Parameters"
    params::Tp
    "Lens for continuation parameter"
    lens::Tl
    "Function to plot solutions"
    plotSolution::Tplot
    "Function to record from solution"
    recordFromSolution::Trec
    "Function to save the full solution"
    save_solution::Tgets
    "Function to update the problem"
    update!::Tupdate
end

"""
$(TYPEDSIGNATURES)

Create a `BVPBifProblem` from a discretized BVP.

## Arguments
- `d_bvp::DiscretizedBVP`: The discretized BVP
- `u0`: Initial guess for the solution
- `params`: Parameters for the problem
- `lens`: Optic for the continuation parameter (e.g., `@optic _.μ`)

## Keyword Arguments
- `jacobian`: Jacobian type (default: `DenseAnalytical()`)
- `record_from_solution`: Function to record from solution (default records period)
- `plot_solution`: Function to plot solution (default: nothing)
- `save_solution`: Function to save the full solution (default: `save_solution_default`)
- `update!`: Function to update the problem (default: `update_default`)

## Example
```julia
model = PeriodicOrbitModel(F; n=2)
disc = Trap(M=100)
d_bvp = discretize(model, disc)

x0 = generate_solution(d_bvp, t -> [cos(t), sin(t)], 2π)
prob = BVPBifProblem(d_bvp, x0, (μ=1.0,), (@optic _.μ))

# Now use with newton/continuation
sol = newton(prob, NewtonPar())
br = continuation(prob, PALC(), ContinuationPar())
```
"""
function BVPBifProblem(
    d_bvp::DiscretizedBVP,
    u0,
    params,
    lens;
    jacobian = AutoDiffDense(),
    record_from_solution = (x, p; k...) -> (period = x[end],),
    plot_solution = (x, p; kwargs...) -> nothing,
    save_solution = save_solution_default,
    update! = update_default
)
    return BVPBifProblem(
        d_bvp,
        jacobian,
        u0,
        params,
        lens,
        plot_solution,
        record_from_solution,
        save_solution,
        update!
    )
end

# ============================================================================
# Interface methods required by BifurcationKit
# ============================================================================

# Get the vector type
_getvectortype(::BVPBifProblem{Tbvp, Tjac, Tu}) where {Tbvp, Tjac, Tu} = Tu
# Accessor methods
getu0(prob::BVPBifProblem) = prob.u0
getparams(prob::BVPBifProblem) = prob.params
getlens(prob::BVPBifProblem) = prob.lens
getparam(prob::BVPBifProblem) = _get(prob.params, prob.lens)
setparam(prob::BVPBifProblem, p0) = set(prob.params, prob.lens, p0)
isinplace(prob::BVPBifProblem) = false

# Recording and plotting
plot_solution(prob::BVPBifProblem) = prob.plotSolution
record_from_solution(prob::BVPBifProblem) = prob.recordFromSolution
record_from_solution(prob::BVPBifProblem, x, p; k...) = prob.recordFromSolution(x, p; k...)
save_solution(prob::BVPBifProblem, x, p) = prob.save_solution(x, p)
@inline update!(prob::BVPBifProblem, args...; kwargs...) = prob.update!(args...; kwargs...)

# Residual - delegate to the DiscretizedBVP
residual(prob::BVPBifProblem, x, p) = bvp_residual(prob.d_bvp, x, p)

# Adjoint Support (required for branch switching and normal forms)
# For now, we assume no easy adjoint is available for arbitrary BVPs
import ..BifurcationKit: has_adjoint, getdelta, BifFunction, dF, d2F, d3F
has_adjoint(::BVPBifProblem) = false
getdelta(::BVPBifProblem) = 1e-8

# Differentials (required for normal forms)
function dF(prob::BVPBifProblem, x, p, dx)
    ForwardDiff.derivative(t -> residual(prob, x .+ t .* dx, p), zero(eltype(x)))
end

function d2F(prob::BVPBifProblem, x, p, dx1, dx2)
    ForwardDiff.derivative(t -> dF(prob, x .+ t .* dx2, p, dx1), zero(eltype(x)))
end

function d3F(prob::BVPBifProblem, x, p, dx1, dx2, dx3)
    ForwardDiff.derivative(t -> d2F(prob, x .+ t .* dx3, p, dx1, dx2), zero(eltype(x)))
end

# Bridge to BifFunction (needed for some internal BK methods like branch switching)
function BifFunction(prob::BVPBifProblem)
    return BifFunction(
        (x, p) -> residual(prob, x, p),      # F
        nothing,                             # F!
        (x, p, dx) -> dF(prob, x, p, dx),    # dF
        nothing,                             # dFad
        (x, p) -> jacobian(prob, x, p),      # J
        nothing,                             # Jᵗ
        nothing,                             # J!
        (x, p, dx1, dx2) -> d2F(prob, x, p, dx1, dx2), # d2F
        (x, p, dx1, dx2, dx3) -> d3F(prob, x, p, dx1, dx2, dx3), # d3F
        nothing,                             # d2Fc
        nothing,                             # d3Fc
        false,                               # isSymmetric
        getdelta(prob),                      # δ
        false,                               # inplace
        nothing                              # jet
    )
end


# Jacobian - dispatch on AutoDiffDense (default behavior)
jacobian(prob::BVPBifProblem{Tbvp, AutoDiffDense}, x, p) where {Tbvp} = bvp_jacobian(prob.d_bvp, prob.jacobian, x, p)

# Make the problem callable (required by BifurcationKit)
(prob::BVPBifProblem)(x, p) = bvp_residual(prob.d_bvp, x, p)

# is_symmetric defaults to false
is_symmetric(::BVPBifProblem) = false

# re_make for modifying the problem
function re_make(prob::BVPBifProblem; 
    d_bvp = prob.d_bvp,
    jacobian = prob.jacobian,
    u0 = prob.u0,
    params = prob.params,
    lens = prob.lens,
    plot_solution = prob.plotSolution,
    record_from_solution = prob.recordFromSolution,
    save_solution = prob.save_solution,
    update! = prob.update!
)
    return BVPBifProblem(
        d_bvp,
        jacobian,
        u0,
        params,
        lens,
        plot_solution,
        record_from_solution,
        save_solution,
        update!
    )
end

# ============================================================================
# Helper functions specific to BVP problems
# ============================================================================

"""
$(TYPEDSIGNATURES)

Get the underlying DiscretizedBVP from a BVPBifProblem.
"""
get_bvp(prob::BVPBifProblem) = prob.d_bvp

"""
$(TYPEDSIGNATURES)

Get the period from a BVPBifProblem solution.
"""
getperiod(prob::BVPBifProblem, x, p) = x[end]

"""
$(TYPEDSIGNATURES)

Extract the periodic orbit from a BVPBifProblem solution.
"""
get_periodic_orbit(prob::BVPBifProblem, x, p) = get_periodic_orbit(prob.d_bvp, x, p)
