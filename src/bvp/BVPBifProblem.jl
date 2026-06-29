# BVP Bifurcation Problem
#
# This file defines the BVPBifProblem struct which wraps a DiscretizedBVP
# for use with BifurcationKit's continuation and bifurcation algorithms.

import BifurcationKit: AbstractBifurcationProblem, 
                       save_solution_default,
                       plot_solution, record_from_solution, save_solution, update!,
                       _getvectortype, getu0, getparams, getlens, getparam, setparam, _get,
                       is_symmetric, re_make, set,
                       residual, jacobian, isinplace
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
struct BVPBifProblem{Tbvp <: DiscretizedBVP, Tjac, Tu, Tp, Tl, Tplot, Trec, Tupdate} <: AbstractBVPBifProblem
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
    "Function to update the problem"
    update!::Tupdate
end

"""
$(TYPEDSIGNATURES)

Create a `BVPBifProblem` from a discretized BVP.

## Arguments
- `d_bvp::DiscretizedBVP`: The discretized BVP
- `u0`: Initial guess for the solution
- `params`: [Optional] Parameters for the problem 
- `lens`: [Optional] Optic for the continuation parameter (e.g., `@optic _.μ`)

## Keyword Arguments
- `jacobian`: Jacobian type (default: `DenseAnalytical()`)
- `record_from_solution`: Function to record from solution (default records period)
- `plot_solution`: Function to plot solution (default: Nothing)
- `update!`: Function to update the problem (default: `update_default`)

## Example
```julia
model = BVPBifProblem(F, g; n=2)
disc = Trapeze(M=100)
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
    record_from_solution = nothing, # default value used for dispatch
    plot_solution = nothing,
    update! = BK.update_default
)
    return BVPBifProblem(
        d_bvp,
        jacobian,
        u0,
        params,
        lens,
        plot_solution,
        record_from_solution,
        update!
    )
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Interface methods required by BifurcationKit
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Get the vector type
_getvectortype(::BVPBifProblem{Tbvp, Tjac, Tu}) where {Tbvp, Tjac, Tu} = Tu
# Accessor methods
getu0(prob::BVPBifProblem) = prob.u0
getparams(prob::BVPBifProblem{Tbvp, Tjac, Tu}) where {Tbvp, Tjac, Tu} = prob.params # Careful: specific type definition suggested by Aqua.jl to avoid ambiguities
getparams(prob::BVPBifProblem{Tbvp, Tjac, Tu, Nothing}) where {Tbvp, Tjac, Tu} = getparams(get_bvp(prob))
getlens(prob::BVPBifProblem{Tbvp, Tjac, Tu, Tp}) where {Tbvp, Tjac, Tu, Tp} = prob.lens # Careful: specific type definition suggested by Aqua.jl to avoid ambiguities
getlens(prob::BVPBifProblem{Tbvp, Tjac, Tu, Tp, Nothing}) where {Tbvp, Tjac, Tu, Tp} = getlens(get_bvp(prob))
getparam(prob::BVPBifProblem) = _get(getparams(prob), getlens(prob))
setparam(prob::BVPBifProblem, p0) = set(getparams(prob), getlens(prob), p0)
isinplace(::BVPBifProblem) = false

# Recording and plotting, based on dispatch
BK.plot_solution(prob::BVPBifProblem{Tbvp, Tjac, Tu, Tp, Tl, Tplot}) where {Tbvp, Tjac, Tu, Tp, Tl, Tplot} = prob.plotSolution
BK.plot_solution(prob::BVPBifProblem{Tbvp, Tjac, Tu, Tp, Tl, Nothing}) where {Tbvp, Tjac, Tu, Tp, Tl} = BK.plot_solution(get_bvp(prob))

BK.record_from_solution(prob::BVPBifProblem, x, p; k...) = prob.recordFromSolution(x, p; k...)
BK.record_from_solution(prob::BVPBifProblem{Tbvp, Tjac, Tu, Tp, Tl, Tplot, Nothing}) where {Tbvp, Tjac, Tu, Tp, Tl, Tplot} = BK.record_from_solution(get_bvp(prob))
@inline BK.update!(prob::BVPBifProblem, args...; kwargs...) = prob.update!(args...; kwargs...)

# Residual - delegate to the DiscretizedBVP
residual(prob::BVPBifProblem, x, p) = bvp_residual(get_bvp(prob), x, p)

# Adjoint Support (required for branch switching and normal forms)
# For now, we assume no easy adjoint is available for arbitrary BVPs
import ..BifurcationKit: has_adjoint, getdelta, BifFunction, dF, d2F, d3F
has_adjoint(::BVPBifProblem) = false
getdelta(::BVPBifProblem) = 1e-8 # TODO remove this hack

# Differentials (required for normal forms)
function dF(prob::BVPBifProblem, x, p, dx)
    FD.derivative(t -> residual(prob, x .+ t .* dx, p), zero(eltype(x)))
end

function d2F(prob::BVPBifProblem, x, p, dx1, dx2)
    FD.derivative(t -> dF(prob, x .+ t .* dx2, p, dx1), zero(eltype(x)))
end

function d3F(prob::BVPBifProblem, x, p, dx1, dx2, dx3)
    FD.derivative(t -> d2F(prob, x .+ t .* dx3, p, dx1, dx2), zero(eltype(x)))
end
# Jacobian - dispatch on AutoDiffDense (default behavior)
jacobian(prob::BVPBifProblem, x, p) = bvp_jacobian(get_bvp(prob), prob.jacobian, x, p)

# is_symmetric defaults to false
is_symmetric(::BVPBifProblem) = false

# re_make for modifying the problem
function re_make(prob::BVPBifProblem; 
    d_bvp = prob.d_bvp,
    jacobian = prob.jacobian,
    u0 = getu0(prob),
    params = getparams(prob),
    lens = getlens(prob),
    plot_solution = prob.plotSolution,
    record_from_solution = prob.recordFromSolution,
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
        update!
    )
end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper functions specific to BVP problems
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
$(TYPEDSIGNATURES)

Get the underlying DiscretizedBVP from a BVPBifProblem.
"""
get_bvp(prob::BVPBifProblem) = prob.d_bvp
get_solution_bvp(br::BK.AbstractBranchResult, ind::Int) = get_solution_bvp(BK.getprob(br), br.sol[ind].x, BK.setparam(br, br.sol[ind].p))
get_solution_bvp(prob::BVPBifProblem, x, p) = get_solution_bvp(get_bvp(prob), x, p)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# save_solution functions specific to BVP problems
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
save_solution(prob::BVPBifProblem, x, p) = save_solution(prob.d_bvp, x, p)
save_solution(::DiscretizedBVP, x, _) = x

function save_solution(bvp::DiscretizedBVP{<: BVPModel, <: Collocation}, x, pars) # TODO: duplicate of existing function
    coll = bvp.cache.po_coll
    if BK.meshadapt(coll) # mildly type unstable but Union{T1, T2} handles it
        return BK.BVPSavedSolutionAndState(copy(BK.get_times(coll)),
                x,
                copy(BK.getmesh(coll.mesh_cache)),
                BK._copy(coll.ϕ),
                )
    else
        return x
    end
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function (sol::BK.BVPInterpolation{ <:  DiscretizedBVP{ Tmodel, <: Collocation}})(t0) where {Tmodel}
    d_bvp = sol.pb
    model = get_model(d_bvp)
    coll = d_bvp.cache.po_coll
    interval = get_time_interval(model)
    δT = interval[2] - interval[1]
    xm = get_time_slices(d_bvp, BK.getx(sol))
    BK.__interpolate_posolution(coll, t0, xm, δT)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function BK.update!(prob::BVPBifProblem{ <: DiscretizedBVP{ Tmodel, <: Collocation}}, 
                    x::BK.BVPSavedSolutionAndState) where {Tmodel <: BVPModel}
    d_bvp = get_bvp(prob)
    coll = d_bvp.cache.po_coll
    BK.update_mesh!(coll, x._mesh)
    return true
end

function __update_bvp_coll!(d_bvp::DiscretizedBVP, bvpsol, params, iter, state, update_pred = true)
    disc = get_discretizer(d_bvp)
    model = get_model(d_bvp)
    interval = get_time_interval(model)
    δT = interval[2] - interval[1]
    coll = d_bvp.cache.po_coll
    has_mesh_been_updated = false
    if meshadapt(disc) == false
        return true
    end
    update_every_step = disc.update_every_step
    step = state.step
    if BK.converged(state) &&
            BK.in_bisection(state) == false &&
            BK.mod_counter(step, update_every_step) == 1 &&
            step > 2
            @debug "[Collocation] update mesh"
        has_mesh_been_updated = true
        old_bvp = BK._copy(bvpsol) # avoid possible overwrite in compute_error!
        oldmesh = BK.get_times(coll) .* δT
        ####################################
        # get solution, we copy x because it is overwritten at the end of this function
        sol = BK.BVPInterpolation(deepcopy(d_bvp), copy(old_bvp), nothing)

        (; newmesh, ϕ) = BK._compute_error!(coll, sol, old_bvp, δT;
                            verbosity = disc.verbose_mesh_adapt,
                            K = coll.K,
                            par = BK.setparam(iter, BK.getp(state)))
        # update solution
        newsol = generate_solution(d_bvp, sol)
        old_bvp .= newsol
        success = true
        if ~success # stop continuation if mesh adaptation fails
            return false
        end
    end
    if has_mesh_been_updated && update_pred
        # we recompute the tangent predictor
        @debug "[collocation] update predictor"
        BK.getpredictor!(state, iter)
    end
    return true
end

function BK.update!(prob::BVPBifProblem{ <: DiscretizedBVP{ Tmodel, <: Collocation}},
                    iter::BK.ContIterable{Tkind},
                    state; 
                    update_pred = true) where {Tmodel <: BVPModel, Tkind}
    d_bvp = get_bvp(prob)
    bvpsol = BK._copy(BK.getx(state))
    __update_bvp_coll!(d_bvp, bvpsol, BK.setparam(iter, BK.getp(state)), iter, state)
end

function BK.update!(prob::BVPBifProblem{ <: DiscretizedBVP{ Tmodel, <: Collocation}},
                    iter::BK.ContIterable{Tkind},
                    state; 
                    update_pred = true) where {Tmodel <: BVPModel, Tkind <: BK.AbstractTwoParamCont}
    d_bvp = get_bvp(prob)
    Z = BK.getsolution(state)
    params = BK.getparams(iter, state)
    𝐌𝐚 = BK.get_formulation(BK.getprob(iter))
    bvpsol = BK.getvec(Z.u, 𝐌𝐚)
    __update_bvp_coll!(d_bvp, bvpsol, params, iter, state, false)
end