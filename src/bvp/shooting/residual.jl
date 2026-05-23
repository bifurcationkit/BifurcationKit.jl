# Shooting Residual Implementation
# 
# This file implements bvp_residual for Shooting discretization.
# It reuses the existing po_residual_bare! from BifurcationKit's StandardShooting.

import BifurcationKit: po_residual_bare!, get_mesh_size, isparallel, evolve

"""
$(TYPEDSIGNATURES)

Compute the residual for shooting discretization.
Calls BifurcationKit's po_residual_bare! and adds phase condition.
"""
function bvp_residual(bvp::DiscretizedBVP{<:BVPModel, <:Shooting}, X, p)
    model = get_model(bvp)
    disc = get_discretizer(bvp)
    n = state_dimension(model)
    t0, tf = get_time_interval(model)
    M = disc.M

    # Extract shooting points and period
    U = reshape(@view(X[1:n*M]), n, M)
    T = tf - t0

    # Allocate output
    out = similar(X)
    outU = reshape(@view(out[1:n*M]), n, M)
    
    # Core residual computation using BVP-specific po_residual_bare!
    bvp_residual_bare!(bvp, outU, U, p, T)
    return out
end

"""
$(TYPEDSIGNATURES)

Core shooting residual computation for DiscretizedBVP.
This implements the same logic as BifurcationKit's po_residual_bare! for ShootingProblem.
"""
@views function bvp_residual_bare!(bvp::DiscretizedBVP{<:BVPModel, <:Shooting}, outc, xc, pars, T)
    model = get_model(bvp)
    sh = get_cache(bvp) # TODO this is a hack for now
    M = sh.M
    if ~isparallel(sh)
        for ii in 1:M-1
            ip1 = ii+1
            outc[:, ii] .= BifurcationKit.evolve(sh.flow, xc[:, ii], pars, sh.ds[ii] * T).u .- xc[:, ip1]
        end
        outc[:, M] .= model.g(xc[:, 1], BifurcationKit.evolve(sh.flow, xc[:, M], pars, sh.ds[M] * T).u, pars)
    else
        solOde = BifurcationKit.evolve(sh.flow, xc, pars, sh.ds .* T)
        for ii in 1:M-1
            ip1 = ii+1
            outc[:, ii] .= @views solOde[ii][2] .- xc[:, ip1]
        end
        outc[:, M] .= model.g(xc[:, 1], solOde[M][2], pars)
    end
end

"""
Integrate ODE from u0 for time dt using specified algorithm.
"""
function integrate_shooting end

# Default: nothing algorithm uses Euler
function integrate_shooting(F, u0, p, dt, ::Nothing)
    return euler_integrate(F, u0, p, dt)
end

# Fallback for unknown algorithm types
function integrate_shooting(F, u0, p, dt, alg)
    error("""
    ODE integration with algorithm $(typeof(alg)) requires OrdinaryDiffEq.jl.
    
    Either:
    1. Use `Shooting(alg=nothing)` for built-in Euler integration
    2. Load OrdinaryDiffEq: `using OrdinaryDiffEq`
    """)
end

"""Simple Euler integration."""
function euler_integrate(F, u0, p, dt; nsteps=100)
    h = dt / nsteps
    u = copy(u0)
    for _ in 1:nsteps
        u .+= h .* F(u, p)
    end
    return u
end

"""Compute the phase condition scalar."""
function _shooting_phase(model::BVPModel, u0, uT, p)
    g_val = model.g(u0, uT, p)
    
    if g_val isa AbstractVector
        if has_phase_constraint(model)
            return evaluate_phase(model, u0, p)
        else
            return g_val[1]
        end
    else
        return g_val
    end
end
