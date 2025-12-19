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
    model = bvp.model
    disc = bvp.discretizer
    n = state_dimension(model)
    M = disc.M
    
    # Extract shooting points and period
    U = reshape(@view(X[1:n*M]), n, M)
    T = X[end]
    
    # Allocate output
    out = similar(X)
    outU = reshape(@view(out[1:n*M]), n, M)
    
    # Core residual computation using BVP-specific po_residual_bare!
    po_residual_bare!(bvp, outU, U, p, T)
    
    # Phase condition (last scalar)
    u0 = @view U[:, 1]
    dt = T / M
    uT = integrate_shooting(model.F, @view(U[:, M]), p, dt, disc.alg)
    out[end] = _shooting_phase(model, u0, uT, p)
    
    return out
end

"""
$(TYPEDSIGNATURES)

Core shooting residual computation for DiscretizedBVP.
This implements the same logic as BifurcationKit's po_residual_bare! for ShootingProblem.
"""
@views function po_residual_bare!(bvp::DiscretizedBVP{<:BVPModel, <:Shooting}, outc, xc, pars, T)
    model = bvp.model
    disc = bvp.discretizer
    n, M = size(xc)
    dt = T / M
    
    for ii in 1:M
        ip1 = (ii == M) ? 1 : ii+1
        # Integrate from xc[:, ii] for time dt
        u_final = integrate_shooting(model.F, xc[:, ii], pars, dt, disc.alg)
        outc[:, ii] .= u_final .- xc[:, ip1]
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
