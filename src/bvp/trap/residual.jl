# Trapezoid Residual Implementation
#
# This file implements bvp_residual for Trap discretization.
# Uses BifurcationKit's optimized po_residual_bare! with topology correction.

import BifurcationKit: po_residual_bare!, get_time_slices

"""
$(TYPEDSIGNATURES)

Compute the residual for trapezoidal discretization.
Uses BifurcationKit's PeriodicOrbitTrapProblem for efficient computation.
"""
function bvp_residual(bvp::DiscretizedBVP{<:BVPModel, <:Trap}, X, p)
    model = bvp.model
    disc = bvp.discretizer
    po_trap = bvp.cache.po_trap
    n = state_dimension(model)
    M = disc.M

    # Extract time slices and period
    Xc = reshape(@view(X[1:n*M]), n, M)
    T = X[end]
    
    # Determine result type (promote X and p)
    T_res = eltype(X)
    if p isa NamedTuple
        for v in values(p)
            T_res = promote_type(T_res, eltype(v))
        end
    else
        T_res = promote_type(T_res, eltype(p))
    end
    
    # Get output buffer from cache
    # Robust check: only use cache for Float64 to avoid chunk mismatch in Dual
    out = similar(X, T_res)
    outc = reshape(@view(out[1:n*M]), n, M)
    
    # Call BifurcationKit's optimized kernel
    po_residual_bare!(po_trap, outc, Xc, p, T)

    # Boundary condition: g(u(0), u(T), p) = 0
    u0 = @view Xc[:, 1]
    uT = @view Xc[:, M]
    g_val = model.g(u0, uT, p)
    outc[:, M] .= g_val
    
    # Phase condition (last scalar)
    if has_phase_constraint(model)
        out[end] = evaluate_phase(model, Xc[:, 1], p, T)
    else
        # Default: use BifurcationKit's phase condition vectors from cache
        if iszero(po_trap.ϕ)
             out[end] = Xc[1, 1] # Fallback for initialization
        else
             u_flat = vec(Xc)
             out[end] = LinearAlgebra.dot(u_flat, po_trap.ϕ) - LinearAlgebra.dot(po_trap.xπ, po_trap.ϕ)
        end
    end
    
    return out
end
