# Collocation Residual Implementation
#
# This file implements bvp_residual for Collocation discretization.
# Uses BifurcationKit's optimized po_residual_bare! for collocation.

import BifurcationKit: po_residual_bare!

"""
$(TYPEDSIGNATURES)

Compute the residual for collocation discretization.
Uses BifurcationKit's PeriodicOrbitOCollProblem for efficient computation.
"""
function bvp_residual(bvp::DiscretizedBVP{<:BVPModel, <:Collocation}, X, p)
    model = bvp.model
    disc = bvp.discretizer 
    po_coll = bvp.cache.po_coll
    nf = state_dimension(model)
    Ntst, m = disc.Ntst, disc.m
    N_total = 1 + Ntst * m
    
    # Extract solution
    Xc = reshape(@view(X[1:nf*N_total]), nf, N_total)
        
    # Determine result type (promote X and p)
    T_res = eltype(X)
    if p isa NamedTuple
        for v in values(p)
            T_res = promote_type(T_res, eltype(v))
        end
    else
        T_res = promote_type(T_res, eltype(p) )
    end
    
    # Get output buffer from cache
    # Robust check: only use cache for Float64 to avoid chunk mismatch in Dual
    out = similar(X)
    outc = reshape(@view(out[1:nf*N_total]), nf, N_total)
    
    # Core residual computation from BifurcationKit
    # This writes to outc[:, 1:Ntst*m]
    po_residual_bare!(po_coll, outc, Xc, p, 1)
    
    # Boundary condition: g(u(0), u(T), p) = 0
    u0 = @view Xc[:, 1]
    uT = @view Xc[:, end]
    g_val = model.g(u0, uT, p)
    outc[:, end] .= g_val
    
    # Phase condition
    if has_phase_constraint(model)
        # @assert false
        # out[end] = evaluate_phase(model, Xc[:, 1], p, T)
    else
        # Default: user set T-1 in example or we could use integral phase
        # out[end] = T - 1.0 # Standard for Bratu example
    end

    return out
end
