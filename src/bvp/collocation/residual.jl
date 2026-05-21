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
function bvp_residual(d_bvp::DiscretizedBVP{<:BVPModel, <:Collocation}, X, p)
    model = get_model(d_bvp)
    disc = get_discretizer(d_bvp)
    po_coll = d_bvp.cache.po_coll
    nf = state_dimension(model)
    Ntst, m = disc.Ntst, disc.m
    N_total = 1 + Ntst * m
    interval = get_time_interval(model)
    δT = interval[2] - interval[1]

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
    #po_residual_bare!(po_coll, outc, Xc, p, 1)
    po_residual_bare!(po_coll, outc, Xc, δT, BifurcationKit.get_Ls(po_coll), p; compute_phase = Val(false))

    # Boundary condition: g(u(0), u(T), p) = 0
    u0 = @view Xc[:, 1]
    uT = @view Xc[:, end]
    g_val = model.g(u0, uT, p)
    outc[:, end] .= g_val
    return out
end
