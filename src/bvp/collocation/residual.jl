function bvp_residual(d_bvp::DiscretizedBVP{<: BVPModel, <: Collocation}, X, p)
    model = get_model(d_bvp)
    disc = get_discretizer(d_bvp)
    po_coll = d_bvp.cache.po_coll
    nf = state_dimension(model)
    Ntst, m = disc.Ntst, disc.m
    N_total = 1 + Ntst * m
    interval = get_time_interval(model)
    δT = interval[2] - interval[1]

    # Extract solution
    Xm = reshape(@view(X[1:nf*N_total]), nf, N_total)

    # Get output buffer from cache
    # Robust check: only use cache for Float64 to avoid chunk mismatch in Dual
    out = similar(X)
    outm = reshape(@view(out[1:nf*N_total]), nf, N_total)

    # Core residual computation from BifurcationKit
    # This writes to outm[:, 1:Ntst*m]
    #po_residual_bare!(po_coll, outm, Xm, p, 1)
    BK.po_residual_bare!(po_coll, outm, Xm, δT, BK.get_Ls(po_coll), p; compute_phase = Val(false))

    # Boundary condition: g(u(0), u(T), p) = 0
    u0 = @view Xm[:, 1]
    uT = @view Xm[:, end]
    g_val = model.g(u0, uT, p)
    outm[:, end] .= g_val
    return out
end
