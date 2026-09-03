function bvp_residual(d_bvp::DiscretizedBVP{<: BVPModel, <: Collocation}, X, p)
    model = get_model(d_bvp)
    disc = get_discretizer(d_bvp)
    po_coll = d_bvp.cache.po_coll
    nf = state_dimension(model)
    Ntst, m = disc.Ntst, disc.m
    N_total = 1 + Ntst * m
    interval = get_time_interval(model)
    δT = interval[2] - interval[1]

    N = nf * N_total
    length(X) == N || throw(ArgumentError("bvp_residual: expected length(X) == $N, got $(length(X))"))

    # Extract solution
    Xm = reshape(@view(X[1:N]), nf, N_total)

    # Allocate output; every entry is written below
    out = similar(X, N)
    outm = reshape(out, nf, N_total)

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
