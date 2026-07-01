function __bvp_residual_collocation(d_bvp, X, p, compute_phase::Val{Tv} = Val(false)) where {Tv}

    # il faut s insopierer de periodicorbit/Collocation.jl

    model = get_model(d_bvp)
    disc = get_discretizer(d_bvp)
    po_coll = d_bvp.cache.po_coll
    nf = state_dimension(model) # TODO stay at the level of d_bvp
    Ntst, m = disc.Ntst, disc.m # TODO remove
    N_total = 1 + Ntst * m # TODO remove
    interval = get_time_interval(model) # TODO stay at the level of d_bvp
    δT = interval[2] - interval[1]

    # Extract solution
    # Xm = get_time_slices(d_bvp, out)
    Xm = reshape(@view(X[1:nf*N_total]), nf, N_total)

    # Get output buffer from cache
    # Robust check: only use cache for Float64 to avoid chunk mismatch in Dual
    # TODO use get_time_slices(d_bvp, out)
    out = similar(X)
    outm = reshape(@view(out[1:nf*N_total]), nf, N_total)

    # Core residual computation from BifurcationKit
    # This writes to outm[:, 1:Ntst*m]
    #po_residual_bare!(po_coll, outm, Xm, p, 1)
    phase = BK.po_residual_bare!(po_coll, outm, Xm, δT, BK.get_Ls(po_coll), p; compute_phase)  # TODO Val(d_bvp isa DiscretizedPO)

    # Boundary condition: g(u(0), u(T), p) = 0
    u0 = @view Xm[:, 1]
    uT = @view Xm[:, end]
    g_val = model.g(u0, uT, p)
    outm[:, end] .= g_val
    return out, phase
end

function bvp_residual(d_bvp::DiscretizedBVP{<: BVPModel, <: Collocation}, X, p)
    out, _ = __bvp_residual_collocation(d_bvp, X, p, Val(false))
    return out
end

# DiscretizedPO n est pas encore connu
# function bvp_residual(d_bvp::DiscretizedPO{<: BVPModel, <: Collocation}, X, p)
#     @assert false "WIP"
#     phase = __bvp_residual_collocation(d_bvp, X, p, Val(true))
#     out[end] = phase
# end
