"""
$(TYPEDSIGNATURES)

Compute the residual for shooting discretization.
Calls BifurcationKit's po_residual_bare! and adds phase condition.
"""
function bvp_residual(d_bvp::DiscretizedBVP{<:BVPModel, <:Shooting}, X, p)
    model = get_model(d_bvp)
    disc = get_discretizer(d_bvp)
    n = state_dimension(model)
    t0, tf = get_time_interval(model)
    M = mesh_size(disc)

    # Extract shooting points and period
    Xm = reshape(@view(X[1:n*M]), n, M)
    T = tf - t0

    # Allocate output
    out = similar(X)
    outm = reshape(@view(out[1:n*M]), n, M)
    
    # Core residual computation using BVP-specific po_residual_bare!
    bvp_residual_bare!(d_bvp, outm, Xm, p, T)
    return out
end

"""
$(TYPEDSIGNATURES)

Core shooting residual computation for DiscretizedBVP.
This implements the same logic as BifurcationKit's po_residual_bare! for ShootingProblem.
"""
@views function bvp_residual_bare!(bvp::DiscretizedBVP{<:BVPModel, <:Shooting}, outm, xm, pars, T)
    model = get_model(bvp)
    sh = get_cache(bvp) # TODO this is a hack for now
    M = sh.M
    #TODO must use VI
    if ~BK.isparallel(sh)
        for ii in 1:M-1
            ip1 = ii+1
            outm[:, ii] .= BK.evolve(sh.flow, xm[:, ii], pars, sh.ds[ii] * T).u .- xm[:, ip1]
        end
        outm[:, M] .= model.g(xm[:, 1], BK.evolve(sh.flow, xm[:, M], pars, sh.ds[M] * T).u, pars)
    else
        solOde = BK.evolve(sh.flow, xm, pars, sh.ds .* T)
        for ii in 1:M-1
            ip1 = ii+1
            outm[:, ii] .= @views solOde[ii][2] .- xm[:, ip1]
        end
        outm[:, M] .= model.g(xm[:, 1], solOde[M][2], pars)
    end
end
