# Trapezoid Residual Implementation
#
# This file implements bvp_residual for Trapeze discretization

function bvp_residual(bvp::DiscretizedBVP{ <: BVPModel, <: Trapeze}, X, p)
    model = bvp.model
    disc = bvp.discretizer
    po_trap = bvp.cache.po_trap
    n = state_dimension(model)
    M = disc.M

    interval = get_time_interval(model)
    δT = interval[2] - interval[1]

    # Extract time slices
    Xc = reshape(@view(X[1:n*M]), n, M)

    # Get output buffer; element type follows X (covers Dual numbers in AD)
    out = similar(X)
    outc = reshape(@view(out[1:n*M]), n, M)

    # Sequential trapezoid scheme for M-1 intervals using potrap_scheme!.
    # M time points span [t0, tf] via M-1 intervals with normalized mesh weights.
    # tmp is a dedicated buffer (separate from outc) holding F(u_i) across steps.
    # potrap_scheme!(po_trap, dest, u_curr, u_prev, par, h/2, tmp)
    #   requires tmp = F(u_prev) on entry, writes F(u_curr) into tmp on exit.
    tmp = similar(X, n)
    BK.residual!(po_trap.prob_vf, tmp, @view(Xc[:, 1]), p)  # seed: F(u_1)

    for i in 1:M-1
        h_i = δT * BK.get_time_step(disc.mesh, i)
        BK.potrap_scheme!(po_trap, @view(outc[:, i]), @view(Xc[:, i+1]), @view(Xc[:, i]), p, h_i/2, tmp)
        # outc[:, i] = u_{i+1} - u_i - (h_i/2)*(F(u_i) + F(u_{i+1}))
        # tmp now holds F(u_{i+1}) for the next iteration
    end

    # Boundary condition: g(u(0), u(T), p) = 0
    u0 = @view Xc[:, 1]
    uT = @view Xc[:, M]
    outc[:, M] .= model.g(u0, uT, p)

    return out
end
