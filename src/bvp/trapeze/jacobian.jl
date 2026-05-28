@views function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Trapeze}, ::BK.Dense, u0::AbstractVector, pars) where {Tmodel}
    model = get_model(d_bvp)
    disc = get_discretizer(d_bvp)
    trap = d_bvp.cache.po_trap
    u0m = get_time_slices(d_bvp, u0)
    interval = get_time_interval(model)
    δT = interval[2] - interval[1]

    M, N = size(trap)
    Aγ = BK.BA.BlockArray(BK.SPA.spzeros(M * N, M * N), N * ones(Int64, M), N * ones(Int64, M))
    Iₙ = BK.SPA.SparseMatrixCSC(1.0 * LA.I, N, N)

    tmpJ = BK.jacobian(trap.prob_vf, u0m[:, 1], pars)
    h = δT * BK.get_time_step(disc.mesh, 1)
    Aγ[BK.BA.Block(1, 1)] = -Iₙ - (h / 2) .* tmpJ
    Aγ[BK.BA.Block(1, 2)] = Iₙ - (h / 2) .* BK.jacobian(trap.prob_vf, u0m[:, 2], pars)

    for ii in 2:(M - 1)
        h = δT * BK.get_time_step(disc.mesh, ii)
        tmpJ = BK.jacobian(trap.prob_vf, u0m[:, ii], pars)
        Aγ[BK.BA.Block(ii, ii)] = -Iₙ - (h / 2) .* tmpJ
        Aγ[BK.BA.Block(ii, ii + 1)] = Iₙ - (h / 2) .* BK.jacobian(trap.prob_vf, u0m[:, ii + 1], pars)
    end

    u1 = u0m[:, 1]
    uM = u0m[:, M]
    Aγ[BK.BA.Block(M, 1)] = FD.jacobian(z -> model.g(z, uM, pars), u1)
    Aγ[BK.BA.Block(M, M)] = FD.jacobian(z -> model.g(u1, z, pars), uM)

    return Aγ|>Array
end