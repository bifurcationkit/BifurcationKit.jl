@views function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, ::BK.DenseAnalytical, u, pars) where {Tmodel}
    disc = get_discretizer(d_bvp)
    model = get_model(d_bvp)
    coll = d_bvp.cache.po_coll # TODO: a bit of a hack for now
    𝒯 = eltype(coll)
    Jcoll = zeros(𝒯, length(coll), length(coll))
    n, m, Ntst = size(coll)
    um = reshape(u, n, 1 + Ntst * m)
    period = one(𝒯)
    BK._collocation_analytical_jacobian!(Jcoll, 
                                coll, 
                                u, 
                                pars,
                                um,
                                period;
                                _compute_borders = Val(false))
    u0 = um[:, 1]
    uf = um[:, end]
    Jcoll[end-n+1:end, 1:n] .= FD.jacobian(z -> model.g(z, uf, pars), u0)
    Jcoll[end-n+1:end, end-n+1:end] .= FD.jacobian(z -> model.g(u0, z, pars), uf)
    return Jcoll
end

function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, ::BK.FullSparse, u::AbstractVector, pars) where {Tmodel}
    model = get_model(d_bvp)
    coll = d_bvp.cache.po_coll
    n, m, Ntst = size(coll)
    N = n * (1 + Ntst * m)
    𝒯 = eltype(coll)

    upad = vcat(u, one(𝒯))
    Jfull = BK.po_analytical_jacobian_sparse(coll, upad, pars)
    J = Jfull[1:N, 1:N]

    um = reshape(u, n, 1 + Ntst * m)
    u0 = um[:, 1]
    uf = um[:, end]
    J[end-n+1:end, 1:n] = FD.jacobian(z -> model.g(z, uf, pars), u0)
    J[end-n+1:end, end-n+1:end] = FD.jacobian(z -> model.g(u0, z, pars), uf)

    return J
end