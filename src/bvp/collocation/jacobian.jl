@views function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, ::BK.DenseAnalytical, u, pars) where {Tmodel}
    disc = get_discretizer(d_bvp)
    model = get_model(d_bvp)
    coll = d_bvp.cache.po_coll # TODO: a bit of a hack for now
    𝒯 = eltype(coll)
    Jcoll = zeros(𝒯, length(coll), length(coll))
    n, m, Ntst = size(coll)
    uc = reshape(u, n, 1 + Ntst * m)
    period = one(𝒯)
    BK._po_analytical_jacobian!(Jcoll, 
                                coll, 
                                u, 
                                pars,
                                uc,
                                period;
                                _compute_borders = Val(false))
    u0 = uc[:, 1]
    uf = uc[:, end]
    Jcoll[end-n+1:end, 1:n] .= ForwardDiff.jacobian(z -> model.g(z, uf, pars), u0)
    Jcoll[end-n+1:end, end-n+1:end] .= ForwardDiff.jacobian(z -> model.g(u0, z, pars), uf)
    return Jcoll
end

