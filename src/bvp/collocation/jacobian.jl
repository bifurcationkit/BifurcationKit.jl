# Jacobian interface for BVP collocation

@views function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, ::BK.DenseAnalytical, u, pars) where {Tmodel}
    mesh_cache = get_cache(d_bvp).mesh_cache
    m, Ntst = size(mesh_cache)
    n = state_dimension(d_bvp)
    
    Jcoll = zeros(eltype(u), n_mesh_pts(m, Ntst)*n, n_mesh_pts(m, Ntst)*n)
    VF = BK.BifurcationProblem((x, p) -> get_model(d_bvp).F(x, p), zeros(0), nothing, nothing; inplace = false)
    
    bvp_jacobian_dense!(Jcoll, d_bvp, VF, u, pars; _compute_borders = Val(false))
                        
    um = get_time_slices(d_bvp, u)
    u0 = um[:, 1]
    uf = um[:, end]
    model = get_model(d_bvp)
    Jcoll[end-n+1:end, 1:n] .= FD.jacobian(z -> model.g(z, uf, pars), u0)
    Jcoll[end-n+1:end, end-n+1:end] .= FD.jacobian(z -> model.g(u0, z, pars), uf)
    
    return Jcoll
end

@views function bvp_jacobian(d_po::DiscretizedPO{Tmodel, <: Collocation}, ::BK.DenseAnalytical, u, pars) where {Tmodel}
    n = state_dimension(d_po)
    n_unknowns = length(d_po)
    J = zeros(eltype(u), n_unknowns, n_unknowns)
    VF = BK.BifurcationProblem((x, p) -> get_model(d_po).F(x, p), zeros(0), [1.0], 1; inplace = false)
    
    bvp_jacobian_dense!(J, d_po, VF, u, pars; ∂ϕ = d_po.section.∂ϕ)
    
    return J
end

function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, ::BK.FullSparse, u::AbstractVector, pars) where {Tmodel}
    mesh_cache = get_cache(d_bvp).mesh_cache
    m, Ntst = size(mesh_cache)
    n = state_dimension(d_bvp)
    
    # Needs a BlockArray properly initialized, but we just call the skeleton for now
    VF = BK.BifurcationProblem((x, p) -> get_model(d_bvp).F(x, p), zeros(0), [1.0], 1; inplace = false)
    J = bvp_jacobian_sparse_blocks!(nothing, d_bvp, VF, u, pars; _compute_borders = Val(false))
    
    # We would add the boundary conditions here later
    return J
end

function bvp_jacobian(d_po::DiscretizedPO{Tmodel, <: Collocation}, ::BK.FullSparse, u::AbstractVector, pars) where {Tmodel}
    VF = BK.BifurcationProblem((x, p) -> get_model(d_po).F(x, p), zeros(0), nothing, nothing; inplace = false)
    J = bvp_jacobian_sparse_blocks!(nothing, d_po, VF, u, pars; ∂ϕ = d_po.section.∂ϕ)
    return J
end