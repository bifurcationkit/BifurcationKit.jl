# Jacobian interface for BVP collocation

function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, ::BK.DenseAnalytical, u, pars; kwargs...) where {Tmodel}
    mesh_cache = get_cache(d_bvp).mesh_cache
    m, Ntst = size(mesh_cache)
    n = state_dimension(d_bvp)
    
    T = eltype(u)
    Jcoll = zeros(T, n_mesh_pts(m, Ntst)*n, n_mesh_pts(m, Ntst)*n)
    pb = BK.BifurcationProblem((x, p) -> get_model(d_bvp).F(x, p), zeros(T, 0), [one(T)], 1; inplace = false)
    
    bvp_jacobian_dense!(Jcoll, d_bvp, pb.VF, u, pars; _compute_borders = Val(false), kwargs...)
                        
    um = get_time_slices(d_bvp, u)
    u0 = um[:, 1]
    uf = um[:, end]
    model = get_model(d_bvp)
    Jcoll[end-n+1:end, 1:n] .= FD.jacobian(z -> model.g(z, uf, pars), u0)
    Jcoll[end-n+1:end, end-n+1:end] .= FD.jacobian(z -> model.g(u0, z, pars), uf)
    
    return Jcoll
end

function bvp_jacobian(d_po::DiscretizedPO{Tmodel, <: Collocation}, ::BK.DenseAnalytical, u, pars; kwargs...) where {Tmodel}
    n = state_dimension(d_po)
    n_unknowns = length(d_po)
    T = eltype(u)
    J = zeros(T, n_unknowns, n_unknowns)
    pb = BK.BifurcationProblem((x, p) -> get_model(d_po).F(x, p), zeros(T, 0), [one(T)], 1; inplace = false)
    
    bvp_jacobian_dense!(J, d_po, pb.VF, u, pars; ∂ϕ = d_po.section.∂ϕ, kwargs...)
    
    return J
end

function bvp_jacobian(d_bvp::DiscretizedBVP{Tmodel, <: Collocation}, ::BK.FullSparse, u::AbstractVector, pars; kwargs...) where {Tmodel}
    mesh_cache = get_cache(d_bvp).mesh_cache
    m, Ntst = size(mesh_cache)
    n = state_dimension(d_bvp)
    𝒯 = eltype(u)
    
    blocks = n * ones(Int, n_mesh_pts(m, Ntst))
    Jblock = BK.BA.BlockArray(BK.SPA.spzeros(𝒯, length(d_bvp), length(d_bvp)), blocks, blocks)
    
    pb = BK.BifurcationProblem((x, p) -> get_model(d_bvp).F(x, p), zeros(𝒯, 0), [one(𝒯)], 1; inplace = false)
    bvp_jacobian_sparse_blocks!(Jblock, d_bvp, pb.VF, u, pars; _compute_borders = Val(false), kwargs...)
    
    Jsparse = BK.block_to_sparse(Jblock)
    
    um = get_time_slices(d_bvp, u)
    u0 = um[:, 1]
    uf = um[:, end]
    model = get_model(d_bvp)
    Jsparse[end-n+1:end, 1:n] .= FD.jacobian(z -> model.g(z, uf, pars), u0)
    Jsparse[end-n+1:end, end-n+1:end] .= FD.jacobian(z -> model.g(u0, z, pars), uf)
    
    return Jsparse
end

function bvp_jacobian(d_po::DiscretizedPO{Tmodel, <: Collocation}, ::BK.FullSparse, u::AbstractVector, pars; kwargs...) where {Tmodel}
    mesh_cache = get_mesh_cache(d_po)
    m, Ntst = size(mesh_cache)
    n = state_dimension(d_po)
    𝒯 = eltype(u)
    
    blocks = n * ones(Int, n_mesh_pts(m, Ntst) + 1); blocks[end] = 1
    Jblock = BK.BA.BlockArray(BK.SPA.spzeros(𝒯, length(d_po), length(d_po)), blocks, blocks)
    
    pb = BK.BifurcationProblem((x, p) -> get_model(d_po).F(x, p), zeros(𝒯, 0), [one(𝒯)], 1; inplace = false)
    bvp_jacobian_sparse_blocks!(Jblock, d_po, pb.VF, u, pars; ∂ϕ = d_po.section.∂ϕ, kwargs...)
    
    return BK.block_to_sparse(Jblock)
end