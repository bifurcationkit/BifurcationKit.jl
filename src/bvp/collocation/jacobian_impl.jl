# Jacobian implementations for collocation method
# These functions will host the heavy analytical logic.

@views function bvp_jacobian_dense!(J, prob, VF, u::AbstractVector{𝒯}, pars;
                             ∂ϕ = nothing, _compute_borders = Val(true), _transpose = Val(false),
                             ρD = one(𝒯), ρF = one(𝒯), ρI = zero(𝒯)) where {𝒯}
    n = BVP.state_dimension(prob)
    mesh_cache = BVP.get_cache(prob).mesh_cache
    m = length(mesh_cache.gauss_weight)
    coll_cache = BVP.get_cache(prob).coll_cache
    Ntst = length(BVP.getmesh(mesh_cache)) - 1
    nJ = length(prob)
    
    L, ∂L = BVP.get_Ls(mesh_cache) # L is of size (m+1, m)
    mesh = BVP.getmesh(mesh_cache)
    ω = mesh_cache.gauss_weight
    phase = zero(𝒯)
    
    um = get_time_slices(prob, u)
    period = _bvp_coll_get_T(prob, u)

    pj = zeros(𝒯, n, m)
    In = coll_cache.In # this helps greatly the for loop for J0 below
    J0 = zeros(𝒯, n, n)

    # put boundary condition
    J[nJ-n:nJ-1, nJ-n:nJ-1] .= In
    J[nJ-n:nJ-1, 1:n] .= (-1) .* In

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)
    rgNx = UnitRange(1, n)
    rgNy = UnitRange(1, n)

    for j in 1:Ntst
        dt = (mesh[j+1] - mesh[j]) / 2
        α = period * dt
        LinearAlgebra.mul!(pj, um[:, rg], L) # pj ≈ (L * uj')'
        # put the jacobian of the vector field
        for l in 1:m
            _rgX = rgNx .+ (l-1)*n
            
            if _transpose isa Val{false} || _transpose == false
                BK.jacobian!(VF, J0, pj[:, l], pars)
            else
                J0 .= LinearAlgebra.transpose(BK.jacobian(VF, pj[:,l], pars))
            end

            for l2 in 1:m+1
                J[_rgX, rgNy .+ (l2-1)*n ] .= @. (-α * L[l2, l] * ρF) * J0 +
                                                (ρD * ∂L[l2, l] - α * L[l2, l] * ρI) * In
            end
            if _compute_borders isa Val{true} || _compute_borders == true
                # add derivative w.r.t. the period
                BK.residual!(VF, J[_rgX, nJ], pj[:, l], pars)
                J[_rgX, nJ] .*= (-dt)

                if ∂ϕ !== nothing
                    phase += LinearAlgebra.dot(pj[:, l], ∂ϕ[:, (j-1)*m + l]) * ω[l]
                end
            end
        end
        rg = rg .+ m
        rgNx = rgNx .+ (m * n)
        rgNy = rgNy .+ (m * n)
    end

    if _compute_borders isa Val{true} || _compute_borders == true
        J[end, begin:end-1] .= coll_cache.∇phase ./ period
        J[nJ, nJ] = -phase / period^2
    end
    return J
end

@views function bvp_jacobian_sparse_blocks!(J::BK.BA.BlockArray, prob, VF, u::AbstractVector{𝒯}, pars;
                                     ∂ϕ = nothing, _compute_borders = Val(true), _transpose = Val(false),
                                     ρD = one(𝒯), ρF = one(𝒯), ρI = zero(𝒯)) where {𝒯}
    n = BVP.state_dimension(prob)
    mesh_cache = BVP.get_cache(prob).mesh_cache
    m = length(mesh_cache.gauss_weight)
    Ntst = length(BVP.getmesh(mesh_cache)) - 1
    coll_cache = BVP.get_cache(prob).coll_cache
    
    n_blocks = size(J.blocks, 1)

    L, ∂L = BVP.get_Ls(mesh_cache) # L is of size (m+1, m)
    ω = mesh_cache.gauss_weight
    mesh = BVP.getmesh(mesh_cache)
    
    um = get_time_slices(prob, u)
    period = _bvp_coll_get_T(prob, u)

    pj = zeros(𝒯, n, m)
    phase = zero(𝒯)

    # vector field jacobian placeholder
    J0 = BK.jacobian(VF, u[1:n], pars) # this works for sparse
    if !(J0 isa BK.SPA.AbstractSparseMatrix)
        J0 = BK.SPA.sparse(J0)
    end
    In = coll_cache.In # typically an identity matrix, maybe sparse
    
    # put boundary condition
    view(J, BK.BA.Block(BVP.n_mesh_pts(m, Ntst), BVP.n_mesh_pts(m, Ntst))) .= In
    view(J, BK.BA.Block(BVP.n_mesh_pts(m, Ntst), 1)) .= (-1) .* In

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)
    rgNx = UnitRange(1, n)
    iₚ = 0
    i∇ = 1

    for j in 1:Ntst
        dt = (mesh[j+1] - mesh[j]) / 2
        α = period * dt
        LinearAlgebra.mul!(pj, um[:, rg], L) # pj ≈ (L * uj')'
        # put the jacobian of the vector field
        for l in 1:m
            if _transpose isa Val{false} || _transpose == false
                BK.jacobian!(VF, J0, pj[:, l], pars)
            else
                J0 .= LinearAlgebra.transpose(BK.jacobian(VF, pj[:, l], pars))
            end

            for l2 in 1:m+1
                view(J, BK.BA.Block(l + (j-1)*m, l2 + (j-1)*m)) .= @. (-α * L[l2, l] * ρF) * J0 +
                                                (ρD * ∂L[l2, l] - α * L[l2, l] * ρI) * In
            end
            
            if _compute_borders isa Val{true} || _compute_borders == true
                # add derivative w.r.t. the period
                BK.residual!(VF, view(J, BK.BA.Block(l + (j-1)*m, n_blocks)), pj[:, l], pars)
                view(J, BK.BA.Block(l + (j-1)*m, n_blocks)) .*= (-dt)

                if ∂ϕ !== nothing
                    phase += LinearAlgebra.dot(pj[:, l], ∂ϕ[:, iₚ + l]) * ω[l]
                    view(J, BK.BA.Block(n_blocks, i∇)) .= coll_cache.∇phase[rgNx]' ./ period
                end
            end
            i∇ += 1; rgNx = rgNx .+ n
        end
        iₚ += m
        rg = rg .+ m
    end
    
    if _compute_borders isa Val{true} || _compute_borders == true
        if ∂ϕ !== nothing
            view(J, BK.BA.Block(n_blocks, i∇)) .= coll_cache.∇phase[rgNx]' ./ period # last bit
            J.blocks[end, end][1] = -phase / period^2
        end
    end
    return J
end

@views function bvp_jacobian_sparse_inplace!(J::BK.SPA.SparseMatrixCSC, indx, prob, VF, u::AbstractVector{𝒯}, pars;
                                      ∂ϕ = nothing, _compute_borders = Val(true), _transpose = Val(false),
                                      ρD = one(𝒯), ρF = one(𝒯), ρI = zero(𝒯)) where {𝒯}
    n = BVP.state_dimension(prob)
    mesh_cache = BVP.get_cache(prob).mesh_cache
    m = length(mesh_cache.gauss_weight)
    Ntst = length(BVP.getmesh(mesh_cache)) - 1
    coll_cache = BVP.get_cache(prob).coll_cache

    L, ∂L = BVP.get_Ls(mesh_cache) # L is of size (m+1, m)
    ω = mesh_cache.gauss_weight
    mesh = BVP.getmesh(mesh_cache)
    
    period = _bvp_coll_get_T(prob, u)
    um = get_time_slices(prob, u)
    
    phase = zero(𝒯)
    pj = zeros(𝒯, n, m)
    In = BK.SPA.sparse(LinearAlgebra.I(n))
    
    J0 = BK.jacobian(VF, um[1:n], pars)
    if !(J0 isa BK.SPA.AbstractSparseMatrix)
        J0 = BK.SPA.sparse(J0)
    end
    tmpJ = copy(J0 + In)

    # put boundary condition
    J.nzval[indx[BVP.n_mesh_pts(m, Ntst), BVP.n_mesh_pts(m, Ntst)]] = In.nzval
    J.nzval[indx[BVP.n_mesh_pts(m, Ntst), 1]] = -In.nzval

    # loop over the mesh intervals
    rg = UnitRange(1, m+1)
    rgNx = UnitRange(1, n)

    for j in 1:Ntst
        LinearAlgebra.mul!(pj, um[:, rg], L) # pj ≈ (L * uj')'
        dt = (mesh[j+1]-mesh[j]) / 2
        α = period * dt
        # put the jacobian of the vector field
        for l in 1:m
            if _transpose isa Val{false} || _transpose == false
                @inbounds BK.jacobian!(VF, J0, pj[:, l], pars)
            else
                J0 .= LinearAlgebra.transpose(BK.jacobian(VF, pj[:,l], pars))
            end

            for l2 in 1:m+1
                tmpJ .= (-α * L[l2, l]) .* (ρF .* J0 + ρI * LinearAlgebra.I) .+ ρD * (∂L[l2, l] .* In)
                J.nzval[indx[ l + (j-1) * m, l2 + (j-1)*m] ] .= (tmpJ).nzval
            end
            
            if _compute_borders isa Val{true} || _compute_borders == true
                # add derivative w.r.t. the period
                J[rgNx .+ (l-1)*n, end] .= BK.residual(VF, pj[:,l], pars) .* (-dt)
                if ∂ϕ !== nothing
                    phase += LinearAlgebra.dot(pj[:, l], ∂ϕ[:, (j-1)*m + l]) * ω[l]
                end
            end
        end
        rg = rg .+ m
        rgNx = rgNx .+ (m * n)
    end

    if _compute_borders isa Val{true} || _compute_borders == true
        if ∂ϕ !== nothing
            J[end, begin:end-1] .= coll_cache.∇phase ./ period
            J[end, end] = -phase / period^2
        end
    end
    return J
end
