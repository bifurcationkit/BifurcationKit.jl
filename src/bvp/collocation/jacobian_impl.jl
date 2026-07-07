# Jacobian implementations for collocation method
# These functions will host the heavy analytical logic.

@views function bvp_jacobian_dense!(J, prob, VF, u::AbstractVector{𝒯}, pars;
                             ∂ϕ = nothing, _compute_borders = Val(true), 
                             ρD = one(𝒯), ρF = one(𝒯), ρI = zero(𝒯)) where {𝒯}
    n = BVP.state_dimension(prob)
    mesh_cache = BVP.get_cache(prob).mesh_cache
    m = length(mesh_cache.gauss_weight)
    mesh_cache = BVP.get_cache(prob).mesh_cache
    coll_cache = BVP.get_cache(prob).coll_cache
    Ntst = length(BVP.getmesh(mesh_cache)) - 1
    nJ = length(prob)
    
    L, ∂L = BVP.get_Ls(mesh_cache) # L is of size (m+1, m)
    mesh = BVP.getmesh(mesh_cache)
    ω = mesh_cache.gauss_weight
    phase = zero(𝒯)
    
    um = _bvp_coll_get_Xm(prob, u, n, Ntst, m)
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
            BK.jacobian!(VF, J0, pj[:, l], pars)

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

function bvp_jacobian_sparse_blocks!(J, prob, VF, u, pars;
                                     ∂ϕ = nothing, _transpose = Val(false),
                                     ρD = 1.0, ρF = 1.0, ρI = 0.0)
    error("Analytical sparse block jacobian for BVP collocation not yet implemented.")
    return J
end

function bvp_jacobian_sparse_inplace!(J, indx, prob, VF, u, pars;
                                      ∂ϕ = nothing, _transpose = Val(false),
                                      ρD = 1.0, ρF = 1.0, ρI = 0.0)
    error("Analytical sparse inplace jacobian for BVP collocation not yet implemented.")
    return J
end
