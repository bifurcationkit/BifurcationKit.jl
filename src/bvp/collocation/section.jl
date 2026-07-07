using LinearAlgebra: mul!, dot

"""
$(TYPEDSIGNATURES)

Updates the phase condition section for a DiscretizedPO using Collocation.
This initializes or updates `ϕ`, `∂ϕ`, and `∇phase` based on the new reference solution `x`.
"""
@views function BK.updatesection!(d_po::DiscretizedPO{<:POModel, <:Collocation}, x::AbstractVector, par)
    @debug "[BVP.Collocation] update section"
    n = state_dimension(d_po)
    disc = get_discretizer(d_po)
    (; Ntst, m) = disc
    
    # Extract caches and section
    cache = get_cache(d_po)
    mesh_cache = cache.mesh_cache
    coll_cache = cache.coll_cache
    section = d_po.section

    # 1. Update the "normals" (ϕ)
    # x contains the state and the period at the end. We only want the state.
    section.ϕ .= x[1:length(section.ϕ)]
    ϕ = section.ϕ

    # 2. Update ∂ϕ (derivative of ϕ at Gauss points)
    L, ∂L = get_Ls(mesh_cache)
    ϕm = get_time_slices(d_po, ϕ) # size (n, Ntst*m+1)
    pϕ = get_tmp(coll_cache.∂gj, ϕm) # zeros(𝒯, n, m)
    
    rg = axes(ϕm, 2)[1:m+1] # 1:m+1
    @inbounds for j in 1:Ntst
        mul!(pϕ, ϕm[:, rg], ∂L)
        section.∂ϕ[:, (j-1)*m .+ (1:m)] .= pϕ
        rg = rg .+ m
    end

    # 3. Update ∇phase (gradient of the phase condition)
    ω = get_gauss_weight(mesh_cache)
    rg_phase = 1:n
    coll_cache.∇phase .= 0
    @inbounds for j = 1:Ntst
        for k₁ = 1:m+1
            for l = 1:m
                coll_cache.∇phase[rg_phase] .+= (L[k₁, l] * ω[l]) .* section.∂ϕ[:, (j-1)*m + l]
            end
            if k₁ < m + 1
                rg_phase = rg_phase .+ n
            end
        end
    end
    
    return d_po
end
