
# ─────────────────────────────────────────────────────────────────────────────
# Helpers: type-dispatched extraction of T, Xm, BC, and ∂ϕ
# These allow __bvp_residual_collocation to handle both DiscretizedBVP and
# DiscretizedPO without any conditional branches in the inner loop.
# ─────────────────────────────────────────────────────────────────────────────

# Period T: fixed interval for BVP, free unknown (last element of X) for PO
function _bvp_coll_get_T(d_bvp::DiscretizedBVP, X)
    t0, tf = get_time_interval(get_model(d_bvp))
    tf - t0
end
_bvp_coll_get_T(::DiscretizedPO, X) = X[end]

# State matrix Xm of size (n, Ntst*m+1):
#   BVP → X is purely the state
#   PO  → X = [state..., T], strip the period


# Boundary condition written into the last column of outm:
#   BVP → general g(u(0), u(T), p) = 0
#   PO  → periodicity u(T) - u(0) = 0
function _bvp_coll_apply_bc!(d_bvp::DiscretizedBVP, outm, Xm, p)
    outm[:, end] .= get_model(d_bvp).g(Xm[:, 1], Xm[:, end], p)
end
function _bvp_coll_apply_bc!(::DiscretizedPO, outm, Xm, p)
    @. outm[:, end] = Xm[:, end] - Xm[:, 1]
end

# Derivative of the phase reference ∂ϕ (n × Ntst*m):
#   BVP → no phase condition, return nothing
#   PO  → stored in the section
_bvp_coll_get_∂ϕ(::DiscretizedBVP) = nothing
_bvp_coll_get_∂ϕ(d_po::DiscretizedPO) = d_po.section.∂ϕ

# ─────────────────────────────────────────────────────────────────────────────
# New clean implementation — no periodicorbit structs.
#
# Requires get_cache(d_bvp) to expose:
#   cache.mesh_cache  :: MeshCollocationCache  (mesh, L, ∂L, Gauss weights)
#   cache.coll_cache    :: CollocationCache           (pre-allocated temporaries)
#
# This will replace the legacy path once discretize() is updated to build
# these caches directly rather than wrapping a Collocation object.
# ─────────────────────────────────────────────────────────────────────────────
@views function __bvp_residual_collocation(d_bvp, X, p, compute_phase::Val{CP} = Val(false)) where {CP}
    model  = get_model(d_bvp)
    disc   = get_discretizer(d_bvp)
    cache  = get_cache(d_bvp)         # needs: cache.mesh_cache, cache.coll_cache
    n      = state_dimension(d_bvp)
    Ntst   = get_ntst(disc)
    m      = get_m(disc)
    𝒯      = eltype(X)

    T  = _bvp_coll_get_T(d_bvp, X)
    Xm = get_time_slices(d_bvp, X)

    out  = similar(X)
    outm = get_time_slices(d_bvp, out)

    # Mesh data from cache
    mesh  = getmesh(cache.mesh_cache)          # Ntst+1 normalized time nodes in [0, 1]
    L, ∂L = get_Ls(cache.mesh_cache)           # Lagrange interpolation/differentiation matrices
    ω     = get_gauss_weight(cache.mesh_cache) # Gauss quadrature weights (length m)
    ∂ϕ    = _bvp_coll_get_∂ϕ(d_bvp)           # (n, Ntst*m) or nothing

    # Pre-allocated temporaries (must be compatible with eltype(X) for ForwardDiff)
    pj   = get_tmp(cache.coll_cache.gj,  X)  # (n, m)  — values at Gauss points
    ∂pj  = get_tmp(cache.coll_cache.∂gj, X)  # (n, m)  — derivatives at Gauss points
    tmp  = get_tmp(cache.coll_cache.tmp,  X)  # (n,)    — buffer for F(u, p)

    phase = zero(𝒯)
    rg    = UnitRange(1, m + 1)  # column range for current mesh interval

    @inbounds for j in 1:Ntst
        dt = (mesh[j + 1] - mesh[j]) / 2   # half-width of interval j
        α  = T * dt                          # time-scaling factor

        # Evaluate solution and its derivative at the m Gauss points
        # via Lagrange interpolation over the m+1 nodes of interval j
        LA.mul!( pj, Xm[:, rg], L)   # pj[:,l]  = u(τₗ)   for l = 1..m
        LA.mul!(∂pj, Xm[:, rg], ∂L)  # ∂pj[:,l] = u'(τₗ)  for l = 1..m

        # Collocation equations: u'(τₗ) = T·dt · F(u(τₗ), p)
        for l in Base.OneTo(m)
            tmp .= model.F(pj[:, l], p)              # write F(u,p) into pre-allocated buffer
            @. outm[:, rg[l]] = ∂pj[:, l] - α * tmp
        end

        # Phase condition: ∫₀¹ <u(t), ∂ϕ(t)> dt  (accumulated in the same loop = free)
        if CP === true && !isnothing(∂ϕ)
            @inbounds for l in Base.OneTo(m)
                phase += LA.dot(pj[:, l], ∂ϕ[:, (j - 1) * m + l]) * ω[l]
            end
        end

        rg = rg .+ m
    end

    # Boundary condition (type-specific: general g for BVP, periodicity for PO)
    _bvp_coll_apply_bc!(d_bvp, outm, Xm, p)

    return out, phase / T
end

# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

# BVP: clean new path
function bvp_residual(d_bvp::DiscretizedBVP{<: BVPModel, <: Collocation}, X, p)
    out, _ = __bvp_residual_collocation(d_bvp, X, p, Val(false))
    return out
end

# PO: clean new path — phase condition is returned in out[end]
function bvp_residual(d_po::DiscretizedPO{<: BVPModel, <: Collocation}, X, p)
    out, phase_normalized = __bvp_residual_collocation(d_po, X, p, Val(true))
    out[end] = phase_normalized
    return out
end
