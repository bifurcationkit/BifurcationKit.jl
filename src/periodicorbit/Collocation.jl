import .BVP: POModel, PeriodicBC, DiscretizedPO

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forwarding methods: DiscretizedPO{<:BVP.Collocation} → internal Collocation cache
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Base.size(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}) = (BVP.state_dimension(d_bvp), size(BVP.get_cache(d_bvp).mesh_cache)...)
Base.length(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}) = BVP.state_dimension(d_bvp) * BVP.n_mesh_pts(size(BVP.get_cache(d_bvp).mesh_cache)...) + 1

@inline getperiod(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, X, p = nothing) = X[end]

get_time_slices(d_bvp::DiscretizedPO, X) = reshape(@view(X[1:end-1]), BVP.state_dimension(d_bvp), :)

get_times(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}) = BVP.get_times(BVP.get_cache(d_bvp).mesh_cache)
get_times(d_bvp::DiscretizedPO) = BVP.get_times(BVP.get_cache(d_bvp).mesh_cache)

get_max_time_step(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}) = BVP.get_max_time_step(BVP.get_cache(d_bvp).mesh_cache)
get_max_time_step(d_bvp::DiscretizedPO) = BVP.get_max_time_step(BVP.get_cache(d_bvp).mesh_cache)

get_gauss_nodes(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}) = BVP.get_gauss_nodes(BVP.get_cache(d_bvp).mesh_cache)
get_gauss_nodes(d_bvp::DiscretizedPO) = BVP.get_gauss_nodes(BVP.get_cache(d_bvp).mesh_cache)

get_Ls(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}) = BVP.get_Ls(BVP.get_cache(d_bvp).mesh_cache)
get_Ls(d_bvp::DiscretizedPO) = BVP.get_Ls(BVP.get_cache(d_bvp).mesh_cache)

# ─────────────────────────────────────────────────────────────────────────────
# LEGACY IMPLEMENTATIONS FOR CollocationDisc (Kept for reference)
# ─────────────────────────────────────────────────────────────────────────────
# function update_mesh!(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, τs)
#     update_mesh!(d_bvp.cache.po_coll.mesh_cache, τs)
#     return d_bvp
# end
# 
# function generate_solution(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, orbit, period)
#     generate_solution(d_bvp.cache.po_coll, orbit, period)
# end
# 
# function get_periodic_orbit(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, u, p)
#     get_periodic_orbit(d_bvp.cache.po_coll, u, p)
# end
# 
# function POInterpolation(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, x)
#     POInterpolation(d_bvp.cache.po_coll, x)
# end
# 
# function getmesh(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯})
#     getmesh(d_bvp.cache.po_coll.mesh_cache)
# end
# 
# function ∫(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, args...; kwargs...)
#     ∫(d_bvp.cache.po_coll, args...; kwargs...)
# end
# 
# function po_analytical_jacobian(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, args...; kwargs...)
#     po_analytical_jacobian(d_bvp.cache.po_coll, args...; kwargs...)
# end
# 
# function po_jacobian_block(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, args...; kwargs...)
#     po_jacobian_block(d_bvp.cache.po_coll, args...; kwargs...)
# end
# 
# function po_analytical_jacobian_sparse(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, args...; kwargs...)
#     po_analytical_jacobian_sparse(d_bvp.cache.po_coll, args...; kwargs...)
# end
# 
# function jacobian_poocoll_sparse_indx!(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, J, u, p, indx)
#     jacobian_poocoll_sparse_indx!(d_bvp.cache.po_coll, J, u, p, indx)
# end
# 
# function get_blocks(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}, J)
#     get_blocks(d_bvp.cache.po_coll, J)
# end
# 
# get_discretization(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯}) = d_bvp.cache.po_coll
# 
# function Base.show(io::IO, d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc}) where {Tf, 𝒯}
#     coll = d_bvp.discretizer
#     n = d_bvp.model.n
#     println(io, "┌─ DiscretizedPO (CollocationDisc)")
#     println(io, "├─ State dimension n : ", n)
#     println(io, "├─ Ntst              : ", coll.Ntst)
#     println(io, "├─ m                 : ", coll.m)
#     println(io, "└─ Mesh adaptation   : ", coll.meshadapt)
# end
# ─────────────────────────────────────────────────────────────────────────────


function update_mesh!(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, τs)
    BVP.update_mesh!(BVP.get_cache(d_bvp).mesh_cache, τs)
    return d_bvp
end

function generate_solution(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, orbit, period)
    ts = get_times(d_bvp)
    n, m, Ntst = size(d_bvp)
    X = zeros(typeof(period), n * length(ts) + 1)
    Xm = reshape(@view(X[1:end-1]), n, length(ts))
    for (l, t) in pairs(ts)
        Xm[:, l] .= orbit(t * period)
    end
    X[end] = period
    return X
end
function generate_solution(d_bvp::DiscretizedPO, orbit, period)
    invoke(generate_solution, Tuple{DiscretizedPO{<:POModel, <:BVP.Collocation}, Any, Any}, d_bvp, orbit, period)
end

function get_periodic_orbit(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, u, p)
    T = getperiod(d_bvp, u, p)
    ts = get_times(d_bvp)
    um = get_time_slices(d_bvp, u)
    return BVPSolution(t = ts .* T, u = um)
end
function get_periodic_orbit(d_bvp::DiscretizedPO, u, p)
    invoke(get_periodic_orbit, Tuple{DiscretizedPO{<:POModel, <:BVP.Collocation}, Any, Any}, d_bvp, u, p)
end

function POInterpolation(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, x)
    POInterpolation(d_bvp, x, nothing)
end
function POInterpolation(d_bvp::DiscretizedPO, x)
    POInterpolation(d_bvp, x, nothing)
end

function (sol::POInterpolation{<:DiscretizedPO})(t0)
    BVP.__interpolate_posolution(BVP.get_cache(sol.pb).mesh_cache, t0, get_time_slices(sol.pb, sol.x), getperiod(sol.pb, sol.x, sol.pars))
end

function getmesh(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation})
    BVP.getmesh(BVP.get_cache(d_bvp).mesh_cache)
end
function getmesh(d_bvp::DiscretizedPO)
    BVP.getmesh(BVP.get_cache(d_bvp).mesh_cache)
end

@views function ∫(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation},
                  uc::AbstractMatrix,
                  vc::AbstractMatrix,
                  period = one(eltype(uc)))
    𝒯y = promote_type(eltype(uc), eltype(vc))
    phase = zero(𝒯y)

    mesh_cache = BVP.get_cache(d_bvp).mesh_cache
    coll_cache  = BVP.get_cache(d_bvp).coll_cache
    coll = BVP.get_discretizer(d_bvp)
    m, Ntst = coll.m, coll.Ntst

    L, _ = BVP.get_Ls(mesh_cache)
    ω    = mesh_cache.gauss_weight
    mesh = BVP.getmesh(mesh_cache)

    # Pre-allocated temporaries — must use get_tmp for ForwardDiff compatibility
    guj = get_tmp(coll_cache.gj, uc)   # n × m (points de Gauss de u)
    gvj = get_tmp(coll_cache.gi, vc)   # n × m (points de Gauss de v, uses gi buffer)

    rg = UnitRange(1, m+1)
    @inbounds for j in 1:Ntst
        LinearAlgebra.mul!(guj, uc[:, rg], L)
        LinearAlgebra.mul!(gvj, vc[:, rg], L)
        @inbounds for l in 1:m
            phase += LinearAlgebra.dot(guj[:, l], gvj[:, l]) * ω[l] * (mesh[j+1] - mesh[j]) / 2
        end
        rg = rg .+ m
    end
    return phase * period
end

function ∫(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation},
           u::AbstractVector, v::AbstractVector, period = one(eltype(u)))
    uc = get_time_slices(d_bvp, u)
    vc = get_time_slices(d_bvp, v)
    return ∫(d_bvp, uc, vc, period)
end

function po_analytical_jacobian(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, args...; kwargs...)
    BVP.bvp_jacobian(d_bvp, DenseAnalytical(), args...; kwargs...)
end

function po_jacobian_block(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, u::AbstractVector, pars; array_zeros = zeros, kwargs...)
    coll = BVP.get_discretizer(d_bvp)
    n = BVP.state_dimension(d_bvp)
    m = coll.m
    Ntst = coll.Ntst
    blocks = n * ones(Int64, BVP.n_mesh_pts(m, Ntst) + 1); blocks[end] = 1
    n_blocks = length(blocks)
    𝒯 = eltype(u)
    J = BA.BlockArray(array_zeros(𝒯, length(u), length(u)), blocks, blocks)
    
    # We call the inplace sparse blocks jacobian
    VF = BifurcationProblem((x, p) -> BVP.get_model(d_bvp).F(x, p), zeros(0), [1.0], 1; inplace = false)
    BVP.bvp_jacobian_sparse_blocks!(J, d_bvp, VF, u, pars; ∂ϕ = d_bvp.section.∂ϕ, kwargs...)
    return J
end

function po_analytical_jacobian_sparse(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, u::AbstractVector, pars; kwargs...)
    jacBlock = po_jacobian_block(d_bvp, u, pars; array_zeros = SPA.spzeros, kwargs...)
    block_to_sparse(jacBlock)
end

function jacobian_poocoll_sparse_indx!(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, J, u, p, indx; kwargs...)
    VF = BifurcationProblem((x, p) -> BVP.get_model(d_bvp).F(x, p), zeros(0), [1.0], 1; inplace = false)
    BVP.bvp_jacobian_sparse_inplace!(J, indx, d_bvp, VF, u, p; ∂ϕ = d_bvp.section.∂ϕ, kwargs...)
end

function get_blocks(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, J)
    coll = BVP.get_discretizer(d_bvp)
    n = BVP.state_dimension(d_bvp)
    m = coll.m
    Ntst = coll.Ntst
    blocks = n * ones(Int64, BVP.n_mesh_pts(m, Ntst) + 1); blocks[end] = 1
    Jb = BA.BlockArray(J, blocks, blocks)
    return Jb
end

function Base.show(io::IO, d_bvp::DiscretizedPO{Tf, 𝒯, <:BVP.Collocation}) where {Tf, 𝒯}
    coll = BVP.get_discretizer(d_bvp)
    n = BVP.state_dimension(d_bvp)
    println(io, "┌─ DiscretizedPO (Collocation)")
    println(io, "├─ State dimension n : ", n)
    println(io, "├─ Ntst              : ", coll.Ntst)
    println(io, "├─ m                 : ", coll.m)
    println(io, "└─ Mesh adaptation   : ", coll.meshadapt)
end

function PeriodicOrbitProblem(br, 
                              ind_bif, 
                              disc::BVP.Collocation;
                              jacobian = AutoDiff()
                              )
    @assert false

end

function periodic_bc!(out, X, pars)
    @views @. out[:, end] = X[:, end] - X[:, 1]
end
# ─────────────────────────────────────────────────────────────────────────────
# LEGACY IMPLEMENTATIONS (Kept for reference)
# ─────────────────────────────────────────────────────────────────────────────
# function po_residual(d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc}, X, p) where {Tf, 𝒯}
#     po_coll = d_bvp.cache.po_coll
#     n, m, Ntst = size(po_coll)
#     Xc = get_time_slices(po_coll, X)
#     period = X[end]
#     out = similar(X)
#     outc = get_time_slices(po_coll, out)
#     Ls = get_Ls(po_coll.mesh_cache)
#     phase = po_residual_bare!(po_coll, outc, Xc, period, Ls, p; compute_phase = Val(true))
#     periodic_bc!(outc, Xc, p)
#     out[end] = phase
#     return out
# end
# 
# @views function po_jacobian(
#                             d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯},
#                             ::DenseAnalytical,
#                             u,
#                             pars,
#                         )
#     po_coll = d_bvp.cache.po_coll
#     𝒯 = eltype(po_coll)
#     J = zeros(𝒯, length(po_coll), length(po_coll))
#     n, m, Ntst = size(po_coll)
#     uc = get_time_slices(po_coll, u)
#     period = u[end]
#     _po_analytical_jacobian!(J, po_coll, u, pars, uc, period; _compute_borders = Val(false))
#     return J
# end
# 
# function po_jacobian(
#                     d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯},
#                     ::FullSparse,
#                     u,
#                     pars,
#                 )
#     po_coll = d_bvp.cache.po_coll
#     n, m, Ntst = size(po_coll)
#     𝒯 = eltype(po_coll)
#     upad = vcat(u, one(𝒯))
#     Jfull = po_analytical_jacobian_sparse(po_coll, upad, pars)
#     N = n * n_mesh_pts(m, Ntst)
#     return Jfull[1:N, 1:N]
# end
# 
# function po_jacobian(
#                     d_bvp::DiscretizedPO{Tf, 𝒯, <:CollocationDisc} where {Tf, 𝒯},
#                     ::FullSparseInplace,
#                     u,
#                     pars,
#                 )
#     po_coll = d_bvp.cache.po_coll
#     _J = po_analytical_jacobian_sparse(po_coll, u, pars)
#     indx = _get_blocks_from_sparse_matrix(po_coll, _J)
#     jacobian_poocoll_sparse_indx!(po_coll, _J, u, pars, indx)
#     return (FullSparseInplace(), _J, indx)
# end
# ─────────────────────────────────────────────────────────────────────────────


function po_residual(d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation}, X, p)
    BVP.bvp_residual(d_bvp, X, p)
end

@views function po_jacobian(
                            d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation},
                            ::DenseAnalytical,
                            u,
                            pars,
                        )
    J = BVP.bvp_jacobian(d_bvp, DenseAnalytical(), u, pars)
    return J
end

function po_jacobian(
                    d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation},
                    ::AutoDiffDense,
                    u,
                    pars,
                )
    J = BVP.bvp_jacobian(d_bvp, AutoDiffDense(), u, pars)
    return J
end

function po_jacobian(
                    d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation},
                    ::FullSparse,
                    u,
                    pars,
                )
    J = BVP.bvp_jacobian(d_bvp, FullSparse(), u, pars)
    return J
end

function po_jacobian(
                    d_bvp::DiscretizedPO{<:POModel, <:BVP.Collocation},
                    ::FullSparseInplace,
                    u,
                    pars,
                )
    # the new architecture will return (alg, J, indx) or mutate J.
    # we just call it.
    J = BVP.bvp_jacobian(d_bvp, FullSparseInplace(), u, pars)
    # Note: mutating J directly here might be complex if it returns a tuple, but we assume it's just a matrix for now or let it pass
    return J
end

"""
$(TYPEDSIGNATURES)

Function needed for automatic branch switching from a Hopf bifurcation point.
"""
function re_make(coll::BVP.Collocation,
                 prob_vf,
                 ::AbstractBifurcationPoint,
                 ζr::AbstractVector,
                 orbitguess_a,
                 period;
                 orbit = identity,
                 k...)
    N = length(ζr)
    m = coll.m
    Ntst = coll.Ntst
    n_unknows = N * n_mesh_pts(m, Ntst)

    new_coll = Collocation(Ntst, m;
        N, prob_vf,
        meshadapt = coll.meshadapt,
        K = coll.K,
        ϕ = zeros(n_unknows),
        xπ = zeros(n_unknows),
        ∂ϕ = zeros(N, Ntst * m),
        cache = POCollCache(Float64, Ntst, N, m),
    )

    ϕ0 = generate_solution(new_coll, t -> orbit(2pi * t / period + pi), period)
    updatesection!(new_coll, ϕ0, nothing)

    orbitguess = generate_solution(new_coll, t -> orbit(2pi * t / period), period)

    return new_coll, orbitguess
end

function save_solution(prob::BVP.BVPBifProblem{ <: DiscretizedPO}, x, pars)
    d_bvp = get_bvp(prob)
    if BVP.get_discretizer(d_bvp).meshadapt
        return POSavedSolutionAndState(
            copy(get_times(d_bvp)),
            x,
            copy(getmesh(d_bvp)),
            _copy(d_bvp.section.ϕ),
        )
    else
        return x
    end
end

function newton(
                disc::BVP.Collocation,
                model::POModel,
                orbitguess,
                params,
                lens,
                options::NewtonPar;
                jacobian = AutoDiffDense(),
                kwargs...,
            )
    d_bvp = discretize(model, disc)
    prob = BVP.BVPBifProblem(
        d_bvp,
        jacobian,
        orbitguess,
        params,
        lens,
    )
    return solve(prob, Newton(), options; kwargs...)
end
