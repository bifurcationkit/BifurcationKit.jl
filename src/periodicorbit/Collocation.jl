struct PeriodicBC; end
const POModel{Tf, 𝒯} = BVP.BVPModel{Tf, PeriodicBC, 𝒯}
const DiscretizedPO{Tf, 𝒯, Tdisc, Tcache} = BVP.DiscretizedBVP{POModel{Tf, 𝒯}, Tdisc, Tcache} # TODO remove, use instead:
 # struct DiscretizedPO <: AbstractDiscretizedPO
    #     d_bvp::DiscretizedBVP
    #     section
    #     mesh
    # end

# cf test/.../stuartLandauCollocationDisc.jl

struct CollocationDisc <: BVP.AbstractDiscretizer
    Ntst::Int
    m::Int
    meshadapt::Bool
    K::Float64
end
CollocationDisc(; Ntst::Int = 20, m::Int = 4, meshadapt::Bool = false, K = 100.0) =
    CollocationDisc(Ntst, m, meshadapt, K)

@inline get_mesh_size(coll::CollocationDisc) = coll.Ntst

function periodic_bc!(out, X, pars)
    @views @. out[:, end] = X[:, end] - X[:, 1]
end

function POModel(F, 𝒯 = Float64; k...)
    BVP.BVPModel(F, PeriodicBC(); t0 = zero(𝒯), tf = one(𝒯), k...)
end

function Base.show(io::IO, model::POModel)
    println(io, "┌─ POModel")
    println(io, "├─ State dimension n : ", model.n == 0 ? "unspecified" : model.n)
    println(io, "└─ Vector field F    : ", typeof(model.F).name.name)
end

function discretize(model::POModel, disc::CollocationDisc)
    n = model.n
    (; Ntst, m, meshadapt, K) = disc
    prob_vf = BifurcationProblem(
        (u, p) -> model.F(u, p),
        zeros(n),
        (dummy = 0.0,),
        (@optic _.dummy);
        inplace = false,
        record_from_solution = (x, p; k...) -> nothing,
    )
    po_coll = Collocation(Ntst, m; N = n, prob_vf, meshadapt, K) # TODO: remnove this and add section, mesh cache, etc
    return BVP.DiscretizedBVP(model, disc, (; po_coll))
end

Base.size(d_bvp::DiscretizedPO{ <: CollocationDisc}) = size(d_bvp.cache.po_coll)
Base.length(d_bvp::DiscretizedPO{ <: CollocationDisc}) = length(d_bvp.cache.po_coll)
@inline getperiod(d_bvp::DiscretizedPO{ <: CollocationDisc}, X, p = nothing) = X[end]
get_time_slices(d_bvp::DiscretizedPO{ <: CollocationDisc}, X) = get_time_slices(d_bvp.cache.po_coll, X)
get_time_slices(d_bvp::DiscretizedPO, X) = get_time_slices(BVP.get_cache(d_bvp).po_coll, X)

# function Base.show(io::IO, d_bvp::DiscretizedPO{ <: CollocationDisc})
#     println(io, "┌─ DiscretizedPO")
#     println(io, "├─ State dimension : ", state_dimension(d_bvp))
#     println(io, "├─ Total unknowns  : ", length(d_bvp))
#     println(io, "├─ Model           : POModel")
#     print(io,   "└─ Discretizer     : CollocationDisc")
# end

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Forwarding methods: DiscretizedPO{<:CollocationDisc} → internal Collocation cache
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

get_times(d_bvp::DiscretizedPO{ <: CollocationDisc}) = get_times(d_bvp.cache.po_coll)
get_times(d_bvp::DiscretizedPO) = get_times(BVP.get_cache(d_bvp).po_coll)
get_max_time_step(d_bvp::DiscretizedPO{ <: CollocationDisc}) = get_max_time_step(d_bvp.cache.po_coll)
get_gauss_nodes(d_bvp::DiscretizedPO{ <: CollocationDisc}) = get_gauss_nodes(d_bvp.cache.po_coll)
get_gauss_nodes(d_bvp::DiscretizedPO) = get_gauss_nodes(BVP.get_cache(d_bvp).po_coll)
get_Ls(d_bvp::DiscretizedPO{ <: CollocationDisc}) = get_Ls(d_bvp.cache.po_coll.mesh_cache)
get_Ls(d_bvp::DiscretizedPO) = get_Ls(BVP.get_cache(d_bvp).po_coll.mesh_cache)



# what follows is really bad for now














function update_mesh!(d_bvp::DiscretizedPO{ <: CollocationDisc}, τs)
    update_mesh!(d_bvp.cache.po_coll.mesh_cache, τs)
    return d_bvp
end

function generate_solution(d_bvp::DiscretizedPO{ <: CollocationDisc}, orbit, period)
    generate_solution(d_bvp.cache.po_coll, orbit, period)
end
function generate_solution(d_bvp::DiscretizedPO, orbit, period)
    generate_solution(BVP.get_cache(d_bvp).po_coll, orbit, period)
end

function get_periodic_orbit(d_bvp::DiscretizedPO{ <: CollocationDisc}, u, p)
    get_periodic_orbit(d_bvp.cache.po_coll, u, p)
end
function get_periodic_orbit(d_bvp::DiscretizedPO, u, p)
    get_periodic_orbit(BVP.get_cache(d_bvp).po_coll, u, p)
end

function POInterpolation(d_bvp::DiscretizedPO{ <: CollocationDisc}, x)
    POInterpolation(d_bvp.cache.po_coll, x)
end
function POInterpolation(d_bvp::DiscretizedPO, x)
    POInterpolation(BVP.get_cache(d_bvp).po_coll, x)
end

function getmesh(d_bvp::DiscretizedPO{ <: CollocationDisc})
    getmesh(d_bvp.cache.po_coll.mesh_cache)
end
function getmesh(d_bvp::DiscretizedPO)
    getmesh(BVP.get_cache(d_bvp).po_coll.mesh_cache)
end

function ∫(d_bvp::DiscretizedPO{ <: CollocationDisc}, args...; kwargs...)
    ∫(d_bvp.cache.po_coll, args...; kwargs...)
end

function po_analytical_jacobian(d_bvp::DiscretizedPO{ <: CollocationDisc}, args...; kwargs...)
    po_analytical_jacobian(d_bvp.cache.po_coll, args...; kwargs...)
end

function po_jacobian_block(d_bvp::DiscretizedPO{ <: CollocationDisc}, args...; kwargs...)
    po_jacobian_block(d_bvp.cache.po_coll, args...; kwargs...)
end

function po_analytical_jacobian_sparse(d_bvp::DiscretizedPO{ <: CollocationDisc}, args...; kwargs...)
    po_analytical_jacobian_sparse(d_bvp.cache.po_coll, args...; kwargs...)
end

function jacobian_poocoll_sparse_indx!(d_bvp::DiscretizedPO{ <: CollocationDisc}, J, u, p, indx)
    jacobian_poocoll_sparse_indx!(d_bvp.cache.po_coll, J, u, p, indx)
end

function get_blocks(d_bvp::DiscretizedPO{ <: CollocationDisc}, J)
    get_blocks(d_bvp.cache.po_coll, J)
end

get_discretization(d_bvp::DiscretizedPO{ <: CollocationDisc}) = d_bvp.cache.po_coll

function Base.show(io::IO, d_bvp::DiscretizedPO{Tf, 𝒯, <: CollocationDisc}) where {Tf, 𝒯}
    coll = d_bvp.discretizer
    n = d_bvp.model.n
    println(io, "┌─ DiscretizedPO (CollocationDisc)")
    println(io, "├─ State dimension n : ", n)
    println(io, "├─ Ntst              : ", coll.Ntst)
    println(io, "├─ m                 : ", coll.m)
    println(io, "└─ Mesh adaptation   : ", coll.meshadapt)
end

function PeriodicOrbitProblem(br, 
                              ind_bif, 
                              disc::CollocationDisc;
                              jacobian = AutoDiff()
                              )
    @assert false

end

function po_residual(d_bvp::DiscretizedPO{Tf, 𝒯, <: CollocationDisc}, X, p) where {Tf, 𝒯}
    po_coll = d_bvp.cache.po_coll
    n, m, Ntst = size(po_coll)
    Xc = get_time_slices(po_coll, X)
    period = X[end]
    out = similar(X)
    outc = get_time_slices(po_coll, out)
    Ls = get_Ls(po_coll.mesh_cache)
    phase = po_residual_bare!(po_coll, outc, Xc, period, Ls, p; compute_phase = Val(true))
    periodic_bc!(outc, Xc, p)
    out[end] = phase
    return out
end

@views function po_jacobian(
                            d_bvp::DiscretizedPO{ <: CollocationDisc},
                            ::DenseAnalytical,
                            u,
                            pars,
                        )
    po_coll = d_bvp.cache.po_coll
    𝒯 = eltype(po_coll)
    J = zeros(𝒯, length(po_coll), length(po_coll))
    n, m, Ntst = size(po_coll)
    uc = get_time_slices(po_coll, u)
    period = u[end]
    _po_analytical_jacobian!(J, po_coll, u, pars, uc, period; _compute_borders = Val(false))
    return J
end

function po_jacobian(
                    d_bvp::DiscretizedPO{ <: CollocationDisc},
                    ::FullSparse,
                    u,
                    pars,
                )
    po_coll = d_bvp.cache.po_coll
    n, m, Ntst = size(po_coll)
    𝒯 = eltype(po_coll)
    upad = vcat(u, one(𝒯))
    Jfull = po_analytical_jacobian_sparse(po_coll, upad, pars)
    N = n * n_mesh_pts(m, Ntst)
    return Jfull[1:N, 1:N]
end

function po_jacobian(
                    d_bvp::DiscretizedPO{ <: CollocationDisc},
                    ::FullSparseInplace,
                    u,
                    pars,
                )
    po_coll = d_bvp.cache.po_coll
    _J = po_analytical_jacobian_sparse(po_coll, u, pars)
    indx = _get_blocks_from_sparse_matrix(po_coll, _J)
    jacobian_poocoll_sparse_indx!(po_coll, _J, u, pars, indx)
    return (FullSparseInplace(), _J, indx)
end

residual(prob::BVP.BVPBifProblem{ <: DiscretizedPO}, x, p) = po_residual(get_bvp(prob), x, p)
jacobian(prob::BVP.BVPBifProblem{ <: DiscretizedPO}, x, p) = po_jacobian(get_bvp(prob), prob.jacobian, x, p)
# disambiguation for the ambiguity between the method above and the generic one in BVPBifProblem.jl
function jacobian(prob::BVP.BVPBifProblem{ <: BVP.DiscretizedBVP{BVP.BVPModel{Tf, PeriodicBC, 𝒯}}}, x, p) where {Tf, 𝒯}
    po_jacobian(get_bvp(prob), prob.jacobian, x, p)
end

"""
$(TYPEDSIGNATURES)

Function needed for automatic branch switching from a Hopf bifurcation point.
"""
function re_make(coll::CollocationDisc,
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
    po_coll = get_bvp(prob).cache.po_coll
    if po_coll.meshadapt
        return POSavedSolutionAndState(
            copy(get_times(po_coll)),
            x,
            copy(getmesh(po_coll.mesh_cache)),
            _copy(po_coll.ϕ),
        )
    else
        return x
    end
end

function newton(
                disc::CollocationDisc,
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
