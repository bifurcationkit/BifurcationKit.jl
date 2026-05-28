const POModel{Tf, 𝒯} = BVP.BVPModel{Tf, Nothing, 𝒯}

struct CollocationDisc <: AbstractBoundaryValueDiscretization
    Ntst::Int
    m::Int
    meshadapt::Bool
    K::Float64
end
CollocationDisc(; Ntst::Int = 20, m::Int = 4, meshadapt::Bool = false, K = 100.0) =
    CollocationDisc(Ntst, m, meshadapt, K)

Base.size(disc::CollocationDisc) = (disc.m, disc.Ntst)

function periodic_bc!(out, X, pars)
    @views @. out[:, end] = X[:, end] - X[:, 1]
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
    po_coll = Collocation(Ntst, m; N = n, prob_vf, meshadapt, K)
    return BVP.DiscretizedBVP(model, disc, (; po_coll))
end

Base.size(d_bvp::BVP.DiscretizedBVP{<:POModel,<:CollocationDisc}) = size(d_bvp.cache.po_coll)
Base.length(d_bvp::BVP.DiscretizedBVP{<:POModel,<:CollocationDisc}) = length(d_bvp.cache.po_coll)
@inline get_discretization(d_bvp::BVP.DiscretizedBVP) = d_bvp.discretizer
@inline getperiod(d_bvp::BVP.DiscretizedBVP, X, p = nothing) = X[end]
get_time_slices(d_bvp::BVP.DiscretizedBVP{<:POModel,<:CollocationDisc}, X) =
    get_time_slices(d_bvp.cache.po_coll, X)

function po_residual(d_bvp::BVP.DiscretizedBVP{<:POModel,<:CollocationDisc}, X, p)
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
    d_bvp::BVP.DiscretizedBVP{<:POModel,<:CollocationDisc},
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
    d_bvp::BVP.DiscretizedBVP{<:POModel,<:CollocationDisc},
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
    d_bvp::BVP.DiscretizedBVP{<:POModel,<:CollocationDisc},
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

# BVPBifProblem methods
_getvectortype(::BVP.BVPBifProblem{Tbvp,Tjac,Tu}) where {Tbvp,Tjac,Tu} = Tu
getu0(prob::BVP.BVPBifProblem) = prob.u0
getparams(prob::BVP.BVPBifProblem) = prob.params
getlens(prob::BVP.BVPBifProblem) = prob.lens
getparam(prob::BVP.BVPBifProblem) = _get(prob.params, prob.lens)
setparam(prob::BVP.BVPBifProblem, p0) = set(prob.params, prob.lens, p0)
isinplace(::BVP.BVPBifProblem) = false
is_symmetric(::BVP.BVPBifProblem) = false
has_adjoint(::BVP.BVPBifProblem) = false
residual(prob::BVP.BVPBifProblem, x, p) = po_residual(prob.d_bvp, x, p)
jacobian(prob::BVP.BVPBifProblem, x, p) = po_jacobian(prob.d_bvp, prob.jacobian, x, p)
record_from_solution(prob::BVP.BVPBifProblem) = prob.recordFromSolution
record_from_solution(prob::BVP.BVPBifProblem, x, p; k...) = prob.recordFromSolution(x, p; k...)
plot_solution(prob::BVP.BVPBifProblem) = prob.plotSolution
@inline update!(prob::BVP.BVPBifProblem, args...; kwargs...) = prob.update!(args...; kwargs...)

function re_make(
    prob::BVP.BVPBifProblem;
    d_bvp = prob.d_bvp,
    jacobian = prob.jacobian,
    u0 = getu0(prob),
    params = getparams(prob),
    lens = getlens(prob),
    plotSolution = prob.plotSolution,
    recordFromSolution = prob.recordFromSolution,
    save_solution = prob.save_solution,
    update! = prob.update!,
)
    return BVP.BVPBifProblem(
        d_bvp,
        jacobian,
        u0,
        params,
        lens,
        plotSolution,
        recordFromSolution,
        save_solution,
        update!,
    )
end

function save_solution(prob::BVP.BVPBifProblem, x, pars)
    po_coll = prob.d_bvp.cache.po_coll
    if po_coll.meshadapt
        return POSolutionAndState(
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
