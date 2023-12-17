@inline has_adjoint_MF(::WrapPOColl) = false
@inline has_hessian(::WrapPOColl) = false

d2F(pbwrap::WrapPOColl, x, p, dx1, dx2) = d2PO(z -> pbwrap.prob(z, p), x, dx1, dx2)

function Base.transpose(J::FloquetWrapper{ <: PeriodicOrbitOCollProblem})
    @set J.jacpb = transpose(J.jacpb)
end

function Base.adjoint(J::FloquetWrapper{ <: PeriodicOrbitOCollProblem})
    @set J.jacpb = adjoint(J.jacpb)
end

function jacobian_period_doubling(pbwrap::WrapPOColl, x, par)
    N, m, Ntst = size(pbwrap.prob)
    Jac = jacobian(pbwrap, x, par)
    # put the PD boundary condition
    @set Jac.jacpb = copy(Jac.jacpb)
    J = Jac.jacpb
    J[end-N:end-1, 1:N] .= I(N)
    @set Jac.jacpb = J[1:end-1,1:end-1]
end

function jacobian_neimark_sacker(pbwrap::WrapPOColl, x, par, ω)
    N, m, Ntst = size(pbwrap.prob)
    Jac = jacobian(pbwrap, x, par)
    # put the NS boundary condition
    J = Complex.(copy(Jac.jacpb))
    J[end-N:end-1, end-N:end-1] .= UniformScaling(cis(ω))(N)
    @set Jac.jacpb = J[1:end-1,1:end-1]
end

function continuation(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::Lens,
                    options_cont::ContinuationPar = br.contparams ;
                    bdlinsolver = MatrixBLS(),
                    detect_codim2_bifurcation::Int = 0,
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
    biftype = br.specialpoint[ind_bif].type

    # options to detect codim2 bifurcations
    compute_eigen_elements = options_cont.detect_bifurcation > 0
    _options_cont = detect_codim2_parameters(detect_codim2_bifurcation, options_cont; kwargs...)

    if biftype == :bp
        return continuation_coll_fold(br, ind_bif, lens2, _options_cont; compute_eigen_elements = compute_eigen_elements, kwargs... )
    elseif biftype == :pd
        return continuation_coll_pd(br, ind_bif, lens2, _options_cont; compute_eigen_elements = compute_eigen_elements, kwargs... )
    elseif biftype == :ns
        return continuation_coll_ns(br, ind_bif, lens2, _options_cont; compute_eigen_elements = compute_eigen_elements, kwargs... )
    else
        throw("We continue only Fold / PD / NS points of periodic orbits. Please report this error on the website.")
    end
    nothing
end

"""
$(SIGNATURES)

Continuation of curve of fold bifurcations of periodic orbits computed using collocation method.

# Arguments
- `br` branch of periodic orbits computed with a [`PeriodicOrbitTrapProblem`](@ref)
- `ind_bif` index of the fold point
- `lens2::Lens` second parameter axis
- `options_cont` parameters to be used by a regular [`continuation`](@ref)
"""
function continuation_coll_fold(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::Lens,
                    options_cont::ContinuationPar = br.contparams ;
                    start_with_eigen = false,
                    detect_codim2_bifurcation::Int = 0,
                    bdlinsolver = MatrixBLS(),
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
    biftype = br.specialpoint[ind_bif].type
    bifpt = br.specialpoint[ind_bif]

    # we get the collocation problem
    probco = getprob(br).prob

    if get_plot_backend() == BK_Makie()
        plotsol = (ax,x,p;ax1 = nothing, k...) -> br.prob.plotSolution(ax,x.u,p;k...)
    else
        plotsol = (x,p;k...) -> br.prob.plotSolution(x.u,p;k...)
    end
    probcoFold = BifurcationProblem((x, p) -> probco(x, p), bifpt, getparams(br), getlens(br);
                J = (x, p) -> FloquetWrapper(probco, ForwardDiff.jacobian(z -> probco(z, p), x), x, p),
                d2F = (x, p, dx1, dx2) -> d2PO(z -> probco(z, p), x, dx1, dx2),
                plot_solution = plotsol
                )

    options_foldpo = @set options_cont.newton_options.linsolver = FloquetWrapperLS(options_cont.newton_options.linsolver)

    # perform continuation
    br_fold_po = continuation_fold(probcoFold,
        br, ind_bif, lens2,
        options_foldpo;
        start_with_eigen = start_with_eigen,
        bdlinsolver = FloquetWrapperBLS(bdlinsolver),
        kind = FoldPeriodicOrbitCont(),
        kwargs...
        )
    correct_bifurcation(br_fold_po)
end

"""
$(SIGNATURES)

Continuation of curve of period-doubling bifurcations of periodic orbits computed using collocation method.

# Arguments
- `br` branch of periodic orbits computed with a [`PeriodicOrbitTrapProblem`](@ref)
- `ind_bif` index of the fold point
- `lens2::Lens` second parameter axis
- `options_cont` parameters to be used by a regular [`continuation`](@ref)
"""
function continuation_coll_pd(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::Lens,
                    options_cont::ContinuationPar = br.contparams ;
                    alg = br.alg,
                    start_with_eigen = false,
                    detect_codim2_bifurcation::Int = 0,
                    bdlinsolver = MatrixBLS(),
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
    bifpt = br.specialpoint[ind_bif]
    biftype = bifpt.type

    @assert biftype == :pd "Please open an issue on BifurcationKit website"

    pdpointguess = pd_point(br, ind_bif)

    # we copy the problem for not mutating the one passed by the user
    coll = deepcopy(br.prob.prob)
    N, m, Ntst = size(coll)

    # get the PD eigenvectors
    par = setparam(br, bifpt.param)
    jac = jacobian(br.prob, bifpt.x, par)
    J = jac.jacpb
    nj = size(J, 1)
    J[end, :] .= rand(nj) # must be close to kernel
    J[:, end] .= rand(nj)
    J[end, end] = 0
    # enforce PD boundary condition
    J[end-N:end-1, 1:N] .= I(N)
    rhs = zeros(nj); rhs[end] = 1
    q = J  \ rhs; q = q[1:end-1]; q ./= norm(q) # ≈ ker(J)
    p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

    # perform continuation
    continuation_pd(br.prob, alg,
        pdpointguess, setparam(br, pdpointguess.p),
        getlens(br), lens2,
        p, q,
        options_cont;
        kwargs...,
        # detect_codim2_bifurcation = detect_codim2_bifurcation,
        kind = PDPeriodicOrbitCont(),
        )
end


"""
$(SIGNATURES)

Continuation of curve of Neimark-Sacker bifurcations of periodic orbits computed using collocation method.

# Arguments
- `br` branch of periodic orbits computed with a [`PeriodicOrbitTrapProblem`](@ref)
- `ind_bif` index of the fold point
- `lens2::Lens` second parameter axis
- `options_cont` parameters to be used by a regular [`continuation`](@ref)
"""
function continuation_coll_ns(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::Lens,
                    options_cont::ContinuationPar = br.contparams ;
                    alg = br.alg,
                    start_with_eigen = false,
                    detect_codim2_bifurcation::Int = 0,
                    bdlinsolver = MatrixBLS(),
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: WrapPOColl}
    bifpt = br.specialpoint[ind_bif]
    biftype = bifpt.type

    @assert biftype == :ns "We continue only NS points of Periodic orbits for now"

    nspointguess = ns_point(br, ind_bif)

    # we copy the problem for not mutating the one passed by the user
    coll = deepcopy(br.prob.prob)
    N, m, Ntst = size(coll)

    # get the NS eigenvectors
    par = setparam(br, bifpt.param)
    jac = jacobian(br.prob, bifpt.x, par)
    J = Complex.(copy(jac.jacpb))
    nj = size(J, 1)
    J[end, :] .= rand(nj) # must be close to eigensapce
    J[:, end] .= rand(nj)
    J[end, end] = 0
    # enforce NS boundary condition
    λₙₛ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
    J[end-N:end-1, end-N:end-1] .= UniformScaling(exp(λₙₛ))(N)

    rhs = zeros(nj); rhs[end] = 1
    q = J  \ rhs; q = q[1:end-1]; q ./= norm(q) # ≈ ker(J)
    p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

    # perform continuation
    continuation_ns(br.prob, alg,
        nspointguess, setparam(br, nspointguess.p[1]),
        getlens(br), lens2,
        p, q,
        options_cont;
        kwargs...,
        # detect_codim2_bifurcation = detect_codim2_bifurcation,
        kind = NSPeriodicOrbitCont(),
        )
end
