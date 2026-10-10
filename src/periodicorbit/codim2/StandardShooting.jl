function d2F(wrapsh::PeriodicOrbitFunctionalSh, x, p, dx1, dx2)
    d2PO(z -> po_residual(get_discretization(wrapsh), z, p), x, dx1, dx2)
end

# if the jacobian is matrix based, use transpose
@inline function has_adjoint(wrap::PeriodicOrbitFunctionalSh{ <: Shooting{𝒯p, 𝒯j} }) where {𝒯p, 𝒯j}
    disc = get_discretization(wrap)
    return ~(𝒯j <: AbstractJacobianMatrix) || (isnothing(disc.flow.vjp) == false)
end
@inline has_jvp(wrap::PeriodicOrbitFunctionalSh) = has_jvp(get_discretization(wrap))

function jacobian_period_doubling(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{𝒯p, 𝒯j} }, x, par) where {𝒯p, 𝒯j}
    dx -> jacobian_pd_ns_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, 1, dx)
end

function jacobian_adjoint_period_doubling(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par)
    dx -> jacobian_adjoint_period_doubling_matrix_free(pbwrap, x, par, dx)
end

jacobian_adjoint_period_doubling_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, dx) = jacobian_adjoint_pd_ns_matrix_free(pbwrap, x, par, 1, dx)
jacobian_adjoint_neimark_sacker_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, ω, dx) = jacobian_adjoint_pd_ns_matrix_free(pbwrap, x, par, -cis(-ω), dx)

# matrix-based jacobian of PeriodicOrbitFunctionalSh with PD boundary condition and phase condition
function jacobian_period_doubling_with_border(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{𝒯p, 𝒯j} }, x, par) where {𝒯p, 𝒯j <: AbstractJacobianMatrix}
    M = get_mesh_size(get_discretization(pbwrap))
    N = div(length(x) - 1, M)
    Jac = jacobian(pbwrap, x, par)
    J = copy(Jac)
    # put the PD boundary condition
    J[end-N:end-1, 1:N] .= LA.I(N)
    return J
end

# matrix-based jacobian of PeriodicOrbitFunctionalSh with PD boundary condition without phase condition
function jacobian_period_doubling(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{𝒯p, 𝒯j} }, x, par) where {𝒯p, 𝒯j <: AbstractJacobianMatrix}
    J = jacobian_period_doubling_with_border(pbwrap, x, par)
    return J[begin:end-1, begin:end-1] # TODO: we still want a copy?
end

# matrix free linear operator associated to the monodromy whose zeros are used to detect PD/NS points
# no phase condition is encoded
@views function jacobian_pd_ns_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, α::𝒯, dx) where 𝒯
    sh = get_discretization(pbwrap)
    T  = getperiod(sh, x)
    M  = get_mesh_size(sh)
    N  = div(length(x) - 1, M)

    xc = get_time_slices(sh, x)
    dxc = reshape(dx, N, M)

    # variable to hold the computed result
    out = similar(dx, promote_type(VI.scalartype(dx), 𝒯))
    outc = reshape(out, N, M)

    # jacobian of the flow
    dflowDE(_x, _dx, _T) = ForwardDiff.derivative(z -> evolve(sh.flow, _x .+ z .* _dx, par, _T).u, 0)
    dflow(_x, _dx, _T) = dflowDE(_x, real.(_dx), _T) .+ im .* dflowDE(_x, imag.(_dx), _T)
    dflow(_x, _dx::T, _T) where {T <: AbstractArray{ <: Real}} = dflowDE(_x, _dx, _T)
    if ~isparallel(sh)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            # call jacobian of the flow, jacobian-vector product
            tmp = dflow(xc[:, ii], dxc[:, ii], sh.ds[ii] * T)
            if ii<M
                outc[:, ii] .= tmp .- dxc[:, ip1]
            else
                outc[:, ii] .= tmp .+ α .* dxc[:, ip1]
            end
        end
    else
        @assert false "WIP! No parallel matrix-free shooting for curve of PD/NS"
        # call jacobian of the flow, jacobian-vector product
        solOde = jvp(sh.flow, xc, par, dxc, sh.ds .* T)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            outc[:, ii] .= solOde[ii].du .+ vf(sh.flow, solOde[ii].u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
        end
    end
    return out
end

# matrix free adjoint linear operator associated to the monodromy whose zeros are used to detect PD/NS points
@views function jacobian_adjoint_pd_ns_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, α::𝒯, dx) where 𝒯
    sh = get_discretization(pbwrap)
    T  = getperiod(sh, x)
    M  = get_mesh_size(sh)
    N  = div(length(x) - 1, M)

    xc = get_time_slices(sh, x)
    dxc = reshape(dx, N, M)

    # variable to hold the computed result
    out = similar(dx, promote_type(VI.scalartype(dx), 𝒯))
    outc = reshape(out, N, M)

    # jacobian of the flow
    dflowDE(_x, _dx, _T) = vjp(sh.flow, _x, par, _dx, _T)
    dflow(_x, _dx, _T) = dflowDE(_x, real.(_dx), _T) .+ im .* dflowDE(_x, imag.(_dx), _T)
    dflow(_x, _dx::T, _T) where {T <: AbstractArray{<: Real}} = dflowDE(_x, _dx, _T)

    if ~isparallel(sh)
        for ii in 1:M
            im1 = (ii == 1) ? M : ii-1
            # call jacobian of the flow, jacobian-vector product
            tmp = dflow(xc[:, ii], dxc[:, ii], sh.ds[ii] * T)
            if ii==1
                outc[:, ii] .= tmp .+ α .* dxc[:, im1]
            else
                outc[:, ii] .= tmp .- dxc[:, im1]
            end
        end
    else
        @assert false "WIP! No parallel adjoint matrix-free shooting for curve of PD/NS"
        # call jacobian of the flow, jacobian-vector product
        solOde = jvp(sh.flow, xc, par, dxc, sh.ds .* T)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            outc[:, ii] .= solOde[ii].du .+ vf(sh.flow, solOde[ii].u, par) .* sh.ds[ii] .* dT .- dxc[:, ip1]
        end
    end
    return out
end

function jacobian_neimark_sacker(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{𝒯p, 𝒯j} }, x, par, ω) where {𝒯p, 𝒯j}
    dx -> jacobian_pd_ns_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, -cis(ω), dx)
end

function jacobian_neimark_sacker_with_border(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{Tp, Tj} }, x, par, ω) where {Tp, Tj <: AbstractJacobianMatrix}
    M = get_mesh_size(get_discretization(pbwrap))
    N = div(length(x) - 1, M)
    Jac = jacobian(pbwrap, x, par)
    # put the NS boundary condition
    J = Complex.(copy(Jac))
    J[end-N:end-1, 1:N] .*= cis(ω)
    return J
end

function jacobian_neimark_sacker(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{𝒯p, 𝒯j} }, x, par, ω) where {𝒯p, 𝒯j <: AbstractJacobianMatrix}
    J = jacobian_neimark_sacker_with_border(pbwrap, x, par, ω)
    return J[begin:end-1, begin:end-1]
end

function jacobian_adjoint_neimark_sacker(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, ω)
    dx -> jacobian_adjoint_neimark_sacker_matrix_free(pbwrap, x, par, ω, dx)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function continuation(br::AbstractResult{Tkind, Tprob}, #TODO same function in PeriodicOrbitCollocation
                      ind_bif::Int64,
                      lens2::AllOpticTypes,
                      options_cont::ContinuationPar = br.contparams ;
                      detect_codim2_bifurcation::Int = 0,
                      update_minaug_every_step = 1,
                      kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: PeriodicOrbitFunctionalSh}
    biftype = br.specialpoint[ind_bif].type
    # options to detect codim2 bifurcations
    compute_eigen_elements = options_cont.detect_bifurcation > 0
    _options_cont = modify_contparams_for_codim2(detect_codim2_bifurcation, options_cont; update_minaug_every_step, kwargs...)
    args = (br, ind_bif, lens2, _options_cont)
    kw = (; compute_eigen_elements, update_minaug_every_step, kwargs...)
    if biftype == :bp || biftype == :fold # TODO this is strongly type unstable
        return continuation_sh_fold(args...; kw... )
    elseif biftype == :pd
        return  continuation_sh_pd(args...; kw... )
    elseif biftype == :ns
        return  continuation_sh_ns(args...; kw... )
    end
    error("You passed the bifurcation type = $biftype.\nWe continue only Branch Point /Fold / PD / NS points of periodic orbits for now.")
end

"""
$(SIGNATURES)

Continuation of curve of fold bifurcations of periodic orbits computed using shooting method.

# Arguments
- `br` branch of periodic orbits computed with a [`Shooting`](@ref)
- `ind_bif` index of the fold point
- `lens2::AllOpticTypes` second parameter axis
- `options_cont` parameters to be used by a regular [`continuation`](@ref)
"""
function continuation_sh_fold(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::AllOpticTypes,
                    options_cont::ContinuationPar = br.contparams ;
                    bdlinsolver = MatrixBLS(),
                    Jᵗ = nothing,
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: PeriodicOrbitFunctionalSh}
    pbwrap = getprob(br)
    options_foldpo = options_cont
    br_fold_po = continuation_fold(
        pbwrap,
        br, ind_bif, lens2,
        options_foldpo;
        bdlinsolver,
        kind = FoldPeriodicOrbitCont(),
        kwargs...)
    return _correct_event_labels(br_fold_po)
end

"""
$(SIGNATURES)

Continuation of curve of period-doubling bifurcations of periodic orbits computed using shooting method.

# Arguments
- `br` branch of periodic orbits computed with a [`Shooting`](@ref)
- `ind_bif` index of the PD point
- `lens2::AllOpticTypes` second parameter axis
- `options_cont` parameters to be used by a regular [`continuation`](@ref)
"""
@views function continuation_sh_pd(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::AllOpticTypes,
                    options_cont::ContinuationPar = br.contparams ;
                    alg = getalg(br),
                    start_with_eigen = Val(false), # used to filter the option
                    bdlinsolver = MatrixBLS(),
                    Jᵗ = nothing,
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: PeriodicOrbitFunctionalSh}
    bifpt = br.specialpoint[ind_bif]
    biftype = br.specialpoint[ind_bif].type
    @assert biftype == :pd "Please open an issue on BifurcationKit website"

    par_pd = setparam(br, bifpt.param)
    pdpointguess = pd_point(br, ind_bif)
    # wrap of shooting functional
    pbwrap = deepcopy(getprob(br))
    # we put the problem back to the state it was
    restore_problem!(pbwrap, bifpt.x, par_pd)
    x0 = saved_solution(pdpointguess.u)

    # get the PD eigenvectors
    (;ζ, ζad) = _init_pd_vectors_minaug(pbwrap, x0, par_pd, bdlinsolver, bdlinsolver, nothing, nothing, norm)

    # perform continuation
    continuation_pd(pbwrap, alg,
        pdpointguess, par_pd,
        getlens(br), lens2,
        ζ, ζad,
        options_cont;
        kwargs...,
        bdlinsolver,
        kind = PDPeriodicOrbitCont(),
        )
end

"""
$(SIGNATURES)

Continuation of curve of Neimark-Sacker bifurcations of periodic orbits computed using shooting method.

# Arguments
- `br` branch of periodic orbits computed with a [`Shooting`](@ref)
- `ind_bif` index of the NS point
- `lens2::AllOpticTypes` second parameter axis
- `options_cont` parameters to be used by a regular [`continuation`](@ref)
"""
function continuation_sh_ns(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::AllOpticTypes,
                    options_cont::ContinuationPar = br.contparams ;
                    alg = getalg(br),
                    start_with_eigen = Val(false), # used to filter the option
                    bdlinsolver = MatrixBLS(),
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: PeriodicOrbitFunctionalSh}
    bifpt = br.specialpoint[ind_bif]
    biftype = bifpt.type
    @assert biftype == :ns "We continue only NS points of Periodic orbits for now"

    par_ns = setparam(br, bifpt.param)
    nspointguess = ns_point(br, ind_bif)
    # wrap of collocation functional
    pbwrap = deepcopy(getprob(br))
    # we put the problem back to the state it was
    restore_problem!(pbwrap, bifpt.x, par_ns)
    x0 = saved_solution(nspointguess.u)

    # get the NS eigenvectors
    λₙₛ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
    (;ζ, ζad) = _init_ns_vectors_minaug(pbwrap, x0, par_ns, imag(λₙₛ), bdlinsolver, bdlinsolver, nothing, nothing, norm)

    # perform continuation
    continuation_ns(pbwrap, alg,
        nspointguess, par_ns,
        getlens(br), lens2,
        ζ, ζad,
        options_cont;
        kwargs...,
        bdlinsolver,
        kind = NSPeriodicOrbitCont(),
        )
end
