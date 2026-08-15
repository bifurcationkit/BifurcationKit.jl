function d2F(wrapsh::PeriodicOrbitFunctionalSh, x, p, dx1, dx2)
    d2PO(z -> po_residual(get_discretization(wrapsh), z, p), x, dx1, dx2)
end

# if the jacobian is matrix based, use transpose
@inline has_adjoint(::PeriodicOrbitFunctionalSh{ <: Shooting{Tp, Tj} }) where {Tp, Tj} = ~(Tj <: AbstractJacobianMatrix)
@inline has_jvp(wrap::PeriodicOrbitFunctionalSh) = has_jvp(get_discretization(wrap))

# this function is necessary for pdtest to work in PDMinimallyAugmented problem
function jacobian_period_doubling(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{Tp, Tj} }, x, par) where {Tp, Tj}
    dx -> jacobian_pd_nf_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, 1, dx)
end

# this function is necessary for the jacobian of a PDMinimallyAugmented problem
function jacobian_adjoint_period_doubling(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par)
    dx -> jacobian_adjoint_period_doubling_matrix_free(pbwrap, x, par, dx)
end

jacobian_adjoint_period_doubling_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, dx) = jacobian_adjoint_pd_nf_matrix_free(pbwrap, x, par, 1, dx)

jacobian_adjoint_neimark_sacker_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, ω, dx) = jacobian_adjoint_pd_nf_matrix_free(pbwrap, x, par, -cis(-ω), dx)

# same as above but matrix based
function jacobian_period_doubling_with_border(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{Tp, Tj} }, x, par) where {Tp, Tj <: AbstractJacobianMatrix}
    M = get_mesh_size(get_discretization(pbwrap))
    N = div(length(x) - 1, M)
    Jac = jacobian(pbwrap, x, par)
    J = copy(Jac)
    # put the PD boundary condition
    J[end-N:end-1, 1:N] .= LA.I(N)
    return J
end

function jacobian_period_doubling(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{Tp, Tj} }, x, par) where {Tp, Tj <: AbstractJacobianMatrix}
    J = jacobian_period_doubling_with_border(pbwrap, x, par)
    return J[begin:end-1, begin:end-1]
end

# matrix free linear operator associated to the monodromy whose zeros are used to detect PD/NS points
function jacobian_pd_nf_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, α::𝒯, dx) where 𝒯
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
    dflow(_x, _dx::T, _T) where {T <: AbstractArray{<: Real}} = dflowDE(_x, _dx, _T)
    if ~isparallel(sh)
        for ii in 1:M
            ip1 = (ii == M) ? 1 : ii+1
            # call jacobian of the flow, jacobian-vector product
            tmp = dflow(xc[:, ii], dxc[:, ii], sh.ds[ii] * T)

            if ii<M
                outc[:, ii] .= @views tmp .- dxc[:, ip1]
            else
                outc[:, ii] .= @views tmp .+ α .* dxc[:, ip1]
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
function jacobian_adjoint_pd_nf_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, α::𝒯, dx) where 𝒯
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
                outc[:, ii] .= @views tmp .+ α .* dxc[:, im1]
            else
                outc[:, ii] .= @views tmp .- dxc[:, im1]
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

function jacobian_neimark_sacker(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{Tp, Tj} }, x, par, ω) where {Tp, Tj}
    dx -> jacobian_pd_nf_matrix_free(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, -cis(ω), dx)
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

function jacobian_neimark_sacker(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting{Tp, Tj} }, x, par, ω) where {Tp, Tj <: AbstractJacobianMatrix}
    J = jacobian_neimark_sacker_with_border(pbwrap, x, par, ω)
    return J[begin:end-1, begin:end-1]
end

# this function is necessary for the jacobian of a PDMinimallyAugmented problem
function jacobian_adjoint_neimark_sacker(pbwrap::PeriodicOrbitFunctionalSh{ <: Shooting }, x, par, ω)
    dx -> jacobian_adjoint_neimark_sacker_matrix_free(pbwrap, x, par, ω, dx)
end

function continuation(br::AbstractResult{Tkind, Tprob},
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

    if biftype == :bp || biftype == :fold
        return continuation_sh_fold(br, ind_bif, lens2, _options_cont; compute_eigen_elements, update_minaug_every_step, kwargs... )
    elseif biftype == :pd
        return  continuation_sh_pd(br, ind_bif, lens2, _options_cont; compute_eigen_elements, update_minaug_every_step, kwargs... )
    elseif biftype == :ns
        return  continuation_sh_ns(br, ind_bif, lens2, _options_cont; compute_eigen_elements, update_minaug_every_step, kwargs... )
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
function continuation_sh_pd(br::AbstractResult{Tkind, Tprob},
                    ind_bif::Int64,
                    lens2::AllOpticTypes,
                    options_cont::ContinuationPar = br.contparams ;
                    alg = getalg(br),
                    start_with_eigen = false,
                    Jᵗ = nothing,
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: PeriodicOrbitFunctionalSh}

        bifpt = br.specialpoint[ind_bif]
        pdpointguess = pd_point(br, ind_bif)

        # copy the problem for not mutating the one passed by the user
        pbwrap = getprob(br)

        # get the parameters
        par_pd = setparam(br, pdpointguess.p)

        # compute the full eigenvector, version with bordered problem
        ls = options_cont.newton_options.linsolver
        J = jacobian_period_doubling(pbwrap, bifpt.x, par_pd)
        rhs = zero(bifpt.x)[begin:end-1]; rhs[end] = 1
        q, = ls(J, rhs); q ./= norm(q) #≈ ker(J)
        # p, = ls(transpose(J), rhs); p ./= norm(p)
        p = copy(q)

        # perform continuation
        continuation_pd(getprob(br), alg,
            pdpointguess, setparam(br, pdpointguess.p),
            getlens(br), lens2,
            # ζs, ζs_ad,
            p, q,
            options_cont;
            kwargs...,
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
                    start_with_eigen = false,
                    bdlinsolver = MatrixBLS(),
                    kwargs...) where {Tkind <: PeriodicOrbitCont, Tprob <: PeriodicOrbitFunctionalSh}
    bifpt = br.specialpoint[ind_bif]
    biftype = bifpt.type

    @assert biftype == :ns "We continue only NS points of Periodic orbits for now"
    nspointguess = ns_point(br, ind_bif)

    # copy the problem for not mutating the one passed by the user
    pbwrap = getprob(br)
    par_ns = setparam(br, bifpt.param)

    # compute the eigenspace
    λₙₛ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
    ωₙₛ = λₙₛ / 1im

    J = jacobian_neimark_sacker(pbwrap, bifpt.x, par_ns, ωₙₛ)
    nj = length(bifpt.x) - 1
    q, = bdlinsolver(J, Complex.(rand(nj)), Complex.(randn(nj)), 0, Complex.(zeros(nj)), 1)
    q ./= norm(q)
    p = conj(q)

    # perform continuation
    continuation_ns(getprob(br), alg,
        nspointguess, setparam(br, nspointguess.p[1]),
        getlens(br), lens2,
        p, q,
        # ζs, copy(ζs_ad),
        options_cont;
        kwargs...,
        bdlinsolver,
        kind = NSPeriodicOrbitCont(),
        )
end
