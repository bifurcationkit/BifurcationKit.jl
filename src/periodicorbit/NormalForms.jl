"""
$(SIGNATURES)

Compute the normal form of periodic orbits. We detail the additional keyword arguments specific to periodic orbits

# Optional arguments
- `prm = true` compute the normal form using Poincaré return map (PRM). If false, use the Iooss normal form. 
"""
function get_normal_form(prob::AbstractBifurcationProblem,
            br::ContResult{ <: PeriodicOrbitCont}, id_bif::Int ;
            nev = length(eigenvalsfrombif(br, id_bif)),
            verbose = false,
            ζs = nothing,
            lens = getlens(br),
            Teigvec = _getvectortype(br),
            scaleζ = norm,
            prm = true,
            autodiff = false,
            detailed = true, # to get detailed normal form
            δ = getdelta(prob),
            )
    bifpt = br.specialpoint[id_bif]

    @assert !(bifpt.type == :endpoint) "Don't select an end point!"

    # parameters for normal form
    kwargs_nf = (;nev, verbose, lens, Teigvec, scaleζ)

    if bifpt.type == :pd
        return period_doubling_normal_form(prob, br, id_bif; prm, detailed, δ, kwargs_nf...)
    elseif bifpt.type == :bp
        return branch_normal_form(prob, br, id_bif; prm, detailed, δ, autodiff, kwargs_nf...)
    elseif bifpt.type == :ns
        return neimark_sacker_normal_form(prob, br, id_bif; δ, detailed, prm, kwargs_nf...)
    end

    throw("Normal form for $(bifpt.type) not yet implemented.")
end
####################################################################################################
"""
[WIP] Note that the computation of this normal form is not implemented yet.
"""
function branch_normal_form(pbwrap,
                            br,
                            ind_bif::Int;
                            nev = length(eigenvalsfrombif(br, ind_bif)),
                            verbose = false,
                            lens = getlens(br),
                            Teigvec = vectortype(br),
                            scaleζ = norminf,
                            kwargs_nf...)
    pb = pbwrap.prob
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    par = setparam(br, bifpt.param)
    period = getperiod(pb, bifpt.x, par)

    # let us compute the kernel
    λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    verbose && print("├─ computing nullspace of Periodic orbit problem...")
    ζ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
    # we normalize it by the sup norm because it could be too small/big in L2 norm
    ζ ./= scaleζ(ζ)
    verbose && println("Done!")

    # compute the full eigenvector
    floquetsolver = br.contparams.newton_options.eigsolver
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), real.(ζ))
    ζs = reduce(vcat, ζ_a)

    # normal form for Poincaré map
    nf = BranchPoint(bifpt.x, bifpt.τ, bifpt.param, par, getlens(br), nothing, nothing, nothing, :none)

    return BranchPointPO(bifpt.x, period, real.(ζs), nothing, nf, pb, true)
end
####################################################################################################
function period_doubling_normal_form(pbwrap,
                                br,
                                ind_bif::Int;
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                lens = getlens(br),
                                Teigvec = vectortype(br),
                                scaleζ = norminf,
                                kwargs_nf...)
    pb = pbwrap.prob
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    pars = setparam(br, bifpt.param)
    period = getperiod(pb, bifpt.x, pars)

    # let us compute the kernel
    λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    ζ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
    # we normalize it by the sup norm because it could be too small/big in L2 norm
    ζ ./= scaleζ(ζ)
    verbose && println("Done!")

    # compute the full eigenvector
    floquetsolver = br.contparams.newton_options.eigsolver
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), real.(ζ))
    ζs = reduce(vcat, ζ_a)

    # normal form for Poincaré map
    nf = PeriodDoubling(nothing, nothing, bifpt.param, pars, getlens(br), nothing, nothing, nothing, :none)
    PeriodDoublingPO(bifpt.x, period, real.(ζs), nothing, nf, pb, true)
end

function period_doubling_normal_form(pbwrap::WrapPOSh,
                                br,
                                ind_bif::Int;
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                lens = getlens(br),
                                Teigvec = vectortype(br),
                                detailed = true,
                                scaleζ = norminf,
                                kwargs_nf...)
    verbose && println("━"^53*"\n──▶ Period-doubling normal form computation")
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    pars = setparam(br, bifpt.param)

    # let us compute the kernel
    λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    verbose && print("├─ computing nullspace of Periodic orbit problem...")
    ζ₋₁ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev) .|> real
    # we normalize it by the sup norm because it could be too small/big in L2 norm
    ζ₋₁ ./= scaleζ(ζ₋₁)
    verbose && println("Done!")

    # compute the full eigenvector
    floquetsolver = br.contparams.newton_options.eigsolver
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), real.(ζ₋₁))
    ζs = reduce(vcat, ζ_a)

    pd0 = PeriodDoubling(bifpt.x, nothing, bifpt.param, pars, getlens(br), nothing, nothing, nothing, :none)
    if ~detailed
        period = getperiod(pbwrap.prob, pd0.x0, pd0.params)
        return PeriodDoublingPO(pd0.x0, period, real.(ζs), nothing, pd0, pbwrap.prob, true)
    end

    # newton parameter
    optn = br.contparams.newton_options
    period_doubling_normal_form(pbwrap, pd0, (ζ₋₁, ζs), optn; verbose = verbose, nev = nev, kwargs_nf...)
end

function period_doubling_normal_form(pbwrap::WrapPOSh{ <: PoincareShootingProblem },
                                pd0::PeriodDoubling,
                                (ζ₋₁, ζs),
                                optn::NewtonPar;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                kwargs_nf...)
    psh = pbwrap.prob
    period = getperiod(psh, pd0.x0, pd0.params)
    PeriodDoublingPO(pd0.x0, period, real.(ζs), nothing, pd0, psh, true)
end

function period_doubling_normal_form(pbwrap::WrapPOSh{ <: ShootingProblem },
                                pd0::PeriodDoubling,
                                (ζ₋₁, ζs),
                                optn::NewtonPar;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                δ = 1e-9,
                                kwargs_nf...)
    sh = pbwrap.prob
    pars = pd0.params
    period = getperiod(sh, pd0.x0, pars)
    # compute the Poincaré return map, the section is on the first time slice
    Π = PoincareMap(pbwrap, pd0.x0, pars, optn)
    # Π = PoincareCallback(pbwrap, pd0.x0, pars; radius = 0.1)
    xₛ = get_time_slices(sh, Π.po)[:, 1]

    # If M is the monodromy matrix and E = x - <x,e>e with e the eigen
    # vector of M for the eigenvalue 1, then, we find that
    # eigenvector(P) = E ∘ eigenvector(M)
    # E(x) = x .- dot(ζ₁, x) .* ζ₁

    _nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
    _nrm > 1e-10 && @warn "Residual seems large = $_nrm"

    # dΠ = finite_differences(x -> Π(x, pars).u, xₛ; δ)
    dΠ = jacobian(Π, xₛ, pars)
    J = jacobian(pbwrap, pd0.x0, pars)
    M = MonodromyQaD(J)

    Fₘ = eigen(M)
    F = eigen(dΠ)

    ind₋₁ = argmin(abs.(F.values .+ 1))
    ev₋₁ = F.vectors[:, ind₋₁]
    Fp = eigen(dΠ')
    ind₋₁ = argmin(abs.(Fp.values .+ 1))
    ev₋₁p = Fp.vectors[:, ind₋₁]
    ####

    @debug "" Fₘ.values F.values Fp.values

    # dΠ * ζ₋₁ + ζ₋₁ |> display # not good, need projector E
    # dΠ * ev₋₁ + ev₋₁ |> display
    # dΠ' * ev₋₁p + ev₋₁p |> display
    # e = Fₘ.vectors[:,end]; e ./= norm(e)

    # normalize eigenvectors
    ev₋₁ ./= sqrt(dot(ev₋₁, ev₋₁))
    ev₋₁p ./= dot(ev₋₁, ev₋₁p)

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            J = (x,p) -> jacobian(Π, x, p),
            d2F = (x,p,h1,h2)    -> d2F(Π,x,p,h1,h2).u,
            d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
            )

    pd1 = PeriodDoubling(xₛ, nothing, pd0.p, pars, lens, ev₋₁, ev₋₁p, nothing, :none)
    # normal form computation
    pd = period_doubling_normal_form(probΠ, pd1, optn.linsolver; verbose)
    return PeriodDoublingPO(pd0.x0, period, real.(ζs), nothing, pd, sh, true)
end

function period_doubling_normal_form(pbwrap::WrapPOColl,
                                br,
                                ind_bif::Int;
                                verbose = false,
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                prm = true,
                                detailed = true,
                                kwargs_nf...)
    # first, get the bifurcation point parameters
    verbose && println("━"^53*"\n──▶ Period-Doubling normal form computation")
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    par = setparam(br, bifpt.param)
    period = getperiod(pbwrap.prob, bifpt.x, par)

    if bifpt.x isa POSolutionAndState
        # the solution is mesh adapted, we need to restore the mesh.
        pbwrap = deepcopy(pbwrap)
        update_mesh!(pbwrap.prob, bifpt.x._mesh )
        bifpt = @set bifpt.x = bifpt.x.sol
    end
    pd0 = PeriodDoubling(bifpt.x, nothing, bifpt.param, par, getlens(br), nothing, nothing, nothing, :none)
    if ~prm || ~detailed
        # method based on Iooss method
        return period_doubling_normal_form(pbwrap, pd0; detailed, verbose, nev, kwargs_nf...)
    end
    # method based on Poincaré Return Map (PRM)
    # newton parameter
    optn = br.contparams.newton_options
    return period_doubling_normal_form_prm(pbwrap, pd0, optn; verbose, nev, kwargs_nf...)
end

function period_doubling_normal_form(pbwrap::WrapPOColl,
                                pd::PeriodDoubling;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                detailed = true,
                                kwargs_nf...)
    # based on the article
    # Kuznetsov, Yu. A., W. Govaerts, E. J. Doedel, and A. Dhooge. “Numerical Periodic Normalization for Codim 1 Bifurcations of Limit Cycles.” SIAM Journal on Numerical Analysis https://doi.org/10.1137/040611306.
    # on page 1243
    # there are a lot of mistakes in the above paper, it seems better to look at https://webspace.science.uu.nl/~kouzn101/NBA/LC2.pdf
    # see also Witte, V. De, F. Della Rossa, W. Govaerts, and Yu. A. Kuznetsov. “Numerical Periodic Normalization for Codim 2 Bifurcations of Limit Cycles” SIAM Journal on Applied Dynamical Systems. https://doi.org/10.1137/120874904.

    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    par = pd.params
    p₀ = _get(par, lens)
    T = getperiod(coll, pd.x0, par)
    lens = getlens(coll)
    δ = getdelta(coll)

    # identity matrix for collocation problem
    Icoll = I(coll, _getsolution(pd.x0), par)

    F(u, p) = residual(coll.prob_vf, u, p)
    # dₚF(u, p) = ForwardDiff.derivative(z -> residual(coll.prob_vf, u, _set_param(p, lens, z)), get(par, lens))
    dₚF(u, p) = (residual(coll.prob_vf, u, _set_param(p, lens, p₀ + δ)) .- 
                 residual(coll.prob_vf, u, _set_param(p, lens, p₀ - δ))) ./ (2δ)
    A(u, p, du) = apply(jacobian(coll.prob_vf, u, p), du)
    F11(u, p, du) = (A(u, _set_param(p, lens, p₀ + δ), du) .- 
                     A(u, _set_param(p, lens, p₀ - δ), du)) ./ (2δ)
    B(u, p, du1, du2)      = d2F(coll.prob_vf, u, p, du1, du2)
    C(u, p, du1, du2, du3) = d3F(coll.prob_vf, u, p, du1, du2, du3)

    _rand(n, r = 2) = r .* (rand(n) .- 1/2)        # centered uniform random variables
    local ∫(u,v) = BifurcationKit.∫(coll, u, v, 1) # define integral with coll parameters

    # we first compute the floquet eigenvector for μ = -1
    # we use an extended linear system for this
    #########
    # compute v1
    jac = jacobian(pbwrap, _getsolution(pd.x0), par)
    J = copy(_get_matrix(jac)) # we put copy to not alias FloquetWrapper.jacpb
    nj = size(J, 1)
    J[end, :] .= _rand(nj)
    J[:, end] .= _rand(nj)
    J[end, end] = 0
    # enforce PD boundary condition
    J[end-N:end-1, 1:N] .= I(N)
    J[end-N:end-1, end-N:end-1] .= I(N)

    rhs = zeros(nj); rhs[end] = 1;
    k = J  \ rhs; k = k[1:end-1]; k ./= norm(k) #≈ ker(J)
    l = J' \ rhs; l = l[1:end-1]; l ./= norm(l)

    # update the borders to have less singular matrix J
    J[end, 1:end-1] .= k
    J[1:end-1, end] .= l

    # right Floquet eigenvectors
    vr = J \ rhs

    v₁  = @view vr[1:end-1]
    v₁ ./= sqrt(∫(vr, vr)) # this modifies v₁ by reference

    #########
    # compute v1★
    J★ = analytical_jacobian(coll, _getsolution(pd.x0), par; _transpose = true, ρF = -1)
    J★[end, :] .= _rand(nj)
    J★[:, end] .= _rand(nj)
    J★[end, end] = 0
    # enforce PD boundary condition
    J★[end-N:end-1, 1:N] .= I(N)
    J★[end-N:end-1, end-N:end-1] .= I(N)

    rhs = zeros(nj); rhs[end] = 1;
    k = J★  \ rhs; k = k[1:end-1]; k ./= norm(k) # ≈ ker(J)
    l = J★' \ rhs; l = l[1:end-1]; l ./= norm(l)

    vl = J★ \ rhs
    v₁★ = @view vl[1:end-1]
    v₁★ ./= 2∫(vl, vr)

    # convention notation. We use the ₛ to indicates time slices which
    # are of size (N, Ntxt⋅m + 1)
    v₁ₛ  = get_time_slices(coll, vcat(v₁ ,1))
    v₁★ₛ = get_time_slices(coll, vcat(v₁★,1))

    @assert ∫(v₁★ₛ, v₁ₛ) ≈ 1/2
    @assert ∫(v₁ₛ, v₁ₛ) ≈ 1

    # if we just want the eigenvectors
    if ~detailed
        return PeriodDoublingPO(pd.x0, T, v₁, v₁★, (@set pd.nf = (a = 0, b3 = 0)), coll, false)
    end

    u₀ₛ = get_time_slices(coll, pd.x0) # periodic solution at bifurcation
    Fu₀ₛ = copy(u₀ₛ)
    Aₛ   = copy(u₀ₛ)
    Bₛ   = copy(u₀ₛ)
    Cₛ   = copy(u₀ₛ)
    for i = 1:size(u₀ₛ, 2)
      Fu₀ₛ[:,i] .= F(u₀ₛ[:,i], par)
        Aₛ[:,i] .= A(u₀ₛ[:,i], par, v₁ₛ[:,i])
        Bₛ[:,i] .= B(u₀ₛ[:,i], par, v₁ₛ[:,i], v₁ₛ[:,i])
        Cₛ[:,i] .= C(u₀ₛ[:,i], par, v₁ₛ[:,i], v₁ₛ[:,i], v₁ₛ[:,i])
    end

    # computation of ψ★, recall the BC ψ★(0) = ψ★(1)
    # for this, we generate the linear problem analytically
    # note that we could obtain the same by modifying inplace 
    # the previous linear problem J
    Jψ = analytical_jacobian(coll, _getsolution(pd.x0), par; _transpose = true, ρF = -1)
    Jψ[end-N:end-1, 1:N] .= -I(N)
    Jψ[end-N:end-1, end-N:end-1] .= I(N)
    # build the extended linear problem
    Jψ[end, :] .= _rand(nj)
    Jψ[:, end] .= _rand(nj)
    Jψ[end, end] = 0

    # update the borders to have less singular matrix Jψ
    k = Jψ  \ rhs; k = k[1:end-1]; k ./= norm(k)
    l = Jψ' \ rhs; l = l[1:end-1]; l ./= norm(l)
    Jψ[end, 1:end-1] .= k
    Jψ[1:end-1, end] .= l

    ψ₁★ = Jψ \ rhs
    ψ₁★ₛ = get_time_slices(coll, ψ₁★)
    ψ₁★ ./= 2∫( ψ₁★ₛ, Fu₀ₛ)
    @assert ∫( ψ₁★ₛ, Fu₀ₛ) ≈ 1/2

    # computation of a₁
    a₁ = ∫(ψ₁★ₛ, Bₛ)
            # _plot(vcat(vec(ψ₁★ₛ),1), label = "ψ1star")
            # _plot(vcat(vec(@. Bₛ ),1), label = "Bₛ")
            # return a₁

    # computation of h₂
    rhsₛ = @. Bₛ - 2a₁ * Fu₀ₛ
    @assert abs(∫(rhsₛ, ψ₁★ₛ)) < 1e-12
    rhs = vcat(vec(rhsₛ), 0) # it needs to end with zero for the integral condition
    border_ψ₁ = ForwardDiff.gradient(x -> ∫( reshape(x, size(ψ₁★ₛ)), ψ₁★ₛ),
                                     zeros(length(ψ₁★ₛ))
                                    )
                            # _plot(vcat(vec(rhsₛ),1))
    # we could perhaps save the re-computation of J here and use the previous J
    jac = jacobian(pbwrap, _getsolution(pd.x0), par)
    J = copy(_get_matrix(jac))
    J[end-N:end-1, 1:N] .= -I(N)
    J[end-N:end-1, end-N:end-1] .= I(N)
    # add borders
    J[end, 1:end-1] .= border_ψ₁ # integral condition
    J[:, end] .= ψ₁★
    J[end, end] = 0
    h₂ = J \ (Icoll * rhs)
    # h₂ ./= 2Ntst # this seems necessary to have something comparable to ApproxFun
                # h₂ = Icoll * h₂;@set! h₂[end]=0
    h₂ₛ = get_time_slices(coll, h₂)
                # a cause de Icoll
                # h₂ₛ[:, end] .= h₂ₛ[:,1]
    if abs(∫( ψ₁★ₛ, h₂ₛ)) > 1e-10
        @warn "The integral ∫(ψ₁★ₛ, h₂ₛ) should be zero. We found $(∫(  ψ₁★ₛ, h₂ₛ ))"
    end
    if abs(h₂[end]) > 1e-10
        @warn "The value h₂[end] should be zero. We found $(h₂[end])"
    end

    # computation of c
    # we need B(t, v₁(t), h₂(t))
    for i=1:size(Bₛ, 2)
        Bₛ[:,i] .= B(u₀ₛ[:,i], par, v₁ₛ[:,i], h₂ₛ[:,i])
    end
                # _plot(vcat(vec( Bₛ ),1), label = "Bₛ for h2")
                # _plot(vcat(vec(@. Bₛ * v₁★ₛ ),1), label = "Bₛ*v1star")

    c = 1/(3T) * ∫( v₁★ₛ, Cₛ ) + 
                 ∫( v₁★ₛ, Bₛ ) -
         2a₁/T * ∫( v₁★ₛ, Aₛ )

                    @debug "" ∫( v₁★ₛ, Bₛ ) 2a₁/T * ∫( v₁★ₛ, Aₛ )

    # computation of a₀₁
    ∂Fu₀ₛ = copy(u₀ₛ)
    for i = 1:size(u₀ₛ, 2)
        ∂Fu₀ₛ[:,i] .= dₚF(u₀ₛ[:,i], par)
    end
    a₀₁ = 2∫(ψ₁★ₛ, ∂Fu₀ₛ)

    # computation of h₀₁
    #                     ∂ₜh₀₁ - A(t)h₀₁ = F₀₁(t) - a₀₁⋅∂u₀
    rhsₛ = copy(u₀ₛ)
    for i = 1:size(u₀ₛ, 2)
        rhsₛ[:,i] .= ∂Fu₀ₛ[:,i] .- a₀₁ .* Fu₀ₛ[:,i]
    end
    rhs = vcat(vec(rhsₛ), 0) # it needs to end with zero for the integral condition
    jac = jacobian(pbwrap, _getsolution(pd.x0), par)
    J = copy(_get_matrix(jac))
    J[end-N:end-1, 1:N] .= -I(N)
    J[end-N:end-1, end-N:end-1] .= I(N)
    # add borders
    J[end, 1:end-1] .= border_ψ₁ # integral condition
    J[:, end] .= ψ₁★
    J[end, end] = 0
    h₀₁ = J \ (Icoll * rhs)
    h₀₁ₛ = get_time_slices(coll, h₀₁)

    # computation of c₁₁
    #                   < w★, -B(t,h01,w) - F11*w + c11*w + a01*wdot > = 0
    # hence:
    #                   c11 = < w★, B(t,h01,w) + F11*w + c11*w - a01*wdot >
    for i = 1:size(u₀ₛ, 2)
        rhsₛ[:,i] .= B(u₀ₛ[:,i], par, v₁★ₛ[:,i], h₀₁ₛ[:,i]) .+ F11(u₀ₛ[:,i], par, v₁★ₛ[:,i])
    end

    c₁₁ = ∫(v₁★ₛ, rhsₛ) - a₀₁ * ∫(v₁★ₛ, Aₛ)
    c₁₁ *= 2

    # we want the parameter a, not the rescaled a₁
    nf = (a = a₁/T, b3 = c, h₂ₛ, ψ₁★ₛ, v₁ₛ, a₀₁, c₁₁) # keep b3 for PD-codim 2
    newpd = @set pd.nf = nf
    @debug "[PD-NF-Iooss]" a₁ c
    if real(c) < 0
        @set! newpd.type = :SuperCritical
    else
        @set! newpd.type = :SubCritical
    end
    return PeriodDoublingPO(pd.x0, T, v₁, v₁★, newpd, coll, false)
end

function period_doubling_normal_form_prm(pbwrap::WrapPOColl,
                                    pd0::PeriodDoubling,
                                    optn::NewtonPar;
                                    nev = 3,
                                    δ = 1e-7,
                                    verbose = false,
                                    lens = getlens(pbwrap),
                                    kwargs_nf...)
    @debug "PD normal form collocation, method PRM"
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    pars = pd0.params
    T = getperiod(coll, pd0.x0, pars)

    Π = PoincareMap(pbwrap, pd0.x0, pars, optn)
    xₛ = pd0.x0[1:N]
    dΠ = finite_differences(x -> Π(x,pars).u, xₛ)
    F = eigen(dΠ)

    ind₋₁ = argmin(abs.(F.values .+ 1))
    ev₋₁ = F.vectors[:, ind₋₁]
    Fp = eigen(dΠ')
    ind₋₁ = argmin(abs.(Fp.values .+ 1))
    ev₋₁p = Fp.vectors[:, ind₋₁]

    # normalize eigenvectors
    ev₋₁ ./= sqrt(dot(ev₋₁, ev₋₁))
    ev₋₁p ./= dot(ev₋₁, ev₋₁p)

    δ2 = √δ
    δ3 = δ^(1/3)
    d1Π(x,p,dx) = (Π(x .+ δ .* dx, p).u .- Π(x .- δ .* dx, p).u) ./ (2δ)
    d2Π(x,p,dx1,dx2) = (d1Π(x .+ δ2 .* dx2, p, dx1) .- d1Π(x .- δ2 .* dx2, p, dx1)) ./ (2δ2)
    d3Π(x,p,dx1,dx2,dx3) = (d2Π(x .+ δ3 .* dx3, p, dx1, dx2) .- d2Π(x .- δ3 .* dx3, p, dx1, dx2)) ./ (2δ3)

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            J = (x,p) -> finite_differences(z -> Π(z,p).u, x),
            # d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
            # d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
            d2F = d2Π,
            d3F = d3Π,
            )

    pd1 = PeriodDoubling(xₛ, nothing, pd0.p, pars, lens, ev₋₁, ev₋₁p, nothing, :none)
    pd = period_doubling_normal_form(probΠ, pd1, optn.linsolver; verbose)

    # we get the floquet eigenvectors for μ = -1
    jac = jacobian(pbwrap, pd0.x0, pars)
    # remove borders
    J = copy(_get_matrix(jac))
    nj = size(J, 1)
    J[end, :] .= rand(nj)
    J[:, end] .= rand(nj)
    # enforce PD boundary condition
    J[end-N:end-1, 1:N] .= I(N)
    rhs = zeros(nj); rhs[end] = 1
    q = J  \ rhs; q = q[1:end-1]; q ./= norm(q)
    p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

    J[end, 1:end-1] .= q
    J[1:end-1, end] .= p

    vl = J' \ rhs
    vr = J  \ rhs

    v₁  = @view vr[1:end-1]
    v₁★ = @view vl[1:end-1]

    return PeriodDoublingPO(pd0.x0, pd0.x0[end], v₁, v₁★, pd, coll, true)
end
####################################################################################################
function neimark_sacker_normal_form(pbwrap::WrapPOColl,
                                br::AbstractBranchResult,
                                ind_bif::Int;
                                verbose = false,
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                prm = true,
                                detailed = true,
                                kwargs_nf...)
    verbose && println("━"^53*"\n──▶ Neimark-Sacker normal form computation")
    # get the bifurcation point parameters
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    par = setparam(br, bifpt.param)
    period = getperiod(coll, bifpt.x, par)

    # get the eigenvalue
    eigRes = br.eig
    λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ωₙₛ = abs(imag(λₙₛ))

    if bifpt.x isa POSolutionAndState
        # the solution is mesh adapted, we need to restore the mesh.
        pbwrap = deepcopy(pbwrap)
        update_mesh!(pbwrap.prob, bifpt.x._mesh )
        bifpt = @set bifpt.x = bifpt.x.sol
    end
    ns0 = NeimarkSacker(bifpt.x, nothing, bifpt.param, ωₙₛ, par, getlens(br), nothing, nothing, nothing, :none)

    if ~detailed
        return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap, true)
    end

    if prm # method based on Poincare Return Map (PRM)
        # newton parameter
        optn = br.contparams.newton_options
        return neimark_sacker_normal_form_prm(pbwrap, ns0, optn; verbose = verbose, nev = nev, kwargs_nf...)
    end
    # method based on Iooss method
    # nf = PeriodDoubling(bifpt.x, period, bifpt.param, par, getlens(br), nothing, nothing, nothing, :none)
    neimark_sacker_normal_form(pbwrap, ns0; verbose, nev, kwargs_nf...)
end

function neimark_sacker_normal_form(pbwrap,
                            br,
                            ind_bif::Int;
                            nev = length(eigenvalsfrombif(br, ind_bif)),
                            verbose = false,
                            lens = getlens(br),
                            Teigvec = vectortype(br),
                            kwargs_nf...)
    pb = pbwrap.prob
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    pars = setparam(br, bifpt.param)
    period = getperiod(pb, bifpt.x, pars)

    # get the eigenvalue
    eigRes = br.eig
    λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ωₙₛ = imag(λₙₛ)

    ns0 =  NeimarkSacker(bifpt.x, bifpt.param, ωₙₛ, pars, getlens(br), nothing, nothing, nothing, :none)
    return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap, true)
end

function neimark_sacker_normal_form_prm(pbwrap::WrapPOColl,
                                    ns0::NeimarkSacker,
                                    optn::NewtonPar;
                                    nev = 3,
                                    δ = 1e-7,
                                    verbose = false,
                                    lens = getlens(pbwrap),
                                    kwargs_nf...)
    @debug "method PRM"
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    pars = ns0.params
    T = getperiod(coll, ns0.x0, pars)

    Π = PoincareMap(pbwrap, ns0.x0, pars, optn)
    xₛ = ns0.x0[1:N]
    dΠ = finite_differences(x -> Π(x,pars).u, xₛ)
    F = eigen(dΠ)

    _nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
    _nrm > 1e-12 && @warn  "$_nrm"

    ####
    ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
    ev = F.vectors[:, ind]
    Fp = eigen(dΠ')
    indp = argmin(abs.(log.(complex.(Fp.values)) .+ Complex(0, ns0.ω )))
    evp = Fp.vectors[:, indp]

    # normalize eigenvectors
    ev ./= sqrt(dot(ev, ev))
    evp ./= dot(ev, evp)

    δ2 = √δ
    δ3 = δ^(1/3)
    d1Π(x,p,dx) = ((Π(x .+ δ .* dx, p).u .- Π(x .- δ .* dx, p).u) ./ (2δ))
    d2Π(x,p,dx1,dx2) = ((d1Π(x .+ δ2 .* dx2, p, dx1) .- d1Π(x .- δ2 .* dx2, p, dx1)) ./ (2δ2))
    d3Π(x,p,dx1,dx2,dx3) = ((d2Π(x .+ δ3 .* dx3, p, dx1, dx2) .- d2Π(x .- δ3 .* dx3, p, dx1, dx2)) ./ (2δ3))

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            J = (x,p) -> finite_differences(z -> Π(z,p).u, x),
            d2F = d2Π,
            d3F = d3Π,
            )

    ns1 = NeimarkSacker(xₛ, nothing, ns0.p, ns0.ω, pars, lens, ev, evp, nothing, :none)
    ns = neimark_sacker_normal_form(probΠ, ns1, optn.linsolver; verbose)
    return NeimarkSackerPO(ns0.x0, T, ns0.p, ns0.ω, ev, nothing, ns, coll, true)
end

function neimark_sacker_normal_form(pbwrap::WrapPOColl,
                                        ns::NeimarkSacker;
                                        nev = 3,
                                        verbose = false,
                                        lens = getlens(pbwrap),
                                        _NRMDEBUG = false, # normalise to compare to ApproxFun
                                        kwargs_nf...)
    @debug "method IOOSS, NRM = $_NRMDEBUG"

    # based on the article
    # Kuznetsov, Yu. A., W. Govaerts, E. J. Doedel, and A. Dhooge. “Numerical Periodic Normalization for Codim 1 Bifurcations of Limit Cycles.” SIAM Journal on Numerical Analysis 43, no. 4 (January 2005): 1407–35. https://doi.org/10.1137/040611306.
    # there are a lot of mistakes in the above paper, it seems better to look at https://webspace.science.uu.nl/~kouzn101/NBA/LC2.pdf
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    par = ns.params
    T = getperiod(coll, ns.x0, par)
    # identity matrix for collocation problem
    Icoll = I(coll, ns.x0, par)

    F(u, p) = residual(coll.prob_vf, u, p)
    A(u, p, du) = apply(jacobian(coll.prob_vf, u, p), du)
    B(u, p, du1, du2)      = BilinearMap( (dx1, dx2)      -> d2F(coll.prob_vf, u, p, dx1, dx2))(du1, du2)
    C(u, p, du1, du2, du3) = TrilinearMap((dx1, dx2, dx3) -> d3F(coll.prob_vf, u, p, dx1, dx2, dx3))(du1, du2, du3)

    _plot(x; k...) = (_sol = get_periodic_orbit(coll, x, 1);display(plot(_sol.t, _sol.u'; k...)))
    _rand(n, r = 2) = r .* (rand(n) .- 1/2)        # centered uniform random variables
    local ∫(u,v) = BifurcationKit.∫(coll, u, v, 1) # define integral with coll parameters

    #########
    # compute v1
    # we first compute the NS floquet eigenvector
    # we use an extended linear system for this
     # J = D  -  T*A(t) + iθ/T
    θ = abs(ns.ω)
    J = analytical_jacobian(coll, ns.x0, par; ρI = Complex(0,-θ/T), 𝒯 = ComplexF64)

    nj = size(J, 1)
    J[end, :] .= _rand(nj); J[:, end] .= _rand(nj)
    J[end, end] = 0

    rhs = zeros(nj); rhs[end] = 1;
    k = J  \ rhs; k = k[1:end-1]; k ./= norm(k) # ≈ ker(J)
    l = J' \ rhs; l = l[1:end-1]; l ./= norm(l)

    # update the borders to have less singular matrix J
    J[end, 1:end-1] .= k
    J[1:end-1, end] .= l

    # Floquet eigenvectors
    vr = J  \ rhs
    v₁  = @view vr[1:end-1]
    v₁ ./= sqrt(∫(vr, vr))
    v₁ₛ = get_time_slices(coll, vcat(v₁,1))

                if _NRMDEBUG;v₁ₛ .*= (0.4621019901257435 - 0.2724360760150998im)/v₁ₛ[1,1];end
                # re-scale the eigenvector
                v₁ₛ ./= sqrt(∫(v₁ₛ, v₁ₛ))
                v₁ = vec(v₁ₛ)

    @assert ∫(v₁ₛ, v₁ₛ) ≈ 1

    #########
    # compute ϕ1star
    # Jϕ = D  +  T*At(t)
    Jϕ = analytical_jacobian(coll, ns.x0, par; _transpose = true, ρF = -1)
    Jϕ[end-N:end-1, 1:N] .= -I(N)
    Jϕ[end-N:end-1, end-N:end-1] .= I(N)
    # build the extended linear problem
    Jϕ[end, :] .= _rand(nj)
    Jϕ[:, end] .= _rand(nj)
    Jϕ[end, end] = 0

    # update the borders to have less singular matrix Jψ
    k = Jϕ  \ rhs; k = k[1:end-1]; k ./= norm(k)
    l = Jϕ' \ rhs; l = l[1:end-1]; l ./= norm(l)
    Jϕ[end, 1:end-1] .= k
    Jϕ[1:end-1, end] .= l

    ϕ₁★ = Jϕ \ rhs
    ϕ₁★ₛ = get_time_slices(coll, ϕ₁★)

    u₀ₛ = get_time_slices(coll, ns.x0) # periodic solution at bifurcation
    Fu₀ₛ = copy(u₀ₛ)
    Aₛ   = copy(v₁ₛ)
    Bₛ   = copy(v₁ₛ)
    Cₛ   = copy(v₁ₛ)
    for i = 1:size(u₀ₛ, 2)
      Fu₀ₛ[:,i] .= F(u₀ₛ[:,i], par)
        Bₛ[:,i] .= B(u₀ₛ[:,i], par, v₁ₛ[:,i], conj(v₁ₛ[:,i]))
    end

    #########
    # compute a₁
    ϕ₁★ ./= ∫( ϕ₁★ₛ, Fu₀ₛ)
    @assert ∫( ϕ₁★ₛ, Fu₀ₛ) ≈ 1
    # a = ∫ < ϕ₁★, B(v1, cv1) >
    a₁ = ∫(ϕ₁★ₛ, Bₛ)

    #########
    # compute v1star
    # J = D  +  T*At(t) + iθ/T
    J = analytical_jacobian(coll, ns.x0, par; ρI = Complex(0,-θ/T), 𝒯 = ComplexF64, _transpose = true, ρF = -1)

    nj = size(J, 1)
    J[end, :] .= _rand(nj)
    J[:, end] .= _rand(nj)
    J[end, end] = 0

    rhs = zeros(nj); rhs[end] = 1;
    k = J  \ rhs; k = k[1:end-1]; k ./= norm(k) # ≈ ker(J)
    l = J' \ rhs; l = l[1:end-1]; l ./= norm(l)

    # update the borders to have less singular matrix J
    J[end, 1:end-1] .= k
    J[1:end-1, end] .= l

    # left / right Floquet eigenvectors
    vr = J  \ rhs
    v₁★  = @view vr[1:end-1]
    v₁★ₛ = get_time_slices(coll, vcat(v₁★,1))
    v₁★ₛ ./= conj(∫(v₁★ₛ, v₁ₛ))
                if _NRMDEBUG; v₁★ₛ .*= (-1.0388609772214439 - 4.170067699081798im)/v₁★ₛ[1,1];end
                # re-scale the eigenvector
    v₁★ₛ ./= conj(∫(v₁★ₛ, v₁ₛ))
    v₁★ = vec(v₁★ₛ)

                # return
    @assert ∫(v₁★ₛ, v₁ₛ) ≈ 1
    #########
    # compute h20
    # solution of (D-T A(t) + 2iθ   )h = B(v1, v1)
    # written     (D-T(A(t) - 2iθ/T))h = B
    for i = 1:size(u₀ₛ, 2)
        Bₛ[:,i] .= B(u₀ₛ[:,i], par, v₁ₛ[:,i], v₁ₛ[:,i])
    end
    rhs = vcat(vec(Bₛ), 0)
    J = analytical_jacobian(coll, ns.x0, par; ρI = Complex(0,-2θ/T), 𝒯 = ComplexF64)
    # h₂₀ = J \ (rhs)

    h₂₀= J[1:end-1,1:end-1] \ rhs[1:end-1];h₂₀ = vcat(vec(h₂₀), 0)
    # h₂₀ ./= 2Ntst # this seems necessary to have something comparable to ApproxFun
    h₂₀ = Icoll * h₂₀;@set! h₂₀[end]=0
    h₂₀ₛ = get_time_slices(coll, h₂₀)
                # a cause de Icoll
                h₂₀ₛ[:, end] .= h₂₀ₛ[:,1]

                # _plot(real(vcat(vec(h₂₀ₛ),1)),label="h20")
                # _plot(imag(vcat(vec(Bₛ),1+im)),label="Bₛ")

    #########
    # compute h11
    # solution of (D-TA(t))h = B - a₁F
    for i = 1:size(u₀ₛ, 2)
        Bₛ[:,i] .= B(u₀ₛ[:,i], par, v₁ₛ[:,i], conj(v₁ₛ[:,i]))
    end
    rhsₛ = @. Bₛ - a₁ * Fu₀ₛ
    rhs = vcat(vec(rhsₛ), 0)
    border_ϕ1 = ForwardDiff.gradient(x -> ∫( reshape(x, size(ϕ₁★ₛ)), ϕ₁★ₛ),
                                     zeros(length(ϕ₁★ₛ))
                                    )
    J = analytical_jacobian(coll, ns.x0, par;  𝒯 = ComplexF64)
    J[end-N:end-1, 1:N] .= -I(N)
    J[end-N:end-1, end-N:end-1] .= I(N)
    # add borders
    J[end, 1:end-1] .= border_ϕ1 # integral condition
    J[:, end] .= ϕ₁★
    J[end, end] = 0
    h₁₁ = J \ rhs
    h₁₁ ./= 2Ntst # this seems necessary to have something comparable to ApproxFun
    h₁₁ₛ = get_time_slices(coll, h₁₁)
                # _plot(real(vcat(vec(h₁₁ₛ),1)),label="h11")
                @debug "" abs(∫( ϕ₁★ₛ, h₁₁ₛ))
    if abs(∫( ϕ₁★ₛ, h₁₁ₛ)) > 1e-10
        @warn "The integral ∫(ϕ₁★ₛ, h₁₁ₛ) should be zero. We found $(∫( ϕ₁★ₛ, h₁₁ₛ ))"
    end
    if abs(h₁₁[end]) > 1e-10
        @warn "The value h₁₁[end] should be zero. We found $(h₁₁[end])"
    end
    #########
    # compute d
    # d = <v1★, C(v,v,v)  +  2B(h11, v)  +  B(h20, cv)  +  C(v,v,cv)>/2 + ...
    for i = 1:size(u₀ₛ, 2)
        Bₛ[:,i] .= B(u₀ₛ[:,i], par, h₁₁ₛ[:,i], v₁ₛ[:,i])
        Cₛ[:,i] .= C(u₀ₛ[:,i], par,  v₁ₛ[:,i], v₁ₛ[:,i], conj(v₁ₛ[:,i]))
    end
                # _plot(real(vcat(vec(Bₛ),1)),label="B")

    d = (1/T) * ∫( v₁★ₛ, Cₛ ) + 2 * ∫( v₁★ₛ, Bₛ )

                @debug "B(h11, v1)" d  (1/(2T)) * ∫( v₁★ₛ, Cₛ )     2*∫( v₁★ₛ, Bₛ )

    for i = 1:size(u₀ₛ, 2)
        Bₛ[:,i] .= B(u₀ₛ[:,i], par, h₂₀ₛ[:,i], conj(v₁ₛ[:,i]))
        Aₛ[:,i] .= A(u₀ₛ[:,i], par, v₁ₛ[:,i])
    end
                @debug "B(h20, v1b)" d   ∫( v₁★ₛ, Bₛ )
    d +=  ∫( v₁★ₛ, Bₛ )
    d = d/2
                @debug ""  -a₁/T * ∫( v₁★ₛ, Aₛ ) + im * θ * a₁/T^2   im * θ * a₁/T^2
    d += -a₁/T * ∫( v₁★ₛ, Aₛ ) + im * θ * a₁/T^2

    nf = (a = a₁, d, h₁₁ₛ, ϕ₁★ₛ, v₁★ₛ, h₂₀ₛ, _NRMDEBUG) # keep b3 for ns-codim 2
    ns_new = (@set ns.nf = nf)
    @set! ns_new.type = real(d) < 0 ? :SuperCritical : :SubCritical
    return NeimarkSackerPO(ns.x0, T, ns.p, θ, v₁, v₁★, ns_new, coll, false)
end

function neimark_sacker_normal_form(pbwrap::WrapPOSh{ <: ShootingProblem },
                                br::AbstractBranchResult,
                                ind_bif::Int;
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                lens = getlens(br),
                                Teigvec = vectortype(br),
                                detailed = true,
                                kwargs_nf...)

    # first, get the bifurcation point parameters
    sh = pbwrap.prob
    @assert sh isa ShootingProblem "Something is wrong. Please open an issue on the website"
    verbose && println("━"^53*"\n──▶ Neimark-Sacker normal form computation")

    # bifurcation point
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    pars = setparam(br, bifpt.param)
    period = getperiod(sh, bifpt.x, pars)

    # get the eigenvalue
    eigRes = br.eig
    λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ωₙₛ = imag(λₙₛ)

    ns0 = NeimarkSacker(bifpt.x, nothing, bifpt.param, ωₙₛ, pars, getlens(br), nothing, nothing, nothing, :none)

    if ~detailed
        return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap, true)
    end

    # newton parameter
    optn = br.contparams.newton_options
    return neimark_sacker_normal_form(pbwrap, ns0, (1, 1), optn; verbose, nev, kwargs_nf...)
end

function neimark_sacker_normal_form(pbwrap::WrapPOSh{ <: ShootingProblem },
                                ns0::NeimarkSacker,
                                (ζ₋₁, ζs),
                                optn::NewtonPar;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                kwargs_nf...)
    sh = pbwrap.prob
    pars = ns0.params
    period = getperiod(sh, ns0.x0, pars)
    # compute the Poincaré return map, the section is on the first time slice
    Π = PoincareMap(pbwrap, ns0.x0, pars, optn)
    xₛ = get_time_slices(sh, Π.po)[:, 1]

    _nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
    _nrm > 1e-12 && @warn "[NS normal form PRM], residual = $_nrm"

    dΠ = jacobian(Π, xₛ, pars)
    J = jacobian(pbwrap, ns0.x0, pars)
    M = MonodromyQaD(J)

    Fₘ = eigen(M)
    F = eigen(dΠ)

    ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
    ev = F.vectors[:, ind]
    Fp = eigen(dΠ')
    indp = argmin(abs.(log.(complex.(Fp.values)) .+ Complex(0, ns0.ω )))
    evp = Fp.vectors[:, indp]

    # normalize eigenvectors
    ev ./= sqrt(dot(ev, ev))
    evp ./= dot(evp, ev)

    # @debug "" xₛ ev evp dΠ _nrm pars F.values[ind] Fp.values[indp]
    # @debug "" F.values ns0.x0

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            J = (x,p) -> jacobian(Π, x, p),
            d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
            d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
            )

    ns1 = NeimarkSacker(xₛ, nothing, ns0.p, ns0.ω, pars, lens, ev, evp, nothing, :none)
    # normal form computation
    ns = neimark_sacker_normal_form(probΠ, ns1, DefaultLS(); verbose)

    return NeimarkSackerPO(ns0.x0, period, ns0.p, ns0.ω, real.(ζs), nothing, ns, sh, true)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitTrapProblem},
                    δp,
                    ampfactor;
                    override = false)
    pb = nf.prob

    M, N = size(pb)
    orbitguess0 = nf.po[begin:end-1]
    orbitguess0c = get_time_slices(pb, nf.po)
    ζc = reshape(nf.ζ, N, M)
    orbitguess_c = orbitguess0c .+ ampfactor .*  ζc
    orbitguess_c = hcat(orbitguess_c, orbitguess0c .- ampfactor .*  ζc)
    orbitguess = vec(orbitguess_c[:,1:2:end])
    # we append twice the period
    orbitguess = vcat(orbitguess, 2nf.T)
    return (;orbitguess, pnew = nf.nf.p + δp, prob = pb, ampfactor)
end

function predictor(nf::BranchPointPO{ <: PeriodicOrbitTrapProblem},
                    δp,
                    ampfactor;
                    override = false)
    orbitguess = copy(nf.po)
    orbitguess[begin:end-1] .+= ampfactor .* nf.ζ
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor)
end

function predictor(nf::NeimarkSackerPO,
                    δp,
                    ampfactor;
                    override = false)
    orbitguess = copy(nf.po)
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitOCollProblem }, 
                    δp, 
                    ampfactor; 
                    override = false)
    pbnew = deepcopy(nf.prob)
    N, m, Ntst = size(nf.prob)

    # we update the problem by doubling Ntst
    # we need to keep the mesh for adaptation
    old_mesh = getmesh(pbnew)
    new_mesh = vcat(old_mesh[begin:end-1] /2, old_mesh ./2 .+ 1/2)
    pbnew = set_collocation_size(pbnew, 2Ntst, m)
    update_mesh!(pbnew, new_mesh)

    orbitguess0 = _getsolution(nf.po)[begin:end-1]

    # parameter to scale time
    time_factor = 1

    if ~override
        if nf.prm == true && ~isnothing(nf.nf.nf)
            # normal form based on Poincare return map
            pred = predictor(nf.nf, δp)
            ampfactor *= pred.x1
            δp = pred.δp
        elseif nf.prm == false && get(nf.nf.nf, :c₁₁, nothing) != nothing
            # Iooss normal form
            @unpack c₁₁, b3 = nf.nf.nf
            c₃ = b3
            ∂p = c₁₁ * δp
            if c₃ * ∂p > 0
                ∂p *= -1
                δp *= -1
            end
            ξ = √(abs(∂p / c₃))
            ampfactor *= ξ
        end
    end

    orbitguess_c = orbitguess0 .+ ampfactor .* nf.ζ
    orbitguess = vcat(orbitguess_c[1:end-N], orbitguess0 .- ampfactor .* nf.ζ)

    pbnew.xπ .= orbitguess
    pbnew.ϕ .= circshift(orbitguess, length(orbitguess) ÷ 1)

    # we append the doubled period
    orbitguess = vcat(orbitguess, 2nf.T * time_factor)

    # no need to change pbnew.cache
    return (;orbitguess, pnew = nf.nf.p + δp, prob = pbnew, ampfactor, δp, time_factor)
end
####################################################################################################
"""
$(SIGNATURES)

Compute the predictor for the period bifurcation of periodic orbit.
"""
function predictor(nf::PeriodDoublingPO{ <: ShootingProblem },
                    δp,
                    ampfactor;
                    override = false)
    if ~isnothing(nf.nf.nf) && ~override
        pred = predictor(nf.nf, δp)
        ampfactor = pred.x1
        δp = pred.δp
    end

    pbnew = deepcopy(nf.prob)
    pnew = nf.nf.p + δp
    ζs = nf.ζ
    orbitguess = copy(nf.po)[begin:end-1] .+ ampfactor .* ζs
    orbitguess = vcat(orbitguess, copy(nf.po)[begin:end-1] .- ampfactor .* ζs, nf.po[end])

    @set! pbnew.M = 2nf.prob.M
    @set! pbnew.ds = _duplicate(pbnew.ds) ./ 2
    orbitguess[end] *= 2
    updatesection!(pbnew, orbitguess, setparam(pbnew, pnew))
    return (;orbitguess, pnew, prob = pbnew, ampfactor, δp)
end

"""
$(SIGNATURES)

Compute the predictor for the simple branch point of periodic orbit.
"""
function predictor(nf::BranchPointPO{ <: ShootingProblem },
                    δp,
                    ampfactor;
                    override = false)
    ζs = nf.ζ
    orbitguess = copy(nf.po)
    orbitguess[eachindex(ζs)] .+= ampfactor .* ζs
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PoincareShootingProblem }, 
                    δp, 
                    ampfactor;
                    override = false)
    pbnew = deepcopy(nf.prob)
    ζs = nf.ζ

    @set! pbnew.section = _duplicate(pbnew.section)
    @set! pbnew.M = pbnew.section.M
    orbitguess = copy(nf.po) .+ ampfactor .* ζs
    orbitguess = vcat(orbitguess, orbitguess .- ampfactor .* ζs)

    return (;orbitguess, pnew = nf.nf.p + δp, prob = pbnew, ampfactor)
end

function predictor(nf::BranchPointPO{ <: PoincareShootingProblem},
                    δp,
                    ampfactor;
                    override = false)
    ζs = nf.ζ
    orbitguess = copy(nf.po)
    orbitguess .+= ampfactor .* ζs
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor)
end
