"""
$(SIGNATURES)

Compute the normal form of periodic orbits. Same arguments as the function `getNormalForm` for equilibria. We detail the additional keyword arguments specific to periodic orbits

# Optional arguments
- `prm = true` compute the normal form using Poincaré return map. For collocation, there will be another way to compute the normal form in the future.
"""
function get_normal_form(prob::AbstractBifurcationProblem,
            br::ContResult{ <: PeriodicOrbitCont}, id_bif::Int ;
            nev = length(eigenvalsfrombif(br, id_bif)),
            verbose = false,
            ζs = nothing,
            lens = getlens(br),
            Teigvec = getvectortype(br),
            scaleζ = norm,
            prm = true,
            δ = 1e-8,
            detailed = true, # to get detailed normal form
            )
    bifpt = br.specialpoint[id_bif]

    @assert !(bifpt.type in (:endpoint,)) "Normal form for $(bifpt.type) not implemented"

    # parameters for normal form
    kwargs_nf = (nev = nev, verbose = verbose, lens = lens, Teigvec = Teigvec, scaleζ = scaleζ)

    if bifpt.type == :pd
        return period_doubling_normal_form(prob, br, id_bif; prm = prm, detailed = detailed, δ = δ, kwargs_nf...)
    elseif bifpt.type == :bp
        return branch_normal_form(prob, br, id_bif; kwargs_nf...)
    elseif bifpt.type == :ns
        return neimark_sacker_normal_form(prob, br, id_bif; δ = δ, detailed = detailed, prm = prm, kwargs_nf...)
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
    # TODO: user defined scaleζ
    ζ ./= norm(ζ, Inf)
    verbose && println("Done!")

    # compute the full eigenvector
    floquetsolver = br.contparams.newton_options.eigsolver
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), real.(ζ))
    ζs = reduce(vcat, ζ_a)

    # normal form for Poincaré map
    nf = BranchPoint(nothing, nothing, bifpt.param, par, getlens(br), nothing, nothing, nothing, :none)

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
                                kwargs_nf...)
    pb = pbwrap.prob
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    pars = setparam(br, bifpt.param)
    period = getperiod(pb, bifpt.x, pars)

    # let us compute the kernel
    λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    verbose && print("├─ computing nullspace of Periodic orbit problem...")
    ζ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
    # we normalize it by the sup norm because it could be too small/big in L2 norm
    # TODO: user defined scaleζ
    ζ ./= norm(ζ, Inf)
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
    # TODO: user defined scaleζ
    ζ₋₁ ./= norm(ζ₋₁, Inf)
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
    # ζ₁ = getVectorField(br.prob.prob.flow.prob)(xₛ,pars) |> normalize

    # If M is the monodromy matrix and E = x - <x,e>e with e the eigen
    # vector of M for the eigenvalue 1, then, we find that
    # eigenvector(P) = E ∘ eigenvector(M)
    # E(x) = x .- dot(ζ₁, x) .* ζ₁

    _nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
    _nrm > 1e-10 && @warn "Residual seems large = $_nrm"

    # dP = ForwardDiff.jacobian( x -> Π(x,pars).u, xₛ)
    dP = finite_differences(x -> Π(x,pars).u, xₛ; δ = δ)
    J = jacobian(pbwrap, pd0.x0, pars)
    M = MonodromyQaD(J)

    Fₘ = eigen(M)
    F = eigen(dP)

    # N = length(xₛ)
    # q = rand(N); p = rand(N)
    # rhs = vcat(zeros(N), 1)
    #
    # Pbd = zeros(N+1, N+1)
    # Pbd[1:N, 1:N] .= dP + I;
    # Pbd[end, 1:N] .= p
    # Pbd[1:N, end] .= q
    # ψ = Pbd \ rhs
    # ϕ = Pbd' \ rhs
    #
    # ev₋₁ = ψ[1:end-1]; normalize!(ev₋₁)
    # ev₋₁p = ϕ[1:end-1]; normalize!(ev₋₁p)

    ####
    ind₋₁ = argmin(abs.(F.values .+ 1))
    ev₋₁ = F.vectors[:, ind₋₁]
    Fp = eigen(dP')
    ind₋₁ = argmin(abs.(Fp.values .+ 1))
    ev₋₁p = Fp.vectors[:, ind₋₁]
    ####

    @debug "" Fₘ.values F.values Fp.values

    # @info "Essai de VP"
    # dP * ζ₋₁ + ζ₋₁ |> display # not good, need projector E
    # dP * ev₋₁ + ev₋₁ |> display
    # dP' * ev₋₁p + ev₋₁p |> display
    # e = Fₘ.vectors[:,end]; e ./= norm(e)

    # normalize eigenvectors
    ev₋₁ ./= sqrt(dot(ev₋₁, ev₋₁))
    ev₋₁p ./= dot(ev₋₁, ev₋₁p)

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            J = (x,p) -> finite_differences(z -> Π(z,p).u, x; δ = δ),
            d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
            d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
            )

    pd1 = PeriodDoubling(xₛ, nothing, pd0.p, pars, lens, ev₋₁, ev₋₁p, nothing, :none)
    # normal form computation
    pd = period_doubling_normal_form(probΠ, pd1, DefaultLS(); verbose = verbose)
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

    if bifpt.x isa NamedTuple
        # the solution is mesh adapted, we need to restore the mesh.
        pbwrap = deepcopy(pbwrap)
        updateMesh!(pbwrap.prob, bifpt.x._mesh )
        bifpt = @set bifpt.x = bifpt.x.sol
    end
    pd0 = PeriodDoubling(bifpt.x, nothing, bifpt.param, par, getlens(br), nothing, nothing, nothing, :none)
    if ~detailed || ~prm
        # method based on Iooss method
        return period_doubling_normal_form(pbwrap, pd0; detailed, verbose, nev, kwargs_nf...)
    end
    if prm # method based on Poincare Return Map (PRM)
        # newton parameter
        optn = br.contparams.newton_options
        return period_doubling_normal_form_prm(pbwrap, pd0, optn; verbose, nev, kwargs_nf...)
    end
    return nothing
end

function period_doubling_normal_form(pbwrap::WrapPOColl,
                                pd::PeriodDoubling;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                detailed = true,
                                kwargs_nf...)
    # based on the article
    # Kuznetsov, Yu. A., W. Govaerts, E. J. Doedel, and A. Dhooge. “Numerical Periodic Normalization for Codim 1 Bifurcations of Limit Cycles.” SIAM Journal on Numerical Analysis 43, no. 4 (January 2005): 1407–35. https://doi.org/10.1137/040611306.
    # on page 1243
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    par = pd.params
    T = getperiod(coll, pd.x0, par)

    F(u, p) = residual(coll.prob_vf, u, p)
    A(u, p, du) = apply(jacobian(coll.prob_vf, u, p), du)
    B(u, p, du1, du2)      = d2F(coll.prob_vf, u, p, du1, du2)
    C(u, p, du1, du2, du3) = d3F(coll.prob_vf, u, p, du1, du2, du3)

    _rand(n, r = 2) = r .* (rand(n) .- 1/2) # centered uniform random variables
    local ∫(u,v) = BifurcationKit.∫(coll, u, v, 1) # define integral with coll parameters

    # we first compute the PD floquet eigenvector (for μ = -1)
    # we use an extended linear system for this
    jac = jacobian(pbwrap, pd.x0, par)
    J = copy(jac.jacpb)
    nj = size(J, 1)
    J[end, :] .= _rand(nj)
    J[:, end] .= _rand(nj)
    J[end, end] = 0
    # enforce PD boundary condition
    J[end-N:end-1, 1:N] .= I(N)
    J[end-N:end-1, end-N:end-1] .= I(N)

    # the transpose problem is a bit different. 
    # transposing J means the boundary condition is wrong
    # however it seems Prop A.1 says the opposite

    rhs = zeros(nj); rhs[end] = 1;
    q = J  \ rhs; q = q[1:end-1]; q ./= norm(q) #≈ ker(J)
    p = J' \ rhs; p = p[1:end-1]; p ./= norm(p)

    # update the borders to have less singular matrix Jψ
    J[end, 1:end-1] .= q
    J[1:end-1, end] .= p

    # left / right Floquet eigenvectors
    vl = J' \ rhs
    vr = J  \ rhs

    v₁  = @view vr[1:end-1]
    v₁★ = @view vl[1:end-1]

    v₁ ./= sqrt(∫(vr, vr)) # this modifies v₁ by reference
    v₁★ ./= 2∫(vl, vr)

    # convention notation. We use the ₛ to indicates time slices which
    # are of size (N, Ntxt⋅m + 1)
    v₁ₛ = get_time_slices(coll, vcat(v₁,1))
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
    # the previous linear problem Jψ
    Jψ = analytical_jacobian(coll, pd.x0, par; _transpose = true, ρ = -1)
    Jψ[end-N:end-1, 1:N] .= -I(N)
    Jψ[end-N:end-1, end-N:end-1] .= I(N)
    # build the extended linear problem
    Jψ[end, :] .= _rand(nj)
    Jψ[:, end] .= _rand(nj)
    Jψ[end, end] = 0

    # update the borders to have less singular matrix Jψ
    q = Jψ  \ rhs; q = q[1:end-1]; q ./= norm(q)
    p = Jψ' \ rhs; p = p[1:end-1]; p ./= norm(p)
    Jψ[end, 1:end-1] .= q
    Jψ[1:end-1, end] .= p

    ψ₁★ = Jψ \ rhs
    ψ₁★ₛ = get_time_slices(coll, ψ₁★)
    ψ₁★ ./= 2∫( ψ₁★ₛ, Fu₀ₛ)
    @assert ∫( ψ₁★ₛ, Fu₀ₛ) ≈ 1/2
    a₁ = ∫(ψ₁★ₛ, Bₛ)

    # computation of h₂
    rhsₛ = @. Bₛ - 2a₁ * Fu₀ₛ
    @assert abs(∫(rhsₛ, ψ₁★ₛ)) < 1e-12
    rhs = vcat(vec(rhsₛ), 0) # it needs to end with zero for the integral condition
    border_ψ₁ = ForwardDiff.gradient(x -> ∫( reshape(x, size(ψ₁★ₛ)), ψ₁★ₛ),
                                     zeros(length(ψ₁★ₛ))
                                    )
    # we could perhaps save the re-computation of J here and use the previous J
    jac = jacobian(pbwrap, pd.x0, par)
    J = copy(jac.jacpb)
    J[end-N:end-1, 1:N] .= -I(N)
    J[end-N:end-1, end-N:end-1] .= I(N)
    # add borders
    J[end, 1:end-1] .= border_ψ₁ # integral condition
    J[:, end] .= ψ₁★
    J[end, end] = 0
    h₂ = J \ rhs
    h₂ ./= 2Ntst # this seems necessary to have something comparable to ApproxFun
    h₂ₛ = get_time_slices(coll, h₂)
    if abs(∫( ψ₁★ₛ, h₂ₛ)) > 1e-10
        @warn "The integral ∫(coll, ψ₁★ₛ, h₂ₛ) should be zero. We found $(∫(  ψ₁★ₛ, h₂ₛ ))"
    end
    if abs(h₂[end]) > 1e-10
        @warn "The value h₂[end] should be zero. We found $(h₂[end])"
    end

    # computation of c. We need B(t, v₁(t), h₂(t))
    for i=1:size(Bₛ, 2)
        Bₛ[:,i]  .= B(u₀ₛ[:,i], par, v₁ₛ[:,i], h₂ₛ[:,i])
    end

    c = 1/(3T) * ∫( v₁★ₛ, Cₛ ) + 
                 ∫( v₁★ₛ, Bₛ ) -
         2a₁/T * ∫( v₁★ₛ, Aₛ )

    nf = (a = a₁, b3 = c) # keep b3 for PD-codim 2
    @debug "" a₁ c c/a₁
    return PeriodDoublingPO(pd.x0, T, v₁, v₁★, (@set pd.nf = nf), coll, false)
end

function period_doubling_normal_form_prm(pbwrap::WrapPOColl,
                                    pd0::PeriodDoubling,
                                    optn::NewtonPar;
                                    nev = 3,
                                    δ = 1e-7,
                                    verbose = false,
                                    lens = getlens(pbwrap),
                                    kwargs_nf...)
    @debug "method PRM"
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    pars = pd0.params
    @debug pars typeof(pd0.x0)
    T = getperiod(coll, pd0.x0, pars)

    Π = PoincareMap(pbwrap, pd0.x0, pars, optn)
    xₛ = pd0.x0[1:N]
    dP = finite_differences(x -> Π(x,pars).u, xₛ)
    F = eigen(dP)

    ####
    ind₋₁ = argmin(abs.(F.values .+ 1))
    ev₋₁ = F.vectors[:, ind₋₁]
    Fp = eigen(dP')
    ind₋₁ = argmin(abs.(Fp.values .+ 1))
    ev₋₁p = Fp.vectors[:, ind₋₁]
    ####
    # Π(xₛ, pars).u - xₛ |> display
    # dP * ev₋₁ + ev₋₁ |> display
    # dP' * ev₋₁p + ev₋₁p |> display

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
    pd = period_doubling_normal_form(probΠ, pd1, DefaultLS(); verbose = verbose)

    # we first try to get the floquet eigenvectors for μ = -1
    jac = jacobian(pbwrap, pd0.x0, pars)
    # remove borders
    J = jac.jacpb
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

    v₁ = @view vr[1:end-1]
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
    # get the bifurcation point parameters
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    verbose && println("━"^53*"\n──▶ Neimark-Sacker normal form computation")
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    par = setparam(br, bifpt.param)
    period = getperiod(coll, bifpt.x, par)

    # get the eigenvalue
    eigRes = br.eig
    λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ωₙₛ = imag(λₙₛ)

    if bifpt.x isa NamedTuple
        # the solution is mesh adapted, we need to restore the mesh.
        pbwrap = deepcopy(pbwrap)
        update_mesh!(pbwrap.prob, bifpt.x._mesh )
        bifpt = @set bifpt.x = bifpt.x.sol
    end
    ns0 = NeimarkSacker(bifpt.x, nothing, bifpt.param, ωₙₛ, par, getlens(br), nothing, nothing, nothing, :none)

    if ~detailed
        return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap, true)
    end

    if prm || ~prm
        # newton parameter
        optn = br.contparams.newton_options
        return neimark_sacker_normal_form_prm(pbwrap, ns0, optn; verbose = verbose, nev = nev, kwargs_nf...)
    end
end
####################################################################################################
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
    @debug "methode PRM"
    coll = pbwrap.prob
    N, m, Ntst = size(coll)
    pars = ns0.params
    T = getperiod(coll, ns0.x0, pars)

    Π = PoincareMap(pbwrap, ns0.x0, pars, optn)
    xₛ = ns0.x0[1:N]
    dP = finite_differences(x -> Π(x,pars).u, xₛ)
    F = eigen(dP)

    _nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
    _nrm > 1e-12 && @warn  "$_nrm"

    ####
    ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
    ev = F.vectors[:, ind]
    Fp = eigen(dP')
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
    ns = neimark_sacker_normal_form(probΠ, ns1, DefaultLS(); verbose = verbose)
    return NeimarkSackerPO(ns0.x0, T, ns0.p, ns0.ω, ev, nothing, ns, coll, true)
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
    return neimark_sacker_normal_form(pbwrap, ns0, (1, 1), optn; verbose = verbose, nev = nev, kwargs_nf...)
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
    _nrm > 1e-12 && @warn  "$_nrm"

    dP = finite_differences(x -> Π(x,pars).u, xₛ)
    # dP = ForwardDiff.jacobian(x -> Π(x,pars).u, xₛ)
    J = jacobian(pbwrap, ns0.x0, pars)
    M = MonodromyQaD(J)

    Fₘ = eigen(M)
    F = eigen(dP)

    ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
    ev = F.vectors[:, ind]
    Fp = eigen(dP')
    indp = argmin(abs.(log.(complex.(Fp.values)) .+ Complex(0, ns0.ω )))
    evp = Fp.vectors[:, indp]

    # normalize eigenvectors
    ev ./= sqrt(dot(ev, ev))
    evp ./= dot(evp, ev)

    @debug "" xₛ ev evp dP _nrm pars F.values[ind] Fp.values[indp]
    @debug "" F.values ns0.x0

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            J = (x,p) -> finite_differences(z -> Π(z,p).u, x),
            d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
            d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
            )

    ns1 = NeimarkSacker(xₛ, nothing, ns0.p, ns0.ω, pars, lens, ev, evp, nothing, :none)
    # normal form computation
    ns = neimark_sacker_normal_form(probΠ, ns1, DefaultLS(); verbose = verbose)

    return NeimarkSackerPO(ns0.x0, period, ns0.p, ns0.ω, real.(ζs), nothing, ns, sh, true)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitTrapProblem}, δp, ampfactor)
    pb = nf.prob

    M, N = size(pb)
    orbitguess0 = nf.po[1:end-1]
    orbitguess0c = get_time_slices(pb, nf.po)
    ζc = reshape(nf.ζ, N, M)
    orbitguess_c = orbitguess0c .+ ampfactor .*  ζc
    orbitguess_c = hcat(orbitguess_c, orbitguess0c .- ampfactor .*  ζc)
    orbitguess = vec(orbitguess_c[:,1:2:end])
    # we append twice the period
    orbitguess = vcat(orbitguess, 2nf.T)
    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pb)
end

function predictor(nf::BranchPointPO{ <: PeriodicOrbitTrapProblem}, δp, ampfactor)
    orbitguess = copy(nf.po)
    orbitguess[1:end-1] .+= ampfactor .*  nf.ζ
    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end

function predictor(nf::NeimarkSackerPO, δp, ampfactor)
    orbitguess = copy(nf.po)
    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitOCollProblem }, δp, ampfactor)
    pbnew = deepcopy(nf.prob)
    N, m, Ntst = size(nf.prob)

    # we update the problem by doubling the Ntst
    pbnew = set_collocation_size(pbnew, 2Ntst, m)

    orbitguess0 = nf.po[1:end-1]

    orbitguess_c = orbitguess0 .+ ampfactor .*  nf.ζ
    orbitguess = vcat(orbitguess_c[1:end-N], orbitguess0 .- ampfactor .*  nf.ζ)

    pbnew.xπ .= orbitguess
    pbnew.ϕ .= circshift(orbitguess, length(orbitguess)÷1)

    # we append twice the period
    orbitguess = vcat(orbitguess, 2nf.T)

    # no need to change pbnew.cache
    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pbnew)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: ShootingProblem }, δp, ampfactor)
    pbnew = deepcopy(nf.prob)
    ζs = nf.ζ
    orbitguess = copy(nf.po)[1:end-1] .+ ampfactor .* ζs
    orbitguess = vcat(orbitguess, copy(nf.po)[1:end-1] .- ampfactor .* ζs, nf.po[end])

    @set! pbnew.M = 2nf.prob.M
    @set! pbnew.ds = _duplicate(pbnew.ds) ./ 2
    orbitguess[end] *= 2
    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pbnew)
end

function predictor(nf::BranchPointPO{ <: ShootingProblem }, δp, ampfactor)
    ζs = nf.ζ
    orbitguess = copy(nf.po)
    orbitguess[1:length(ζs)] .+= ampfactor .* ζs
    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end
####################################################################################################
function predictor(nf::PeriodDoublingPO{ <: PoincareShootingProblem }, δp, ampfactor)
    pbnew = deepcopy(nf.prob)
    ζs = nf.ζ

    @set! pbnew.section = _duplicate(pbnew.section)
    @set! pbnew.M = pbnew.section.M
    orbitguess = copy(nf.po) .+ ampfactor .* ζs
    orbitguess = vcat(orbitguess, orbitguess .- ampfactor .* ζs)

    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = pbnew)
end

function predictor(nf::BranchPointPO{ <: PoincareShootingProblem}, δp, ampfactor)
    ζs = nf.ζ
    orbitguess = copy(nf.po)
    orbitguess .+= ampfactor .* ζs
    return (orbitguess = orbitguess, pnew = nf.nf.p + δp, prob = nf.prob)
end
