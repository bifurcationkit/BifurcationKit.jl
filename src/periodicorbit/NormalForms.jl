"""
$(TYPEDSIGNATURES)

Compute the normal form (NF) of bifurcations of periodic orbits. We detail the additional keyword arguments specific to periodic orbits.

# Optional arguments
- `prm = true` compute the normal form using Poincaré return map (PRM). If false, use the Iooss normal form. 
- `nev = length(eigenvalsfrombif(br, id_bif))`,
- `verbose = false`,
- `ζs = nothing`, pass the eigenvectors
- `lens = getlens(br)`,
- `Teigvec = _getvectortype(br)` type of the eigenvectors (can be useful for GPU)
- `scaleζ = norm`, scale the eigenvector
- `autodiff = false` use autodiff or finite differences in some part of the normal form computation
- `detailed = true` whether to compute only a simplified normal form when only basic information is required. This can be useful is cases the computation is long.
- `δ = getdelta(prob)` delta used for derivatives based on finite differences.

# Notes

For collocation, the default method to compute the NF of Period-doubling and Neimark-Sacker bifurcations is Iooss' one [1].

# References

[1] Iooss, "Global Characterization of the Normal Form for a Vector Field near a Closed Orbit.", 1988
"""
function get_normal_form(wrap::AbstractPeriodicOrbitProblem,
                        br::AbstractResult{ <: PeriodicOrbitCont}, 
                        id_bif::Int,
                        Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                        nev = length(eigenvalsfrombif(br, id_bif)),
                        verbose = false,
                        ζs = nothing,
                        lens = getlens(br),
                        scaleζ = norm,
                        autodiff = false,
                        δ = getdelta(wrap),
                        k...
            ) where {𝒯eigvec}
    bifpt = br.specialpoint[id_bif]

    if bifpt.type == :endpoint
        error("Don't select an end point!")
    end

    # parameters for normal form
    kwargs_nf = (;nev, verbose, lens, scaleζ, k...)

    if bifpt.type == :pd
        return period_doubling_normal_form(wrap, br, id_bif, Teigvec; δ, autodiff, kwargs_nf...)
    elseif bifpt.type == :bp
        return branch_normal_form(wrap, br, id_bif, Teigvec; δ, autodiff, kwargs_nf...)
    elseif bifpt.type == :ns
        return neimark_sacker_normal_form(wrap, br, id_bif, Teigvec; δ, autodiff, kwargs_nf...)
    end
    error("Normal form for $(bifpt.type) not yet implemented.")
end
####################################################################################################
"""
[WIP] Note: the computation of this normal form is not yet fully implemented.
"""
function branch_normal_form(pbwrap,
                            br,
                            ind_bif::Int,
                            Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                            verbose = false,
                            lens = getlens(br),
                            scaleζ = norminf,
                            kwargs_nf...) where {𝒯eigvec}
    disc = get_discretization(pbwrap)
    bifpt = br.specialpoint[ind_bif]
    par = setparam(br, bifpt.param)
    period = getperiod(disc, bifpt.x, par)

    # let us compute the kernel
    λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    verbose && print("├─ computing nullspace of Periodic orbit problem...")
    ζ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
    # we normalize it by the sup norm because it could be too small/big in L2 norm
    ζ ./= scaleζ(ζ)
    verbose && println("Done!")

    # compute the eigenvector for shooting problem, 
    # it is of dimension larger than the Poincaré return map.
    floquetsolver = br.contparams.newton_options.eigsolver
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), real.(ζ))
    ζs = reduce(vcat, ζ_a)

    # normal form for the Poincaré return map
    nf = BranchPoint(bifpt.x, bifpt.τ, bifpt.param, par, lens, nothing, nothing, nothing, :none)

    ζ★ = nothing
    return BranchPointPO(bifpt.x, period, real.(ζs), ζ★, nf, disc, true)
end

function branch_normal_form(pbwrap::WrapPOSh,
                            br,
                            ind_bif::Int,
                            Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                            nev = length(eigenvalsfrombif(br, ind_bif)),
                            verbose = false,
                            lens = getlens(br),
                            scaleζ = norminf,
                            detailed::Val{detailed_type} = Val(true),
                            kwargs_nf...) where {𝒯eigvec, detailed_type}
    verbose && println("━"^53*"\n──▶ Branch point of periodic orbit normal form computation")
    prob_sh = get_discretization(pbwrap)
    bifpt = br.specialpoint[ind_bif]
    pars = setparam(br, bifpt.param)
    period = getperiod(prob_sh, bifpt.x, pars)

    # we compute the kernel:
    # it is two-dimensional. One vector is the trivial vector ∂u₀ where
    # u₀ is the periodic orbit. Hence, ζ₀ = F(u₀(0), par)
    po = get_time_slices(prob_sh, bifpt.x)
    ζ₀ = vector_field(prob_sh.flow, po[:, 1], pars)
    ζ₀ ./= scaleζ(ζ₀)

    # get the non trivial null vector
    spectrum = br.eig[bifpt.idx].eigenvals
    ind = sortperm(spectrum; by = abs)
    verbose && println("┌─ two smallest eigenvalues are ", real.(spectrum[ind[1:2]]))
    verbose && println("├─ computing the non trivial null vector")

    # get the scalar products
    floquetsolver = br.contparams.newton_options.eigsolver
    ζ1 = real(geteigenvector(floquetsolver, br.eig[bifpt.idx].eigenvecs, ind[1]))
    ζ2 = real(geteigenvector(floquetsolver, br.eig[bifpt.idx].eigenvecs, ind[2]))
    ζ1 ./= scaleζ(ζ1)
    ζ2 ./= scaleζ(ζ2)
    _dot_prods = (LA.dot(ζ₀, ζ1), LA.dot(ζ₀, ζ2))
    verbose && println("├─ scalar products with ζ₀ are  ", _dot_prods)
    ind0 = argmax(abs.(abs.(_dot_prods) .- 1))
    ζ = ind0 == 1 ? ζ1 : ζ2
    verbose && println("├─ scalar product dot(ζ₀, ζ) is ", LA.dot(ζ₀, ζ))

    # compute the full eigenvector
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), ζ)
    ζs = reduce(vcat, ζ_a)

    # normal form for Poincaré map
    bp0 = BranchPoint(bifpt.x, bifpt.τ, bifpt.param, pars, getlens(br), nothing, nothing, nothing, :none)

    if ~detailed_type
        ζ★ = nothing
        return BranchPointPO(bifpt.x, period, real.(ζs), ζ★, bp0, prob_sh, true)
    end

    # newton parameter
    optn = br.contparams.newton_options
    branch_point_normal_form(pbwrap, bp0, (ζ, ζs, ζ₀), optn, bifpt.τ; verbose, nev, kwargs_nf...)
end

function _get_spectral_basis_prm_bp(dΠ, verbose)
    # the spectrum of  M is {1,1,...} (M is the Monodromy matrix)
    # the spectrum of dΠ is {1,0,...}
    F  = LA.eigen(dΠ)
    
    ind = argmin(abs.(F.values .- 1))
    λ₁ = F.values[ind] # λ₁ ≈ 1
    verbose && println("├─── [PRM] eigenvalue closest to 1 is ", λ₁)
    verbose && println("└─── [PRM] computing the non trivial null vector")

    # get the scalar products
    ev = F.vectors[:, ind]

    Fp = LA.eigen(dΠ')
    indp = argmin(abs.(Fp.values .- λ₁)) # eigenvalue closest to λ₁
    ev★ = Fp.vectors[:, indp]

    # normalize eigenvectors
    ev ./= sqrt(LA.dot(ev, ev))
    ev★ ./= LA.dot(ev★, ev)
    return (;ev, ev★)
end

function branch_point_normal_form(pbwrap::WrapPOSh{ <: ShootingProblem },
                                    bp0::BranchPoint,
                                    (ζ₁, ζs, ζₚₒ),
                                    optn::NewtonPar,
                                    τ;
                                    nev = 3,
                                    verbose = false,
                                    lens = getlens(pbwrap),
                                    autodiff = false,
                                    scaleζ = norminf,
                                    detailed::Val{detailed_type} = Val(true),
                                    kwargs_nf...) where {detailed_type}
    # ζₚₒ is the trivial eigenvector ∂ₜu₀(0)=F(u₀(0)) where u₀ is the periodic orbit
    sh = get_discretization(pbwrap)
    pars = bp0.params
    period = getperiod(sh, bp0.x0, pars)
    # compute the Poincaré return map, the section is on the first time slice
    Π = PoincareMap(pbwrap, bp0.x0, pars, optn)
    xₛ = get_time_slices(sh, Π.po)[:, 1]

    _nrm = norm(Π(xₛ, pars).u .- xₛ, Inf)
    _nrm > 1e-12 && @warn  "[BP normal form PRM], residual = $_nrm"
    
    dΠ = jacobian(Π, xₛ ,pars) # this is close to the finite differences, hence analytical expression should be good
    (;ev, ev★) = _get_spectral_basis_prm_bp(dΠ, verbose)

    probΠ = BifurcationProblem(
                    (x,p) -> Π(x,p).u,
                    xₛ, pars, lens ;
                    J = (x,p) -> jacobian(Π,x,p),
                    jvp = (x,p,dx) -> apply(jacobian(Π,x,p), dx),
                    d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
                    d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
    )

    ζ★ = nothing
    bp1 = BranchPointMap(xₛ, τ, bp0.p, pars, lens, ev, ev★, nothing, :none)
    if detailed_type
        bp = get_normal_form1d_maps(probΠ, bp1, optn.linsolver; verbose, autodiff)
        return BranchPointPO(bp0.x0, period, real.(ζs), ζ★, bp, sh, true)
    end
    return BranchPointPO(bp0.x0, period, real.(ζs), ζ★, nothing, sh, true)
end

function branch_normal_form(pbwrap::WrapPOColl,
                            br,
                            ind_bif::Int,
                            Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                            verbose = false,
                            nev = length(eigenvalsfrombif(br, ind_bif)),
                            prm::Val{prm_type} = Val(false),
                            detailed::Val{detailed_type} = Val(true),
                            kwargs_nf...) where {𝒯eigvec, prm_type, detailed_type}
    @debug "BP-PO normal form for Collocation: use prm by default"
    prm = true
    # first, get the bifurcation point parameters
    verbose && println("━"^53*"\n──▶ Branch point normal form computation")
    bifpt = br.specialpoint[ind_bif]
    par = setparam(br, bifpt.param)
    coll = get_discretization(pbwrap)
    period = getperiod(coll, bifpt.x, par)
    
    if bifpt.x isa POSolutionAndState
        # the solution is mesh adapted, we need to restore the mesh.
        pbwrap = deepcopy(pbwrap)
        update_mesh!(coll, bifpt.x._mesh)
        bifpt = @set bifpt.x = bifpt.x.sol
    end
    
    bp0 = BranchPoint(bifpt.x, bifpt.τ, bifpt.param, par, getlens(br), nothing, nothing, nothing, :none)
    if ~prm_type || ~detailed_type
        # method based on Iooss method
        return branch_normal_form_iooss(pbwrap, bp0; detailed, verbose, nev, kwargs_nf...)
    end
    # method based on Poincaré Return Map (PRM), newton parameter
    optn = br.contparams.newton_options
    # @warn "[BP-PO NF] Computation of BP-PO normal form based on Poincaré return map is not yet unavailable.\nDefaulting to the one based on Iooss form." # A VIRER
    return branch_normal_form_prm(pbwrap, bp0, optn; verbose, nev, kwargs_nf...)
end

function _get_spectral_basis_iooss_bp(pbwrap::WrapPOColl,
                                      bp0::BranchPoint;
                                        )
    coll = get_discretization(pbwrap)
    𝒯 = eltype(coll)
    N, m, Ntst = size(coll)
    pars = bp0.params
    period = getperiod(coll, bp0.x0, pars)
    _rand(n, r = 2) = 𝒯(r) .* (rand(𝒯, n) .- 1//2)

    # we get the nontrivial floquet eigenvectors μ = 1
    # We could use Witte, Virginie De. “Computational Analysis of Bifurcations of Periodic Orbits,” n.d.
    # formula (6.9) on page 201
    # but I am not sure that the formula is correct, ie having a Jordan block [0 1; 0 0].
    # We thus find the 2d kernel using Bordering strategy.
    # We need to know which eigenvector is closest to F(u₀)
    u₀ₛ = get_time_slices(coll, bp0.x0) # periodic solution at bifurcation
    Fu₀ₛ = copy(u₀ₛ)
    Fu₀ = vec(Fu₀ₛ)
    for i in axes(u₀ₛ, 2)
        Fu₀ₛ[:, i] .= residual(coll.prob_vf, u₀ₛ[:, i], pars)
    end

    jac = jacobian(pbwrap, bp0.x0, pars)
    J = copy((jac))
    nj = size(J, 1)
    J[end, begin:end-1] .= Fu₀
    J[:, end] .= _rand(nj)
    J[end,end] = 0
    rhs = zeros(𝒯, nj); rhs[end] = 1

    q = J  \ rhs
    p = J' \ rhs

    # doing this again makes p[end] ≈ 0
    J[end, begin:end-1] .= q[begin:end-1]
    J[begin:end-1, end] .= p[begin:end-1]
    
    # the matrix L₀ = 0
    Jbd = similar(J, nj+1, nj+1) # carefull for sparse matrices
    Jbd[1:nj,1:nj] .= J
    Jbd[1:nj-1,end] .= Fu₀# ∂(coll, vcat(Fu₀,period), pars)[begin:end-1]
    Jbd[end,1:nj-1] .= randn(nj-1)
    Jbd[end-1:end,end-1:end] .= 0
    rhs = zeros(𝒯, nj+1, 2); rhs[end-1:end,end-1:end] .= LA.I(2)
    sols   = Jbd  \ rhs
    sols_t = Jbd' \ rhs

    q₀ = sols[1:nj-1,1]
    q₁ = sols[1:nj-1,2]
    p₀ = sols_t[1:nj-1,1]
    p₁ = sols_t[1:nj-1,2]

    Jbd[1:nj-1,end] .= q₀# ∂(coll, vcat(Fu₀,period), pars)[begin:end-1]
    Jbd[end,1:nj-1] .= p₀

    sols   = Jbd  \ rhs
    q₀ = sols[1:nj-1,1]
    q₁ = sols[1:nj-1,2]

    sols_t = Jbd' \ rhs
    p₀ = sols_t[1:nj-1,1]
    p₁ = sols_t[1:nj-1,2]

    J0 = J[begin:end-1,begin:end-1]

    _ps = (LA.dot(q₀, Fu₀), LA.dot(q₁, Fu₀))
    ind = argmin(abs, _ps)
    # v₁ = q₁#ind==1 ? q₀ : q₁
    v₁ = q₁ ./ norminf(q₁)
    v₀ = q₀ ./ norminf(q₀)
    @debug "" ind _ps J0 * v₁|>norminf J0*v₀|>norminf
    return (;v₁, v₀, coll, period, p₀, p₁, u₀ₛ, Fu₀ₛ, Fu₀)
end

function branch_normal_form_prm(pbwrap::WrapPOColl,
                                bp0::BranchPoint,
                                optn::NewtonPar;
                                nev = 3,
                                δ = 1e-7,
                                verbose = false,
                                lens = getlens(pbwrap),
                                autodiff = false,
                                scaleζ = norminf,
                                detailed = true,
                                kwargs_nf...)
    @debug "PD normal form collocation, method PRM"
    coll = get_discretization(pbwrap)
    𝒯 = eltype(coll)
    N, m, Ntst = size(coll)
    pars = bp0.params
    period = getperiod(coll, bp0.x0, pars)
    # compute the Poincaré return map, the section is on the first time slice
    Π = PoincareMap(pbwrap, bp0.x0, pars, optn)
    xₛ = bp0.x0[1:N]

    _nrm = norm(Π(xₛ, pars).u .- xₛ, Inf)
    _nrm > 1e-12 && @warn  "[BP normal form PRM], residual = $_nrm"

    dΠ = finite_differences(x -> Π(x, pars).u, xₛ)
    (;ev, ev★) = _get_spectral_basis_prm_bp(dΠ, verbose)

    δ1 = convert(𝒯, δ)
    δ2 = sqrt(δ1)
    δ3 = δ1^(1/3)
    d1Π(x,p,dx) = (Π(x .+ δ1 .* dx, p).u .- Π(x .- δ1 .* dx, p).u) ./ (2δ1)
    d2Π(x,p,dx1,dx2) = (d1Π(x .+ δ2 .* dx2, p, dx1) .- d1Π(x .- δ2 .* dx2, p, dx1)) ./ (2δ2)
    d3Π(x,p,dx1,dx2,dx3) = (d2Π(x .+ δ3 .* dx3, p, dx1, dx2) .- d2Π(x .- δ3 .* dx3, p, dx1, dx2)) ./ (2δ3)

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            jvp = d1Π,
            J = (x,p) -> finite_differences(z -> Π(z,p).u, x),
            # d2F = (x,p,h1,h2) -> d2F(Π,x,p,h1,h2).u,
            # d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
            d2F = d2Π,
            d3F = d3Π,
            )

    ζ★ = nothing
    bp1 = BranchPointMap(xₛ, nothing, bp0.p, pars, lens, ev, ev★, nothing, :none)
    if detailed
        bp = get_normal_form1d_maps(probΠ, bp1, optn.linsolver; verbose, autodiff)
        (;v₁, v₀, p₀, p₁) = _get_spectral_basis_iooss_bp(pbwrap, bp0)
        return BranchPointPO(bp0.x0, period, (v₀, v₁), (p₀, p₁), bp, coll, true)
    end
    return BranchPointPO(bp0.x0, period, real.(ζs), ζ★, nothing, coll, true)
end

function branch_normal_form_iooss(pbwrap::WrapPOColl,
                                    bp0::BranchPoint;
                                    nev = 3,
                                    δ = getdelta(pbwrap),
                                    verbose = false,
                                    lens = getlens(pbwrap),
                                    kwargs_nf...)
    @debug "BP normal form collocation, method Iooss"
    (;v₁, v₀, coll, period, p₀, p₁) = _get_spectral_basis_iooss_bp(pbwrap, bp0)
    
    # plot(layout = @layout [a;b;c;d])
    # vsol = get_periodic_orbit(coll, bp0.x0,1)
    # plot!(vsol, ylabel = "po", subplot = 1, legend = :topright)
    # vsol = get_periodic_orbit(coll, vcat(vec(q₀./norminf(vec(q₀))),period), 1)
    # plot!(vsol, ylabel = "q₀", subplot = 2, labels = collect(1:4)', legend = :topright)
    # vsol = get_periodic_orbit(coll, vcat(vec(q₁./norminf(vec(q₁))),period), 1)
    # plot!(vsol, linewidth=2, ylabel = "q₁", subplot=3, labels = collect(1:4)', legend = :topright)
    # vsol = get_periodic_orbit(coll, vcat(v₁,period), 1)
    # plot!(vsol, linewidth=2, ylabel = "v₁", subplot=4, labels = collect(1:4)') |> display
    # @assert false
    return BranchPointPO(bp0.x0, period, (v₀, v₁), (p₀, p₁), bp0, coll, true)
end
####################################################################################################
function period_doubling_normal_form(pbwrap,
                                br,
                                ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                lens = getlens(br),
                                scaleζ = norminf,
                                kwargs_nf...) where {𝒯eigvec}
    pb = get_discretization(pbwrap)
    bifpt = br.specialpoint[ind_bif]
    pars = setparam(br, bifpt.param)
    period = getperiod(pb, bifpt.x, pars)

    # let us compute the kernel
    floquetsolver = br.contparams.newton_options.eigsolver
    λ = (br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    ζ = geteigenvector(floquetsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev)
    # we normalize it by the sup norm because it could be too small/big in L2 norm
    ζ ./= scaleζ(ζ)
    verbose && println("Done!")

    # compute the full eigenvector
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), real.(ζ))
    ζs = reduce(vcat, ζ_a)

    # basic normal form structure, empty for now, just a wrapper for the eigenvectors
    nf = PeriodDoubling(bifpt.x, nothing, bifpt.param, pars, getlens(br), nothing, nothing, nothing, :none)
    ζ★ = nothing
    return PeriodDoublingPO(bifpt.x, period, real.(ζs), ζ★, nf, pb, true)
end

function period_doubling_normal_form(pbwrap::WrapPOSh,
                                br,
                                ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                lens = getlens(br),
                                detailed::Val{detailed_type} = Val(true),
                                scaleζ = norminf,
                                kwargs_nf...) where {𝒯eigvec, detailed_type}
    verbose && println("━"^53*"\n──▶ Period-doubling normal form computation")
    bifpt = br.specialpoint[ind_bif]
    pars = setparam(br, bifpt.param)

    # let us compute the kernel
    λ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
    verbose && print("├─ computing nullspace of Periodic orbit problem...")
    ζ₋₁ = geteigenvector(br.contparams.newton_options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev) .|> real
    # we normalize it by the sup norm because it could be too small/big in L2 norm
    ζ₋₁ ./= scaleζ(ζ₋₁)
    verbose && println("Done!")

    # compute the full eigenvector
    floquetsolver = br.contparams.newton_options.eigsolver
    ζ_a = floquetsolver(Val(:ExtractEigenVector), pbwrap, bifpt.x, setparam(br, bifpt.param), real.(ζ₋₁))
    ζs = reduce(vcat, ζ_a)

    pd0 = PeriodDoubling(bifpt.x, nothing, bifpt.param, pars,lens, nothing, nothing, nothing, :none)
    if ~detailed_type
        sh = get_discretization(pbwrap)
        period = getperiod(sh, pd0.x0, pd0.params)
        ζ★ = nothing
        return PeriodDoublingPO(pd0.x0, period, real.(ζs), ζ★, pd0, sh, true)
    end

    # newton parameter
    optn = br.contparams.newton_options
    return period_doubling_normal_form(pbwrap, pd0, (ζ₋₁, ζs), optn; verbose, nev, kwargs_nf...)
end

function period_doubling_normal_form(pbwrap::WrapPOSh{ <: PoincareShootingProblem },
                                pd0::PeriodDoubling,
                                (ζ₋₁, ζs),
                                optn::NewtonPar;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                kwargs_nf...)
    psh = get_discretization(pbwrap)
    period = getperiod(psh, pd0.x0, pd0.params)
    ζ★ = nothing
    return PeriodDoublingPO(pd0.x0, period, real.(ζs), ζ★, pd0, psh, true)
end

function period_doubling_normal_form(pbwrap::WrapPOSh{ <: ShootingProblem },
                                pd0::PeriodDoubling,
                                (ζ₋₁, ζs),
                                optn::NewtonPar;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                δ = 1e-9,
                                autodiff = false,
                                kwargs_nf...)
    sh = get_discretization(pbwrap)
    pars = pd0.params
    period = getperiod(sh, pd0.x0, pars)
    # compute the Poincaré return map, the section is on the first time slice
    Π = PoincareMap(pbwrap, pd0.x0, pars, optn)
    # Π = PoincareCallback(pbwrap, pd0.x0, pars; radius = 0.1)
    xₛ = get_time_slices(sh, Π.po)[:, 1]

    # If M is the monodromy matrix and E := x - <x,e>e with e the eigen
    # vector of M for the eigenvalue 1, then, we find that
    # eigenvector(P) = E ∘ eigenvector(M)
    # E(x) = x .- dot(ζ₁, x) .* ζ₁

    _nrm = norminf(Π(xₛ, pars).u - xₛ)
    _nrm > 1e-10 && @warn "Residual seems large = $_nrm"

    # dΠ = finite_differences(x -> Π(x, pars).u, xₛ; δ)
    dΠ = jacobian(Π, xₛ, pars)
    F = LA.eigen(dΠ)

    ind₋₁ = argmin(abs.(F.values .+ 1))
    ev₋₁ = F.vectors[:, ind₋₁]
    F★ = LA.eigen(dΠ')
    ind₋₁ = argmin(abs.(F★.values .+ 1))
    ev₋₁★ = F★.vectors[:, ind₋₁]

    # normalize eigenvectors
    ev₋₁ ./= sqrt(LA.dot(ev₋₁, ev₋₁))
    ev₋₁★ ./= LA.dot(ev₋₁, ev₋₁★)

    probΠ = BifurcationProblem(
            (x,p) -> Π(x,p).u,
            xₛ, pars, lens ;
            J = (x,p) -> jacobian(Π, x, p),
            d2F = (x,p,h1,h2)    -> d2F(Π,x,p,h1,h2).u,
            d3F = (x,p,h1,h2,h3) -> d3F(Π,x,p,h1,h2,h3).u
            )

    pd1 = PeriodDoubling(xₛ, nothing, pd0.p, pars, lens, ev₋₁, ev₋₁★, nothing, :none)
    # normal form computation
    pd = period_doubling_normal_form(probΠ, pd1, optn.linsolver; verbose)
    ζ★ = nothing
    return PeriodDoublingPO(pd0.x0, period, real.(ζs), ζ★, pd, sh, true)
end

function period_doubling_normal_form(pbwrap::WrapPOColl,
                                br,
                                ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                verbose = false,
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                prm::Val{prm_type} = Val(false),
                                detailed::Val{detailed_type} = Val(true),
                                kwargs_nf...) where {𝒯eigvec, prm_type, detailed_type}
    # first, get the bifurcation point parameters
    verbose && println("━"^53*"\n──▶ Period-Doubling normal form computation")
    bifpt = br.specialpoint[ind_bif]
    bptype = bifpt.type
    par = setparam(br, bifpt.param)
    coll = get_discretization(pbwrap)
    period = getperiod(coll, bifpt.x, par)

    if bifpt.x isa POSolutionAndState
        # the solution is mesh adapted, we need to restore the mesh.
        pbwrap = deepcopy(pbwrap)
        update_mesh!(coll, bifpt.x._mesh )
        bifpt = @set bifpt.x = bifpt.x.sol
    end
    pd0 = PeriodDoubling(bifpt.x, nothing, bifpt.param, par, getlens(br), nothing, nothing, nothing, :none)
    if ~prm_type
        # method based on Iooss method
        return period_doubling_normal_form_iooss(pbwrap, pd0; detailed, verbose, nev, kwargs_nf...)
    end
    # method based on Poincaré Return Map (PRM)
    # newton parameter
    optn = br.contparams.newton_options
    return period_doubling_normal_form_prm(pbwrap, pd0, optn; verbose, nev, kwargs_nf...)
end

function period_doubling_normal_form_iooss(pbwrap,
                                pd::PeriodDoubling;
                                nev = 3,
                                verbose = false,
                                lens = getlens(pbwrap),
                                detailed::Val{detailed_type} = Val(true),
                                kwargs_nf...) where {detailed_type}
    # function based on the article
    # Kuznetsov, Yu. A., W. Govaerts, E. J. Doedel, and A. Dhooge. “Numerical Periodic Normalization for Codim 1 Bifurcations of Limit Cycles.” SIAM Journal on Numerical Analysis https://doi.org/10.1137/040611306.
    # on page 1243
    # there are a lot of mistakes in the above paper, it seems better to look at https://webspace.science.uu.nl/~kouzn101/NBA/LC2.pdf
    # see also Witte, V. De, F. Della Rossa, W. Govaerts, and Yu. A. Kuznetsov. “Numerical Periodic Normalization for Codim 2 Bifurcations of Limit Cycles” SIAM Journal on Applied Dynamical Systems. https://doi.org/10.1137/120874904.
    coll = get_discretization(pbwrap)
    N = get_state_dim(coll)
    par = pd.params
    p₀ = _get(par, lens)
    T = getperiod(coll, pd.x0, par)
    lens = getlens(coll)
    δ = getdelta(coll)
    𝒯 = eltype(coll)

    # identity matrix for collocation problem
    Icoll = I(coll, _getsolution(pd.x0), par)

    F(u, p) = residual(coll.prob_vf, u, p)
    # dₚF(u, p) = ForwardDiff.derivative(z -> residual(coll.prob_vf, u, set(p, lens, z)), get(par, lens))
    dₚF(u, p) = (residual(coll.prob_vf, u, set(p, lens, p₀ + δ)) .- 
                 residual(coll.prob_vf, u, set(p, lens, p₀ - δ))) ./ (2δ)
    A(u, p, du) = apply(jacobian(coll.prob_vf, u, p), du)
    F11(u, p, du) = (A(u, set(p, lens, p₀ + δ), du) .- 
                     A(u, set(p, lens, p₀ - δ), du)) ./ (2δ)
    B(u, p, du1, du2)      = d2F(coll.prob_vf, u, p, du1, du2)
    C(u, p, du1, du2, du3) = d3F(coll.prob_vf, u, p, du1, du2, du3)

    _rand(n, r = 2) = 𝒯(r) .* (rand(𝒯, n) .- 1//2)  # centered uniform random variables
    local ∫(u, v) = BifurcationKit.∫(coll, u, v, 1) # define integral with coll parameters

    # we first compute the floquet eigenvector for μ = -1
    # we use an extended linear system for this
    #########
    # compute v1
    jac = jacobian(pbwrap, _getsolution(pd.x0), par)
    J = copy(jac) # we put copy to not alias FloquetWrapper.jacpb
    nj = size(J, 1)
    J[end, :] .= _rand(nj)
    J[:, end] .= _rand(nj)
    J[end, end] = 0
    # enforce PD boundary condition
    J[end-N:end-1, 1:N] .= LA.I(N)
    J[end-N:end-1, end-N:end-1] .= LA.I(N)

    rhs = zeros(𝒯, nj); rhs[end] = 1;
    k = J  \ rhs; k = k[begin:end-1]; k ./= norm(k) #≈ ker(J)
    l = J' \ rhs; l = l[begin:end-1]; l ./= norm(l)

    # update the borders to have less singular matrix J
    J[end, begin:end-1] .= k
    J[begin:end-1, end] .= l

    # right Floquet eigenvectors
    vr = J \ rhs

    v₁  = @view vr[begin:end-1]
    v₁ ./= sqrt(∫(vr, vr)) # this modifies v₁ by reference

    #########
    # compute v1★
    J★ = analytical_jacobian(coll, _getsolution(pd.x0), par; _transpose = Val(true), ρF = -1)
    J★[end, :] .= _rand(nj)
    J★[:, end] .= _rand(nj)
    J★[end, end] = 0
    # enforce PD boundary condition
    J★[end-N:end-1, 1:N] .= LA.I(N)
    J★[end-N:end-1, end-N:end-1] .= LA.I(N)

    rhs = zeros(𝒯, nj); rhs[end] = 1;
    k = J★  \ rhs; k = k[begin:end-1]; k ./= norm(k) # ≈ ker(J)
    l = J★' \ rhs; l = l[begin:end-1]; l ./= norm(l)

    vl = J★ \ rhs
    v₁★ = @view vl[begin:end-1]
    v₁★ ./= 2∫(vl, vr)

    # convention notation. We use the ₛ to indicates time slices which
    # are of size (N, Ntxt⋅m + 1)
    v₁ₛ  = get_time_slices(coll, vcat(v₁ , 1))
    v₁★ₛ = get_time_slices(coll, vcat(v₁★, 1))

    @assert ∫(v₁★ₛ, v₁ₛ) ≈ 1/2
    @assert ∫(v₁ₛ, v₁ₛ) ≈ 1

    # if we just want the eigenvectors
    if ~detailed_type
        return PeriodDoublingPO(pd.x0, T, v₁, v₁★, (@set pd.nf = (a = 0, b3 = 0)), coll, false)
    end

    u₀ₛ = get_time_slices(coll, pd.x0) # periodic solution at bifurcation
    Fu₀ₛ = copy(u₀ₛ)
    Aₛ   = copy(u₀ₛ)
    Bₛ   = copy(u₀ₛ)
    Cₛ   = copy(u₀ₛ)
    for i = 1:size(u₀ₛ, 2)
      Fu₀ₛ[:, i] .= F(u₀ₛ[:, i], par)
        Aₛ[:, i] .= A(u₀ₛ[:, i], par, v₁ₛ[:, i])
        Bₛ[:, i] .= B(u₀ₛ[:, i], par, v₁ₛ[:, i], v₁ₛ[:, i])
        Cₛ[:, i] .= C(u₀ₛ[:, i], par, v₁ₛ[:, i], v₁ₛ[:, i], v₁ₛ[:, i])
    end

    # computation of ψ★, recall the BC ψ★(0) = ψ★(1)
    # for this, we generate the linear problem analytically
    # note that we could obtain the same by modifying inplace 
    # the previous linear problem J
    Jψ = analytical_jacobian(coll, _getsolution(pd.x0), par; _transpose = Val(true), ρF = -1)
    Jψ[end-N:end-1, 1:N] .= -LA.I(N)
    Jψ[end-N:end-1, end-N:end-1] .= LA.I(N)
    # build the extended linear problem
    Jψ[end, :] .= _rand(nj)
    Jψ[:, end] .= _rand(nj)
    Jψ[end, end] = 0

    # update the borders to have less singular matrix Jψ
    k = Jψ  \ rhs; k = k[begin:end-1]; k ./= norm(k)
    l = Jψ' \ rhs; l = l[begin:end-1]; l ./= norm(l)
    Jψ[end, begin:end-1] .= k
    Jψ[begin:end-1, end] .= l

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
    if abs(∫(rhsₛ, ψ₁★ₛ)) > 1e-12 
        @warn "[PD-Iooss] The integral ∫(rhsₛ, ψ₁★ₛ) should be zero. We found $(∫(rhsₛ, ψ₁★ₛ))"
    end
    rhs = vcat(vec(rhsₛ), 0) # it needs to end with zero for the integral condition
    border_ψ₁ = ForwardDiff.gradient(x -> ∫( reshape(x, size(ψ₁★ₛ)), ψ₁★ₛ),
                                     zeros(length(ψ₁★ₛ))
                                    )
                            # _plot(vcat(vec(rhsₛ),1))
    # we could perhaps save the re-computation of J here and use the previous J
    jac = jacobian(pbwrap, _getsolution(pd.x0), par)
    J = copy(jac)
    J[end-N:end-1, 1:N] .= -LA.I(N)
    J[end-N:end-1, end-N:end-1] .= LA.I(N)
    # add borders
    J[end, begin:end-1] .= border_ψ₁ # integral condition
    J[:, end] .= ψ₁★
    J[end, end] = 0
    h₂ = J \ (Icoll * rhs)
    # h₂ ./= 2Ntst # this seems necessary to have something comparable to ApproxFun
                # h₂ = Icoll * h₂;@reset h₂[end]=0
    h₂ₛ = get_time_slices(coll, h₂)
                # a cause de Icoll
                # h₂ₛ[:, end] .= h₂ₛ[:,1]
    if abs(∫( ψ₁★ₛ, h₂ₛ)) > 1e-10
        @warn "[PD-Iooss] The integral ∫(ψ₁★ₛ, h₂ₛ) should be zero. We found $(∫(  ψ₁★ₛ, h₂ₛ ))"
    end
    if abs(h₂[end]) > 1e-10
        @warn "[PD-Iooss] The value h₂[end] should be zero. We found $(h₂[end])"
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

    @debug "[PD-Iooss]" ∫( v₁★ₛ, Bₛ ) 2a₁/T * ∫( v₁★ₛ, Aₛ )

    # computation of a₀₁
    ∂Fu₀ₛ = copy(u₀ₛ)
    for i = 1:size(u₀ₛ, 2)
        ∂Fu₀ₛ[:, i] .= dₚF(u₀ₛ[:, i], par)
    end
    a₀₁ = 2∫(ψ₁★ₛ, ∂Fu₀ₛ)

    # computation of h₀₁
    #                     ∂ₜh₀₁ - A(t)h₀₁ = F₀₁(t) - a₀₁⋅∂u₀
    rhsₛ = copy(u₀ₛ)
    for i = 1:size(u₀ₛ, 2)
        rhsₛ[:, i] .= ∂Fu₀ₛ[:, i] .- a₀₁ .* Fu₀ₛ[:, i]
    end
    rhs = vcat(vec(rhsₛ), 0) # it needs to end with zero for the integral condition
    jac = jacobian(pbwrap, _getsolution(pd.x0), par)
    J = copy(jac)
    J[end-N:end-1, 1:N] .= -LA.I(N)
    J[end-N:end-1, end-N:end-1] .= LA.I(N)
    # add borders
    J[end, begin:end-1] .= border_ψ₁ # integral condition
    J[:, end] .= ψ₁★
    J[end, end] = 0
    h₀₁ = J \ (Icoll * rhs)
    h₀₁ₛ = get_time_slices(coll, h₀₁)

    # computation of c₁₁
    #                   < w★, -B(t,h01,w) - F11*w + c11*w + a01*wdot > = 0
    # hence:
    #                   c11 = < w★, B(t,h01,w) + F11*w + c11*w - a01*wdot >
    for i = 1:size(u₀ₛ, 2)
        rhsₛ[:, i] .= B(u₀ₛ[:, i], par, v₁★ₛ[:, i], h₀₁ₛ[:, i]) .+ F11(u₀ₛ[:, i], par, v₁★ₛ[:, i])
    end

    c₁₁ = ∫(v₁★ₛ, rhsₛ) - a₀₁ * ∫(v₁★ₛ, Aₛ)
    c₁₁ *= 2

    # we want the parameter a, not the rescaled a₁
    nf = (a = a₁/T, b3 = c, h₂ₛ, ψ₁★ₛ, v₁ₛ, a₀₁, c₁₁) # keep b3 for PD-codim 2
    newpd = @set pd.nf = nf
    @debug "[PD-NF-Iooss]" a₁ c
    if real(c) < 0
        @reset newpd.type = :SuperCritical
    else
        @reset newpd.type = :SubCritical
    end
    return PeriodDoublingPO(pd.x0, T, v₁, v₁★, newpd, coll, false)
end

function period_doubling_normal_form_prm(pbwrap::WrapPOColl,
                                    pd0::PeriodDoubling,
                                    optn::NewtonPar;
                                    nev = 3,
                                    δ = 1e-7,
                                    verbose = false,
                                    detailed::Val{detailed_type} = Val(true),
                                    lens = getlens(pbwrap),
                                    kwargs_nf...) where {detailed_type}
    @debug "PD normal form collocation, method PRM"
    coll = get_discretization(pbwrap)
    𝒯 = eltype(coll)
    N, m, Ntst = size(coll)
    pars = pd0.params
    T = getperiod(coll, pd0.x0, pars)

    Π = PoincareMap(pbwrap, pd0.x0, pars, optn)
    xₛ = pd0.x0[1:N]
    dΠ = finite_differences(x -> Π(x, pars).u, xₛ)
    F = LA.eigen(dΠ)

    ind₋₁ = argmin(abs.(F.values .+ 1))
    ev₋₁ = F.vectors[:, ind₋₁]
    Fp = LA.eigen(dΠ')
    ind₋₁ = argmin(abs.(Fp.values .+ 1))
    ev₋₁p = Fp.vectors[:, ind₋₁]

    # normalize eigenvectors
    ev₋₁ ./= sqrt(LA.dot(ev₋₁, ev₋₁))
    ev₋₁p ./= LA.dot(ev₋₁, ev₋₁p)

    δ1 = convert(𝒯, δ)
    δ2 = sqrt(δ1)
    δ3 = δ1^(1/3)
    d1Π(x,p,dx) = (Π(x .+ δ1 .* dx, p).u .- Π(x .- δ1 .* dx, p).u) ./ (2δ1)
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
    pd = detailed_type ? period_doubling_normal_form(probΠ, pd1, optn.linsolver; verbose) : nothing

    # we get the floquet eigenvectors for μ = -1
    jac = jacobian(pbwrap, pd0.x0, pars)
    # remove borders
    J = copy((jac))
    nj = size(J, 1)
    J[end, :] .= rand(nj)
    J[:, end] .= rand(nj)
    # enforce PD boundary condition
    J[end-N:end-1, 1:N] .= LA.I(N)
    rhs = zeros(nj); rhs[end] = 1
    q = J  \ rhs; q = q[begin:end-1]; q ./= norm(q)
    p = J' \ rhs; p = p[begin:end-1]; p ./= norm(p)

    J[end, begin:end-1] .= q
    J[begin:end-1, end] .= p

    vl = J' \ rhs
    vr = J  \ rhs

    v₁  = @view vr[begin:end-1]
    v₁★ = @view vl[begin:end-1]

    return PeriodDoublingPO(pd0.x0, pd0.x0[end], v₁, v₁★, pd, coll, true)
end
####################################################################################################
function neimark_sacker_normal_form(pbwrap::AbstractPeriodicOrbitProblem,
                                br::AbstractBranchResult,
                                ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                lens = getlens(br),
                                scaleζ = norminf,
                                kwargs_nf...) where {𝒯eigvec}
    disc = get_discretization(pbwrap)
    bifpt = br.specialpoint[ind_bif]
    pars = setparam(br, bifpt.param)
    period = getperiod(disc, bifpt.x, pars)

    # get the eigenvalue
    eigRes = br.eig
    λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ωₙₛ = imag(λₙₛ)

    ns0 = NeimarkSacker(bifpt.x, nothing, bifpt.param, ωₙₛ, pars, getlens(br), nothing, nothing, nothing, :none)
    return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap, true)
end

function neimark_sacker_normal_form(pbwrap::WrapPOColl,
                                br::AbstractBranchResult,
                                ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                verbose = false,
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                prm::Val{prm_type} = Val(false),
                                detailed::Val{detailed_type} = Val(true),
                                kwargs_nf...) where {𝒯eigvec, prm_type, detailed_type}
    verbose && println("━"^53*"\n──▶ Neimark-Sacker normal form computation")
    # get the bifurcation point parameters
    coll = get_discretization(pbwrap)
    N, m, Ntst = size(coll)
    bifpt = br.specialpoint[ind_bif]
    par = setparam(br, bifpt.param)
    period = getperiod(coll, bifpt.x, par)

    # get the eigenvalue
    eigRes = br.eig
    λₙₛ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ωₙₛ = abs(imag(λₙₛ))

    if bifpt.x isa POSolutionAndState
        # the solution is mesh adapted, we need to restore the mesh.
        pbwrap = deepcopy(pbwrap)
        update_mesh!(coll, bifpt.x._mesh )
        bifpt = @set bifpt.x = bifpt.x.sol
    end
    ns0 = NeimarkSacker(bifpt.x, nothing, bifpt.param, ωₙₛ, par, getlens(br), nothing, nothing, nothing, :none)

    if ~detailed_type
        return NeimarkSackerPO(bifpt.x, period, bifpt.param, ωₙₛ, nothing, nothing, ns0, pbwrap, true)
    end

    if prm_type # method based on Poincare Return Map (PRM)
        # newton parameter
        optn = br.contparams.newton_options
        return neimark_sacker_normal_form_prm(pbwrap, ns0, optn; verbose = verbose, nev = nev, kwargs_nf...)
    end
    # method based on Iooss method
    neimark_sacker_normal_form_iooss(pbwrap, ns0; verbose, nev, kwargs_nf...)
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
    coll = get_discretization(pbwrap)
    𝒯 = eltype(coll)
    N, m, Ntst = size(coll)
    pars = ns0.params
    T = getperiod(coll, ns0.x0, pars)

    Π = PoincareMap(pbwrap, ns0.x0, pars, optn)
    xₛ = ns0.x0[1:N]
    dΠ = finite_differences(x -> Π(x, pars).u, xₛ)
    F = LA.eigen(dΠ)

    _nrm = norm(Π(xₛ, pars).u - xₛ, Inf)
    _nrm > 1e-12 && @warn  "$_nrm"

    ####
    ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
    ev = F.vectors[:, ind]
    Fp = LA.eigen(dΠ')
    indp = argmin(abs.(log.(complex.(Fp.values)) .+ Complex(0, ns0.ω )))
    evp = Fp.vectors[:, indp]

    # normalize eigenvectors
    ev ./= sqrt(LA.dot(ev, ev))
    evp ./= LA.dot(ev, evp)

    δ1 = convert(𝒯, δ)
    δ2 = sqrt(δ1)
    δ3 = δ1^(1/3)
    d1Π(x,p,dx) = (Π(x .+ δ1 .* dx, p).u .- Π(x .- δ1 .* dx, p).u) ./ (2δ1)
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
    ev★ = nothing
    return NeimarkSackerPO(ns0.x0, T, ns0.p, ns0.ω, ev, ev★, ns, coll, true)
end

function neimark_sacker_normal_form_iooss(pbwrap::WrapPOColl,
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
    coll = get_discretization(pbwrap)
    N, m, Ntst = size(coll)
    par = ns.params
    T = getperiod(coll, ns.x0, par)
    𝒯 = eltype(coll)
    # identity matrix for collocation problem
    Icoll = I(coll, ns.x0, par)

    F(u, p) = residual(coll.prob_vf, u, p)
    A(u, p, du) = apply(jacobian(coll.prob_vf, u, p), du)
    B(u, p, du1, du2)      = BilinearMap( (dx1, dx2)      -> d2F(coll.prob_vf, u, p, dx1, dx2))(du1, du2)
    C(u, p, du1, du2, du3) = TrilinearMap((dx1, dx2, dx3) -> d3F(coll.prob_vf, u, p, dx1, dx2, dx3))(du1, du2, du3)

    _plot(x; k...) = (_sol = get_periodic_orbit(coll, x, 1);display(plot(_sol.t, _sol.u'; k...)))
    _rand(n, r = 2) = 𝒯(r) .* (rand(𝒯, n) .- 1//2)        # centered uniform random variables
    local ∫(u, v) = BifurcationKit.∫(coll, u, v, 1) # define integral with coll parameters

    #########
    # compute v1
    # we first compute the NS floquet eigenvector
    # we use an extended linear system for this
     # J = D  -  T*A(t) + iθ/T
    θ = abs(ns.ω)
    J = analytical_jacobian(coll, ns.x0, par; ρI = Complex(0, -θ/T), 𝒯 = Complex{𝒯})

    nj = size(J, 1)
    J[end, :] .= _rand(nj); J[:, end] .= _rand(nj)
    J[end, end] = 0

    rhs = zeros(𝒯, nj); rhs[end] = 1
    k = J  \ rhs; k = k[begin:end-1]; k ./= norm(k) # ≈ ker(J)
    l = J' \ rhs; l = l[begin:end-1]; l ./= norm(l)

    # update the borders to have less singular matrix J
    J[end, begin:end-1] .= k
    J[begin:end-1, end] .= l

    # Floquet eigenvectors
    vr = J  \ rhs
    v₁  = @view vr[begin:end-1]
    v₁ ./= sqrt(∫(vr, vr))
    v₁ₛ = get_time_slices(coll, vcat(v₁,1))
                if _NRMDEBUG; v₁ₛ .*= (-0.4238149014771724 - 0.32924318979676237im)/v₁ₛ[1,1]; end
    # re-scale the eigenvector
    v₁ₛ ./= sqrt(∫(v₁ₛ, v₁ₛ))
    v₁ = vec(v₁ₛ)
    @assert ∫(v₁ₛ, v₁ₛ) ≈ 1

    #########
    # compute ϕ1star
    # Jϕ = D  +  T * Aᵗ(t)
    Jϕ = analytical_jacobian(coll, ns.x0, par; _transpose = Val(true), ρF = -1)
    Jϕ[end-N:end-1, 1:N] .= -LA.I(N)
    Jϕ[end-N:end-1, end-N:end-1] .= LA.I(N)
    # build the extended linear problem
    Jϕ[end, :] .= _rand(nj)
    Jϕ[:, end] .= _rand(nj)
    Jϕ[end, end] = 0

    # update the borders to have less singular matrix Jψ
    k = Jϕ  \ rhs; k = k[begin:end-1]; k ./= norm(k)
    l = Jϕ' \ rhs; l = l[begin:end-1]; l ./= norm(l)
    Jϕ[end, begin:end-1] .= k
    Jϕ[begin:end-1, end] .= l

    ϕ₁★ = Jϕ \ rhs
    ϕ₁★ₛ = get_time_slices(coll, ϕ₁★)

    u₀ₛ = get_time_slices(coll, ns.x0) # periodic solution at bifurcation
    Fu₀ₛ = copy(u₀ₛ)
    Aₛ   = copy(v₁ₛ)
    Bₛ   = copy(v₁ₛ)
    Cₛ   = copy(v₁ₛ)
    for i = 1:size(u₀ₛ, 2)
      Fu₀ₛ[:, i] .= F(u₀ₛ[:, i], par)
        Bₛ[:, i] .= B(u₀ₛ[:, i], par, v₁ₛ[:, i], conj(v₁ₛ[:, i]))
    end

    #########
    # compute a₁
    ϕ₁★ ./= ∫( ϕ₁★ₛ, Fu₀ₛ)
    @assert ∫( ϕ₁★ₛ, Fu₀ₛ) ≈ 1
    # a = ∫ < ϕ₁★, B(v1, cv1) >
    a₁ = ∫(ϕ₁★ₛ, Bₛ)

    #########
    # compute v1star
    # J = D  +  T*Aᵗ(t) + iθ/T
    J = analytical_jacobian(coll, ns.x0, par; ρI = Complex(0, -θ/T), 𝒯 = Complex{𝒯}, _transpose = Val(true), ρF = -1)

    nj = size(J, 1)
    J[end, :] .= _rand(nj)
    J[:, end] .= _rand(nj)
    J[end, end] = 0

    rhs = zeros(𝒯, nj); rhs[end] = 1
    k = J  \ rhs; k = k[begin:end-1]; k ./= norm(k) # ≈ ker(J)
    l = J' \ rhs; l = l[begin:end-1]; l ./= norm(l)

    # update the borders to have less singular matrix J
    J[end, begin:end-1] .= k
    J[begin:end-1, end] .= l

    # left / right Floquet eigenvectors
    vr = J  \ rhs
    v₁★  = @view vr[begin:end-1]
    v₁★ₛ = get_time_slices(coll, vcat(v₁★, 1))
    v₁★ₛ ./= conj(∫(v₁★ₛ, v₁ₛ))
                if _NRMDEBUG; v₁★ₛ .*= (-1.0388609772214439 - 4.170067699081798im)/v₁★ₛ[1,1];end
    # re-scale the eigenvector
    v₁★ₛ ./= conj(∫(v₁★ₛ, v₁ₛ))
    v₁★ = vec(v₁★ₛ)
    @assert ∫(v₁★ₛ, v₁ₛ) ≈ 1
    #########
    # compute h20
    # solution of (D-T A(t) + 2iθ   )h = B(v1, v1)
    # written     (D-T(A(t) - 2iθ/T))h = B
    for i = 1:size(u₀ₛ, 2)
        Bₛ[:, i] .= B(u₀ₛ[:, i], par, v₁ₛ[:, i], v₁ₛ[:, i])
    end
    rhs = vcat(vec(Bₛ), 0)
    J = analytical_jacobian(coll, ns.x0, par; ρI = Complex(0,-2θ/T), 𝒯 = Complex{𝒯})
    # h₂₀ = J \ (rhs)

    h₂₀= J[begin:end-1,begin:end-1] \ rhs[begin:end-1];h₂₀ = vcat(vec(h₂₀), 0)
    # h₂₀ ./= 2Ntst # this seems necessary to have something comparable to ApproxFun
    h₂₀ = Icoll * h₂₀; @reset h₂₀[end] = 0
    h₂₀ₛ = get_time_slices(coll, h₂₀)
                # a cause de Icoll
                h₂₀ₛ[:, end] .= h₂₀ₛ[:,1]

                # _plot(real(vcat(vec(h₂₀ₛ),1)),label="h20")
                # _plot(imag(vcat(vec(Bₛ),1+im)),label="Bₛ")

    #########
    # compute h11
    # solution of (D-TA(t))h = B - a₁F
    for i = 1:size(u₀ₛ, 2)
        Bₛ[:, i] .= B(u₀ₛ[:, i], par, v₁ₛ[:, i], conj(v₁ₛ[:, i]))
    end
    rhsₛ = @. Bₛ - a₁ * Fu₀ₛ
    rhs = vcat(vec(rhsₛ), 0)
    border_ϕ1 = ForwardDiff.gradient(x -> ∫( reshape(x, size(ϕ₁★ₛ)), ϕ₁★ₛ),
                                                zeros(𝒯, length(ϕ₁★ₛ))
                                    )
    J = analytical_jacobian(coll, ns.x0, par;  𝒯 = Complex{𝒯})
    J[end-N:end-1, 1:N] .= -LA.I(N)
    J[end-N:end-1, end-N:end-1] .= LA.I(N)
    # add borders
    J[end, begin:end-1] .= border_ϕ1 # integral condition
    J[:, end] .= ϕ₁★
    J[end, end] = 0
    h₁₁ = J \ rhs
    h₁₁ ./= 2Ntst # this seems necessary to have something comparable to ApproxFun
    h₁₁ₛ = get_time_slices(coll, h₁₁)
                # _plot(real(vcat(vec(h₁₁ₛ),1)),label="h11")
    @debug "[NS-Iooss]" abs(∫( ϕ₁★ₛ, h₁₁ₛ))
    if abs(∫( ϕ₁★ₛ, h₁₁ₛ)) > 1e-10
        @warn "[NS-Iooss] The integral ∫(ϕ₁★ₛ, h₁₁ₛ) should be zero. We found $(∫( ϕ₁★ₛ, h₁₁ₛ ))"
    end
    if abs(h₁₁[end]) > 1e-10
        @warn "[NS-Iooss] The value h₁₁[end] should be zero. We found $(h₁₁[end])"
    end
    #########
    # compute d
    # d = <v1★, C(v,v,v)  +  2B(h11, v)  +  B(h20, cv)  +  C(v,v,cv)>/2 + ...
    for i = 1:size(u₀ₛ, 2)
        Bₛ[:, i] .= B(u₀ₛ[:, i], par, h₁₁ₛ[:, i], v₁ₛ[:, i])
        Cₛ[:, i] .= C(u₀ₛ[:, i], par,  v₁ₛ[:, i], v₁ₛ[:, i], conj(v₁ₛ[:, i]))
    end
                # _plot(real(vcat(vec(Bₛ),1)),label="B")
    d = (1/T) * ∫( v₁★ₛ, Cₛ ) + 2 * ∫( v₁★ₛ, Bₛ )
    @debug "[NS-Iooss] B(h11, v1)" d  (1/(2T)) * ∫( v₁★ₛ, Cₛ )     2*∫( v₁★ₛ, Bₛ )

    for i = 1:size(u₀ₛ, 2)
        Bₛ[:, i] .= B(u₀ₛ[:, i], par, h₂₀ₛ[:, i], conj(v₁ₛ[:, i]))
        Aₛ[:, i] .= A(u₀ₛ[:, i], par, v₁ₛ[:, i])
    end
    @debug "[NS-Iooss] B(h20, v1b)" d   ∫( v₁★ₛ, Bₛ )
    d +=  ∫( v₁★ₛ, Bₛ )
    d = d/2
    @debug "[NS-Iooss] A(h11, v1b)" -a₁/T * ∫( v₁★ₛ, Aₛ ) + im * θ * a₁/T^2   im * θ * a₁/T^2
    d += -a₁/T * ∫( v₁★ₛ, Aₛ ) + im * θ * a₁/T^2

    nf = (a = a₁, d, h₁₁ₛ, ϕ₁★ₛ, v₁★ₛ, h₂₀ₛ, _NRMDEBUG) # keep b3 for ns-codim 2
    ns_new = (@set ns.nf = nf)
    @reset ns_new.type = real(d) < 0 ? :SuperCritical : :SubCritical
    return NeimarkSackerPO(ns.x0, T, ns.p, θ, v₁, v₁★, ns_new, coll, false)
end

function neimark_sacker_normal_form(pbwrap::WrapPOSh{ <: ShootingProblem },
                                br::AbstractBranchResult,
                                ind_bif::Int,
                                Teigvec::Type{𝒯eigvec} = _getvectortype(br);
                                nev = length(eigenvalsfrombif(br, ind_bif)),
                                verbose = false,
                                lens = getlens(br),
                                detailed::Val{detailed_type} = Val(true),
                                kwargs_nf...) where {detailed_type, 𝒯eigvec}

    # first, get the bifurcation point parameters
    sh = get_discretization(pbwrap)
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

    if ~detailed_type
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
    sh = get_discretization(pbwrap)
    pars = ns0.params
    period = getperiod(sh, ns0.x0, pars)
    # compute the Poincaré return map, the section is on the first time slice
    Π = PoincareMap(pbwrap, ns0.x0, pars, optn)
    xₛ = get_time_slices(sh, Π.po)[:, 1]

    _nrm = norminf(Π(xₛ, pars).u - xₛ)
    _nrm > 1e-12 && @warn "[NS normal form PRM], residual = $_nrm"

    dΠ = jacobian(Π, xₛ, pars)
    F  = LA.eigen(dΠ)

    ind = argmin(abs.(log.(complex.(F.values)) .- Complex(0, ns0.ω )))
    ev = F.vectors[:, ind]
    Fp = LA.eigen(dΠ')
    indp = argmin(abs.(log.(complex.(Fp.values)) .+ Complex(0, ns0.ω )))
    evp = Fp.vectors[:, indp]

    # normalize eigenvectors
    ev ./= sqrt(LA.dot(ev, ev))
    evp ./= LA.dot(evp, ev)

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
"""
$(TYPEDSIGNATURES)

Compute the predictor for the period-doubling bifurcation of periodic orbit.
"""
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitTrapProblem},
                    δp,
                    ampfactor;
                    override = false)
    pb = nf.prob
    M, N = size(pb)
    # we update the problem by doubling M
    if pb.mesh isa TimeMesh{Int64}
        # we double the mesh size
        pbnew = @set pb.mesh.ds = 2M
    else
        oldmesh = get_times(pb)
        new_mesh = vcat(old_mesh[begin:end-1] ./2, old_mesh ./2 .+ 1/2)
        pbnew = @set pb.mesh.ds = new_mesh
    end
    @reset pbnew.M = 2M

    orbitguess0c = get_time_slices(pb, nf.po)
    ζc = reshape(nf.ζ, N, M)
    orbitguess_c = @. orbitguess0c + ampfactor * ζc
    orbitguess_c = hcat(orbitguess_c[:,begin:end-1], orbitguess0c .- ampfactor .*  ζc, orbitguess_c[:,1])
    # orbitguess_c = hcat(orbitguess_c, orbitguess0c .- ampfactor .*  ζc)
    # we append twice the period
    orbitguess = vcat(vec(orbitguess_c), 2nf.T)
    # we update the phase condition
    @reset pbnew.xπ = orbitguess[begin:end-1]
    @reset pbnew.ϕ = circshift(orbitguess[begin:end-1], length(orbitguess))
    # we need to duplicate the po as well in case deflation is used
    po0 = get_time_slices(pb, nf.po)
    po = vcat(vec(hcat(po0, po0)), nf.T) 
    return (;orbitguess, pnew = nf.nf.p + δp, prob = pbnew, ampfactor, po, ζc)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the simple branch point of periodic orbit.
"""
function predictor(nf::BranchPointPO{ <: PeriodicOrbitTrapProblem},
                    δp,
                    ampfactor;
                    override = false)
    prob = nf.prob
    M, N = size(prob)
    orbitguess = copy(nf.po)
    orbitguess[begin:end-1] .+= ampfactor .* nf.ζ
    # we update the phase condition
    pbnew = deepcopy(prob)
    pars = set(getparams(prob), getlens(prob), nf.nf.p)
    updatesection!(pbnew, nf.po, pars)
    return (;orbitguess, pnew = nf.nf.p + δp, prob = pbnew, ampfactor, po = nf.po, ζc = reshape(nf.ζ, N, M))
end

function predictor(nf::NeimarkSackerPO,
                    δp,
                    ampfactor;
                    override = false)
    orbitguess = copy(nf.po)
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor, po = nf.po)
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Compute the predictor for the period-doubling bifurcation of periodic orbit.
"""
function predictor(nf::PeriodDoublingPO{ <: PeriodicOrbitOCollProblem }, 
                    δp, 
                    ampfactor; 
                    override = false)
    pbnew = deepcopy(nf.prob)
    N, m, Ntst = size(nf.prob)
    orbitguess0 = _getsolution(nf.po)[begin:end-1]

    # we update the problem by doubling Ntst
    # we need to save the mesh for adaptation
    old_mesh = getmesh(pbnew)
    new_mesh = vcat(old_mesh[begin:end-1]/2, old_mesh ./2 .+ 1/2)
    pbnew = set_collocation_size(pbnew, 2Ntst, m)
    update_mesh!(pbnew, new_mesh)

    # parameter to scale time
    time_factor = 1

    if ~override # we use predictor from normal form
        if nf.prm == true && ~isnothing(nf.nf.nf)
            # normal form based on Poincaré return map
            pred = predictor(nf.nf, δp)
            ampfactor *= pred.x1
            δp = pred.δp
        elseif nf.prm == false && get(nf.nf.nf, :c₁₁, nothing) != nothing
            # Iooss normal form
            (;c₁₁, b3) = nf.nf.nf
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

    orbitguess_c = @. orbitguess0 + ampfactor * nf.ζ
    orbitguess = vcat(orbitguess_c[begin:end-N], orbitguess0 .- ampfactor .* nf.ζ)

    pbnew.xπ .= orbitguess
    ϕ = circshift(orbitguess, length(orbitguess))
    updatesection!(pbnew, ϕ, nothing)

    # we append the doubled period
    orbitguess = vcat(orbitguess, 2nf.T * time_factor)

    # we double po in case of use of deflated newton
    po = vcat(orbitguess0[begin:end-N], orbitguess0, nf.T)

    # no need to change pbnew.cache
    return (;orbitguess, pnew = nf.nf.p + δp, prob = pbnew, ampfactor, δp, time_factor, po)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the simple branch point of periodic orbit.
"""
function predictor(nf::BranchPointPO{ <: PeriodicOrbitOCollProblem},
                    δp,
                    ampfactor;
                    override = false)
    pbnew = deepcopy(nf.prob)
    N, m, Ntst = size(nf.prob)
    orbitguess0 = _getsolution(nf.po)[begin:end-1]

    # we update the problem by doubling Ntst
    # we need to save the mesh for adaptation
    old_mesh = getmesh(pbnew)
    new_mesh = vcat(old_mesh[begin:end-1]/2, old_mesh ./2 .+ 1/2)
    pbnew = set_collocation_size(pbnew, 2Ntst, m)
    update_mesh!(pbnew, new_mesh)

    if ~override # we use predictor from normal form
        if nf.prm == true && ~isnothing(nf.nf.nf)
            # normal form based on Poincaré return map
            pred = predictor(nf.nf, δp)
            ampfactor *= pred.amp
            δp = pred.δp
        elseif nf.prm == false 
            error("Not available yet!! WIP !!")
        end
    end

    orbitguess = copy(nf.po)
    orbitguess[begin:end-1] .+= ampfactor .* nf.ζ[2]
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor, po = nf.po)
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Compute the predictor for the period-doubling bifurcation of periodic orbit.
"""
function predictor(nf::PeriodDoublingPO{ <: ShootingProblem },
                    δp,
                    ampfactor;
                    override = false)
    if ~isnothing(nf.nf.nf) && ~override
        pred = predictor(nf.nf, δp)
        ampfactor = pred.x1 * ampfactor
        δp = pred.δp
    end

    pbnew = deepcopy(nf.prob)
    pnew = nf.nf.p + δp
    ζs = nf.ζ .* ampfactor
    orbitguess = copy(nf.po)[begin:end-1] .+ ζs
    orbitguess = vcat(orbitguess, copy(nf.po)[begin:end-1] .- ζs, 2nf.po[end])

    @reset pbnew.M = 2nf.prob.M
    @reset pbnew.ds = _duplicate(pbnew.ds) ./ 2
    updatesection!(pbnew, orbitguess, setparam(pbnew, pnew))

    po0 = copy(nf.po)[begin:end-1]
    po = vcat(po0, po0, nf.po[end])
    return (;orbitguess, pnew, prob = pbnew, ampfactor, δp, po)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the simple branch point of periodic orbit.
"""
function predictor(nf::BranchPointPO{ <: ShootingProblem },
                    δp,
                    ampfactor;
                    override = false)
    if ~isnothing(nf.nf.nf) && ~override
        pred = predictor(nf.nf, δp)
        ampfactor = pred.amp * ampfactor
        δp = pred.δp
    end

    pbnew = deepcopy(nf.prob)
    pnew = nf.nf.p + δp
    ζs = nf.ζ .* ampfactor
    orbitguess = copy(nf.po)
    orbitguess[eachindex(ζs)] .+= ζs
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor, po = nf.po)
end
####################################################################################################
"""
$(TYPEDSIGNATURES)

Compute the predictor for the period-doubling bifurcation of periodic orbit.
"""
function predictor(nf::PeriodDoublingPO{ <: PoincareShootingProblem }, 
                    δp, 
                    ampfactor;
                    override = false)
    pbnew = deepcopy(nf.prob)
    ζs = nf.ζ

    @reset pbnew.section = _duplicate(pbnew.section)
    @reset pbnew.M = pbnew.section.M
    orbitguess = copy(nf.po) .+ ampfactor .* ζs
    orbitguess = vcat(orbitguess, orbitguess .- ampfactor .* ζs)
    po = vcat(nf.po, nf.po)
    return (;orbitguess, pnew = nf.nf.p + δp, prob = pbnew, ampfactor, po)
end

"""
$(TYPEDSIGNATURES)

Compute the predictor for the simple branch point of periodic orbit.
"""
function predictor(nf::BranchPointPO{ <: PoincareShootingProblem},
                    δp,
                    ampfactor;
                    override = false)
    ζs = nf.ζ
    orbitguess = nf.po .+ ampfactor .* ζs
    return (;orbitguess, pnew = nf.nf.p + δp, prob = nf.prob, ampfactor, po = nf.po)
end
