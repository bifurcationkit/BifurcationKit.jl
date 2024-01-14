function get_adjoint_basis(L★, λs, eigsolver; nev = 3, verbose = false)
    # we compute the eigen-elements of the adjoint of L
    λ★, ev★, cv, = eigsolver(L★, nev)
    ~cv && @warn "Eigen Solver did not converge"
    verbose && Base.display(λ★)
    # vectors to hold eigen-elements for the adjoint of L
    λ★s = Vector{eltype(λs)}()
    # TODO This is a horrible hack to get the type of the left eigenvectors
    ζ★s = Vector{typeof(geteigenvector(eigsolver, ev★, 1))}()

    for (idvp, λ) in enumerate(λs)
        I = argmin(abs.(λ★ .- λ))
        abs(real(λ★[I])) > 1e-2 && @warn "Did not converge to the requested eigenvalues. We found $(real(λ★[I])) !≈ 0. This might not lead to precise normal form computation. You can perhaps increase the argument `nev`."
        verbose && println("──> VP[$idvp] paired with VP★[$I]")
        ζ★ = geteigenvector(eigsolver, ev★, I)
        push!(ζ★s, copy(ζ★))
        push!(λ★s, λ★[I])
        # we change λ★ so that it is not used twice
        λ★[I] = 1e9
    end
    return ζ★s, λ★s
end

"""
$(SIGNATURES)

Return a left eigenvector for an eigenvalue closest to λ. `nev` indicates how many eigenvalues must be computed by the eigensolver. Indeed, for iterative solvers, it may be needed to compute more eigenvalues than necessary.
"""
function get_adjoint_basis(L★, λ::Number, eigsolver; nev = 3, verbose = false)
    λ★, ev★, cv, = eigsolver(L★, nev)
    ~cv && @warn "Eigen Solver did not converge"
    I = argmin(abs.(λ★ .- λ))
    verbose && (println("┌── left eigenvalues = "); display(λ★))
    verbose && println( "├── right eigenvalue = ", λ, 
                      "\n└──  left eigenvalue = ", λ★[I])
    abs(real(λ★[I])) > 1e-2 && @warn "The bifurcating eigenvalue is not that close to Re = 0. We found $(real(λ★[I])) !≈ 0.  You can perhaps increase the argument `nev`."
    ζ★ = geteigenvector(eigsolver, ev★, I)
    return copy(ζ★), λ★[I]
end
####################################################################################################
"""
$(SIGNATURES)

Compute a normal form based on Golubitsky, Martin, David G Schaeffer, and Ian Stewart. Singularities and Groups in Bifurcation Theory. New York: Springer-Verlag, 1985, VI.1.d page 295.
"""
function get_normal_form1d(prob::AbstractBifurcationProblem,
                    br::ContResult,
                    ind_bif::Int;
                    nev = length(eigenvalsfrombif(br, ind_bif)),
                    verbose = false,
                    lens = getlens(br),
                    Teigvec = vectortype(br),
                    tol_fold = 1e-3,
                    scaleζ = norm,
                    autodiff = false)
    bifpt = br.specialpoint[ind_bif]
    τ = bifpt.τ 
    @assert bifpt.type == :bp "The provided index does not refer to a Branch Point with 1d kernel. The type of the bifurcation is $(bifpt.type). The bifurcation point is $bifpt."
    @assert abs(bifpt.δ[1]) == 1 "We only provide normal form computation for simple bifurcation points e.g when the kernel of the jacobian is 1d. Here, the dimension of the kernel is $(abs(bifpt.δ[1]))."

    verbose && println("━"^53*"\n┌─ Normal form Computation for 1d kernel")
    verbose && println("├─ analyse bifurcation at p = ", bifpt.param)

    options = br.contparams.newton_options

    # we need this conversion when running on GPU and loading the branch from the disk
    x0 = convert(Teigvec, bifpt.x)
    p = bifpt.param

    # parameter for vector field
    parbif = set(getparams(br), lens, p)

    # jacobian at bifurcation point
    L = jacobian(prob, x0, parbif)

    # linear solver
    ls = options.linsolver

    # "zero" eigenvalue at bifurcation point, it must be real
    λ = real(br.eig[bifpt.idx].eigenvals[bifpt.ind_ev])
    verbose && println("├─ smallest eigenvalue at bifurcation = ", λ)

    # corresponding eigenvector, it must be real
    if haseigenvector(br) == false
        # we recompute the eigen-elements if there were not saved during the computation of the branch
        verbose && @info "Eigen-elements not saved in the branch. Recomputing them..."
        _λ, _ev, _ = options.eigsolver(L, bifpt.ind_ev + 2)
        @assert _λ[bifpt.ind_ev] ≈ λ "We did not find the correct eigenvalue $λ. We found $(_λ)"
        ζ = real.(geteigenvector(options.eigsolver, _ev, bifpt.ind_ev))
    else
        ζ = real.(geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev))
    end
    ζ ./= scaleζ(ζ)

    # extract eigen-elements for adjoint(L), needed to build spectral projector
    if is_symmetric(prob)
        λ★ = br.eig[bifpt.idx].eigenvals[bifpt.ind_ev]
        ζ★ = copy(ζ)
    else
        _Jt = has_adjoint(prob) ? jad(prob, x0, parbif) : adjoint(L)
        ζ★, λ★ = get_adjoint_basis(_Jt, conj(λ), options.eigsolver; nev = nev, verbose = verbose)
    end

    ζ★ = real.(ζ★); λ★ = real.(λ★)

    @assert abs(dot(ζ, ζ★)) > 1e-10 "We got ζ⋅ζ★ = $((dot(ζ, ζ★))). This dot product should not be zero. Perhaps, you can increase `nev` which is currently $nev."
    ζ★ ./= dot(ζ, ζ★)

    # differentials and projector on Range(L), there are real valued
    R2(dx1, dx2)      = d2F(prob, x0, parbif, dx1, dx2)
    R3(dx1, dx2, dx3) = d3F(prob, x0, parbif, dx1, dx2, dx3)
    E(x) = x .- dot(x, ζ★) .* ζ

    # we compute the reduced equation: a⋅(p - pbif) + x⋅(b1⋅(p - pbif) + b2⋅x/2 + b3⋅x^2/6)
    # coefficient of p
    δ = getdelta(prob)
    if autodiff
        R01 = ForwardDiff.derivative(z->residual(prob, x0, set(parbif, lens, z)),p)
    else
        R01 = (residual(prob, x0, set(parbif, lens, p + δ)) .- 
               residual(prob, x0, set(parbif, lens, p - δ))) ./ (2δ)
    end
    a = dot(R01, ζ★)
    Ψ01, cv, it = ls(L, E(R01))
    ~cv && @debug "[Normal form Ψ01] Linear solver for J did not converge. it = $it"
    verbose && println("┌── Normal form:   aδμ + b1⋅x⋅δμ + b2⋅x²/2 + b3⋅x³/6")
    verbose && println("├─── a    = ", a)

    # coefficient of x*p
    if autodiff
        R11 = ForwardDiff.derivative(z-> apply(jacobian(prob, x0, set(parbif, lens, z)), ζ), p)
    else
        R11 = (apply(jacobian(prob, x0, set(parbif, lens, p + δ)), ζ) - 
               apply(jacobian(prob, x0, set(parbif, lens, p - δ)), ζ)) ./ (2δ)
    end

    b1 = dot(R11 .- R2(ζ, Ψ01), ζ★)
    verbose && println("├─── b1   = ", b1)

    # coefficient of x^2
    b2v = R2(ζ, ζ)
    b2 = dot(b2v, ζ★)
    verbose && println("├─── b2/2 = ", b2/2)

    # coefficient of x^3, recall b2v = R2(ζ, ζ)
    wst, cv, it = ls(L, E(b2v)) # Golub. Schaeffer Vol 1 page 33, eq 3.22
    ~cv && @debug "[Normal form wst] Linear solver for J did not converge. it = $it"
    b3v = R3(ζ, ζ, ζ) .- 3 .* R2(ζ, wst)
    b3 = dot(b3v, ζ★)
    verbose && println("└─── b3/6 = ", b3/6)

    bp = (x0, τ, p, parbif, lens, ζ, ζ★, (;a , b1, b2, b3, Ψ01, wst), :NA)
    if abs(a) < tol_fold
        return 100abs(b2/2) < abs(b3/6) ? Pitchfork(bp[1:end-1]...) : Transcritical(bp...)
    else
        return Fold(bp...)
    end
    # we should never hit this
    return nothing
end

get_normal_form1d(br::Branch, ind_bif::Int; kwargs...) = get_normal_form1d(get_contresult(br), ind_bif; kwargs...)

get_normal_form1d(br::ContResult, ind_bif::Int; kwargs...) = get_normal_form1d(br.prob, br, ind_bif; kwargs...)
"""
$(SIGNATURES)

This function provides prediction for the zeros of the Transcritical bifurcation point.

# Arguments
- `bp::Transcritical` the bifurcation point
- `ds` at with distance relative to the bifurcation point do you want the prediction. Can be negative. Basically the parameter is `p = bp.p + ds`

# Optional arguments
- `verbose` display information
- `ampfactor = 1` factor multiplying prediction
"""
function predictor(bp::Transcritical, ds::T; verbose = false, ampfactor = T(1)) where T
    # this is the predictor from a Transcritical bifurcation.
    # After computing the normal form, we have an issue.
    # We need to determine if the already computed branch corresponds to the solution x = 0 of the normal form.
    # This leads to the two cases below.
    nf = bp.nf
    τ = bp.τ
    a, b1, b2, b3, Ψ01 = nf
    pnew = bp.p + ds
    # we solve b1 * ds + b2 * amp / 2 = 0
    amp = -2ds * b1 / b2 * ampfactor
    dsfactor = T(1)

    # x0  next point on the branch
    # x1  next point on the bifurcated branch
    # xm1 previous point on bifurcated branch 
    if norm(τ.u)>0 && abs(dot(bp.ζ, τ.u)) >= 0.9*norm(τ.u)
        @debug "Constant predictor in Transcritical"
        x1  = bp.x0 .- ds .* Ψ01 # we put minus, because Ψ01  = L \ R01 and GS Vol 1 uses w = -L\R01
        xm1 = bp.x0
        x0  = bp.x0 .+ ds/τ.p .* τ.u
    else
        x0  = bp.x0
        x1  = bp.x0 .+ amp .* real.(bp.ζ) .- ds .* Ψ01
        xm1 = bp.x0 .- amp .* real.(bp.ζ) .+ ds .* Ψ01
    end

    verbose && println("──> Prediction from Normal form, δp = $(pnew - bp.p), amp = $amp")
    return (x0 = x0, x1 = x1, xm1 = xm1, p = pnew, pm1 = bp.p - ds, dsfactor = dsfactor, amp = amp)
end

"""
$(SIGNATURES)

This function provides prediction for the zeros of the Pitchfork bifurcation point.

# Arguments
- `bp::Pitchfork` the bifurcation point
- `ds` at with distance relative to the bifurcation point do you want the prediction. Based on the criticality of the Picthfork, its sign is enforced no matter what you pass. Basically the parameter is `bp.p + abs(ds) * dsfactor` where `dsfactor = ±1` depending on the criticality.

# Optional arguments
- `verbose`    display information
- `ampfactor = 1` factor multiplying prediction
"""
function predictor(bp::Pitchfork, ds::T; verbose = false, ampfactor = T(1)) where T
    nf = bp.nf
    a, b1, b2, b3 = nf

    # we need to find the type, supercritical or subcritical
    dsfactor = b1 * b3 < 0 ? T(1) : T(-1)
    if 1==1
        # we solve b1 * ds + b3 * amp^2 / 6 = 0
        amp = ampfactor * sqrt(-6abs(ds) * dsfactor * b1 / b3)
        pnew = bp.p + abs(ds) * dsfactor
    # else
    #     # we solve b1 * ds + b3 * amp^2 / 6 = 0
    #     amp = ampfactor * abs(ds)
    #     pnew = bp.p + dsfactor * ds^2 * abs(b3/b1/6)
    end
    verbose && println("──> Prediction from Normal form, δp = $(pnew - bp.p), amp = $amp")
    return (x0 = bp.x0, x1 = bp.x0 .+ amp .* real.(bp.ζ), p = pnew, dsfactor = dsfactor, amp = amp)
end

function predictor(bp::Fold, ds::T; verbose = false, ampfactor = T(1)) where T
    @debug "It seems the point is a Saddle-Node bifurcation.\nThe normal form is aδμ + b1⋅x + b2⋅x² + b3⋅x³\n with coefficients \n a = $(bp.nf.a), b1 = $(bp.nf.b1), b2 = $(bp.nf.b2), b3 = $(bp.nf.b3)."
    return nothing
end
####################################################################################################
function factor3d(i,j,k)
    if i == j == k
        return 1/6
    else
        _power = length(unique((i,j,k)))
        if _power == 1
            factor = 1/6 /2
        elseif _power == 2
            factor = 1/2 / 3
        else
            factor = 1.0
        end
        return factor
    end
end

function (bp::NdBranchPoint)(::Val{:reducedForm}, x, p::T) where T
    # formula from https://fr.qwe.wiki/wiki/Taylor's_theorem
    # dimension of the kernel
    N = length(bp.ζ)
    @assert N == length(x)
    out = zero(x)
    # normal form
    nf = bp.nf
    # coefficient p
    out .= p .* nf.a

    # factor to account for factorials
    factor = one(T)

    @inbounds for ii in 1:N
        factor = one(T)
        out[ii] = 0
        # coefficient x*p
        for jj in 1:N
            # coefficient x*p
            out[ii] += p * nf.b1[ii , jj] * x[jj]
            for kk in 1:N
                # coefficients of x^2
                factor = jj == kk ? 1/2 : 1
                out[ii] += nf.b2[ii, jj, kk] * x[jj] * x[kk] * factor / 2

                for ll in 1:N
                    # coefficients of x^3
                    factor = factor3d(ii, jj, kk)
                    out[ii] += nf.b3[ii, jj, kk, ll] * x[jj] * x[kk]  * x[ll] * factor
                end
            end
        end
    end
    return out
end

function (bp::NdBranchPoint)(x, δp::Real)
    out = bp.x0 .+ x[1] .* bp.ζ[1]
    for ii in 2:length(x)
        out .+= x[ii] .* bp.ζ[ii]
    end
    return out
end

"""
$(SIGNATURES)

Print the normal form `bp` with a nice string.
"""
function nf(bp::NdBranchPoint; tol = 1e-6, digits = 4)
    superDigits = [c for c in "⁰ ²³⁴⁵⁶⁷⁸⁹"]

    nf = bp.nf
    N = length(nf.a)
    out = ["" for _ in 1:N]

    for ii = 1:N
        if abs(nf.a[ii]) > tol
            out[ii] *= "$(round(nf.a[ii],digits=digits)) ⋅ p"
        end
        for jj in 1:N
            coeff = round(nf.b1[ii,jj],digits=digits)
            if abs(coeff) > tol
                out[ii] *= " + $coeff * x$jj ⋅ p"
            end
        end

        for jj in 1:N
            for kk in jj:N
                coeff = round(nf.b2[ii,jj,kk] / 2,digits=digits)
                if abs(coeff) > tol
                    if jj == kk
                        out[ii] *= " + $coeff ⋅ x$(jj)²"
                    else
                        out[ii] *= " + $(round(2coeff,digits=digits)) ⋅ x$jj ⋅ x$kk"
                    end
                end

                for ll in kk:N
                    coeff = round(nf.b3[ii,jj,kk,ll] / 6,digits=digits)
                    _pow = zeros(Int64,N)
                    _pow[jj] += 1;_pow[kk] += 1;_pow[ll] += 1;

                    if abs(coeff) > tol
                        if jj == kk == ll
                            out[ii] *= " + $coeff"
                        else
                            out[ii] *= " + $(round(3coeff,digits=digits))"
                        end
                        for mm in 1:N
                            if _pow[mm] > 1
                                out[ii] *= " ⋅ x$mm" * (superDigits[_pow[mm]+1])
                            elseif _pow[mm] == 1
                                out[ii] *= " ⋅ x$mm"
                            end
                        end
                    end
                end
            end
        end
    end
    return out
end

"""
$(SIGNATURES)

Bi-orthogonalise the two sets of vectors.
"""
function biorthogonalise(ζs, ζ★s, verbose)
    # change only the ζ★s to have bi-orthogonal left/right eigenvectors
    # we could use projector P=A(A^{T}A)^{-1}A^{T}
    # we use Gram-Schmidt algorithm instead
    G = [ dot(ζ, ζ★) for ζ in ζs, ζ★ in ζ★s]
    @assert abs(det(G)) > 1e-14 "The Gram matrix is not invertible! det(G) = $(det(G)), G = \n$G $(display(G))"

    # save those in case the first algo fails
    _ζs = deepcopy(ζs)
    _ζ★s = deepcopy(ζ★s)

    # first algo
    tmp = copy(ζ★s[1])
    for ii in eachindex(ζ★s)
        tmp .= ζ★s[ii]
        for jj in eachindex(ζs)
            if ii != jj
                tmp .-= dot(tmp, ζs[jj]) .* ζs[jj] ./ dot(ζs[jj], ζs[jj])
            end
        end
        ζ★s[ii] .= tmp ./ dot(tmp, ζs[ii])
    end

    G = [ dot(ζ, ζ★) for ζ in ζs, ζ★ in ζ★s]

    # we switch to another algo if the above fails
    if norm(G - LinearAlgebra.I, Inf) >= 1e-5
        @warn "Gram matrix not equal to identity. Switching to LU algorithm."
        println("G (det = $(det(G))) = "); display(G)
        G = [ dot(ζ, ζ★) for ζ in _ζs, ζ★ in _ζ★s]
        _F = lu(G; check = true)
        display(inv(_F.L) * inv(_F.P) * G * inv(_F.U))
        ζs = inv(_F.L) * inv(_F.P) * _ζs
        ζ★s = inv(_F.U)' * _ζ★s
    end

    # test the bi-orthogonalization
    G = [ dot(ζ, ζ★) for ζ in ζs, ζ★ in ζ★s]
    verbose && (printstyled(color=:green, "──> Gram matrix = \n"); Base.display(G))
    @assert norm(G - LinearAlgebra.I, Inf) < 1e-5 "Failure in bi-orthogonalisation of the right / left eigenvectors. The left eigenvectors do not form a basis. You may want to increase `nev`, G = \n $(display(G))"
    return ζs, ζ★s
end

"""
$(SIGNATURES)

Compute the normal form of the bifurcation point located at `br.specialpoint[ind_bif]`.

# Arguments
- `prob::AbstractBifurcationProblem`
- `br` result from a call to [`continuation`](@ref)
- `ind_bif` index of the bifurcation point in `br.specialpoint`

# Optional arguments
- `nev` number of eigenvalues used to compute the spectral projection. This number has to be adjusted when used with iterative methods.
- `verbose` whether to display information
- `ζs` list of vectors spanning the kernel of `dF` at the bifurcation point. Useful to enforce the basis for the normal form.
- `lens::Lens` specify which parameter to take the partial derivative ∂pF
- `scaleζ` function to normalise the kernel basis. Indeed, when used with large vectors and `norm`, it results in ζs and the normal form coefficient being super small.
- `autodiff = true` whether to use ForwardDiff for the differentiations w.r.t the parameters that are required to compute the normal form. Used for example for Bogdanov-Takens point. You can set to `autodiff = false` if you wish.
- `detailed = true` whether to compute only a simplified normal form. Used for example for Bogdanov-Takens point.
- `bls = MatrixBLS()` specify Bordered linear solver. Used for example for Bogdanov-Takens point.

# Available method

You can directly call 

    get_normal_form(br, ind_bif ; kwargs...)

which is a shortcut for `get_normal_form(getprob(br), br, ind_bif ; kwargs...)`.

Once the normal form `nf` has been computed, you can call `predictor(nf, δp)` to obtain an estimate of the bifurcating branch.

"""
function get_normal_form(prob::AbstractBifurcationProblem,
                        br::ContResult,
                        id_bif::Int ;
                        nev = length(eigenvalsfrombif(br, id_bif)),
                        verbose = false,
                        ζs = nothing,
                        ζs_ad = nothing,
                        lens = getlens(br),
                        Teigvec = getvectortype(br),
                        scaleζ = norm,
                        detailed = true,
                        autodiff = false,
                        bls = MatrixBLS(),
                        bls_adjoint = bls,
                        bls_block = bls,
                        )
    bifpt = br.specialpoint[id_bif]

    @assert !(bifpt.type in (:endpoint,)) "Normal form for $(bifpt.type) not implemented"

    # parameters for normal form
    kwargs_nf = (nev = nev, verbose = verbose, lens = lens, Teigvec = Teigvec, scaleζ = scaleζ)

    if bifpt.type == :hopf
        return hopf_normal_form(prob, br, id_bif; kwargs_nf..., detailed)
    elseif bifpt.type == :cusp
        return cusp_normal_form(prob, br, id_bif; kwargs_nf...)
    elseif bifpt.type == :bt
        return bogdanov_takens_normal_form(prob, br, id_bif; kwargs_nf..., detailed, autodiff, bls, bls_adjoint = bls_adjoint, bls_block = bls_block, ζs, ζs_ad)
    elseif bifpt.type == :gh
        return bautin_normal_form(prob, br, id_bif; kwargs_nf..., detailed)
    elseif bifpt.type == :zh
        return zero_hopf_normal_form(prob, br, id_bif; kwargs_nf..., detailed, autodiff)
    elseif bifpt.type == :hh
        return hopf_hopf_normal_form(prob, br, id_bif; kwargs_nf..., detailed, autodiff)
    elseif abs(bifpt.δ[1]) == 1 # simple branch point
        return get_normal_form1d(prob, br, id_bif ; autodiff, kwargs_nf...)
    end

    τ = bifpt.τ
    prob_vf = prob

    # kernel dimension:
    N = abs(bifpt.δ[1])

    # in case nev = 0 (number of unstable eigenvalues), we increase nev to avoid bug
    nev = max(2N, nev)
    verbose && println("━"^53*"\n──> Normal form Computation for a $N-d kernel")
    verbose && println("──> analyse bifurcation at p = ", bifpt.param)

    options = br.contparams.newton_options
    ls = options.linsolver

    # bifurcation point
    if ~(bifpt.x isa Teigvec)
        @error "The type of the equilibrium $(typeof(bifpt.x)) does not match the one of the eigenvectors $(Teigvec). You can keep your choice by using the option `Teigvec` in `get_normal_form` to specify the type of the equilibrum."
    end
    x0 = convert(Teigvec, bifpt.x)
    p = bifpt.param

    # parameter for vector field
    parbif = setparam(br, p)

    # jacobian at bifurcation point
    L = jacobian(prob_vf, x0, parbif)

    # we invert L repeatedly, so we try to factorize it
    Linv = L isa AbstractMatrix ? factorize(L) : L

    # "zero" eigenvalues at bifurcation point
    rightEv = br.eig[bifpt.idx].eigenvals
    indev = br.specialpoint[id_bif].ind_ev
    λs = rightEv[indev-N+1:indev]
    verbose && println("──> smallest eigenvalues at bifurcation = ", real.(λs))

    # and corresponding eigenvectors
    if isnothing(ζs) # do we have a basis for the kernel?
        if haseigenvector(br) == false # are the eigenvector saved in the branch?
            @info "No eigenvector recorded, computing them on the fly"
            # we recompute the eigen-elements if there were not saved during the computation of the branch
            _λ, _ev, _ = options.eigsolver(L, length(rightEv))
            verbose && (println("──> (λs, λs (recomputed)) = "); display(hcat(rightEv, _λ[eachindex(rightEv)])))
            if norm(_λ[eachindex(rightEv)] - rightEv, Inf) > br.contparams.tol_stability
                @warn "We did not find the correct eigenvalues (see 1st col). We found the eigenvalues displayed in the second column:\n $(display(hcat(rightEv, _λ[eachindex(rightEv)]))).\n Difference between the eigenvalues:" display(_λ[eachindex(rightEv)] - rightEv)
            end
            ζs = [copy(geteigenvector(options.eigsolver, _ev, ii)) for ii in indev-N+1:indev]
        else
            ζs = [copy(geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvecs, ii)) for ii in indev-N+1:indev]
        end
    end

    # extract eigen-elements for transpose(L), needed to build spectral projector
    # it is OK to re-scale at this stage as the basis ζs is not touched anymore, we
    # only adjust ζ★s
    for ζ in ζs; ζ ./= scaleζ(ζ); end
    if is_symmetric(prob)
        λ★s = copy(λs)
        ζ★s = copy.(ζs)
    else
        _Jt = has_adjoint(prob_vf) ? jad(prob_vf, x0, parbif) : transpose(L)
        ζ★s, λ★s = get_adjoint_basis(_Jt, conj.(λs), options.eigsolver; nev = nev, verbose = verbose)
    end
    ζ★s = real.(ζ★s); λ★s = real.(λ★s)
    ζs = real.(ζs); λs = real.(λs)
    verbose && println("──> VP     = ", λs, "\n──> VP★ = ", λ★s)

    ζs, ζ★s = biorthogonalise(ζs, ζ★s, verbose)

    # these differentials should as is work as we are using real valued vectors
    R2(dx1, dx2) = d2F(prob_vf, x0, parbif, dx1, dx2)
    R3(dx1, dx2, dx3) = d3F(prob_vf, x0, parbif, dx1, dx2, dx3)

    # projector on Range(L)
    function E(x)
        out = copy(x)
        for ii in 1:N
            out .= out .- dot(x, ζ★s[ii]) .* ζs[ii]
        end
        out
    end

    # vector eltype
    Tvec = eltype(ζs[1])

    # coefficients of p
    dgidp = Vector{Tvec}(undef, N)
    δ = getdelta(prob)
    R01 = (residual(prob_vf, x0, set(parbif, lens, p + δ)) .- residual(prob_vf, x0, set(parbif, lens, p - δ))) ./ (2δ)
    for ii in 1:N
        dgidp[ii] = dot(R01, ζ★s[ii])
    end
    verbose && printstyled(color=:green,"──> a (∂/∂p) = ", dgidp, "\n")

    # coefficients of x*p
    d2gidxjdpk = zeros(Tvec, N, N)
    for ii in 1:N, jj in 1:N
        R11 = (apply(jacobian(prob_vf, x0, set(parbif, lens, p + δ)), ζs[jj]) .- apply(jacobian(prob_vf, x0, set(parbif, lens, p - δ)), ζs[jj])) ./ (2δ)
        Ψ01, cv, it = ls(Linv, E(R01))
        ~cv && @warn "[Normal form Nd Ψ01] linear solver did not converge"
        d2gidxjdpk[ii,jj] = dot(R11 .- R2(ζs[jj], Ψ01), ζ★s[ii])
    end
    verbose && (printstyled(color=:green, "\n──> b1 (∂²/∂x∂p)  = \n"); Base.display( d2gidxjdpk ))

    # coefficients of x^2
    d2gidxjdxk = zeros(Tvec, N, N, N)
    for ii in 1:N, jj in 1:N, kk in 1:N
        b2v = R2(ζs[jj], ζs[kk])
        d2gidxjdxk[ii, jj, kk] = dot(b2v, ζ★s[ii])
    end

    if verbose
        printstyled(color=:green, "\n──> b2 (∂²/∂x²) = \n")
        for ii in 1:N
            printstyled(color=:blue, "──> component $ii\n")
            Base.display( d2gidxjdxk[ii,:,:] ./ 2)
        end
    end

    # coefficient of x^3
    d3gidxjdxkdxl = zeros(Tvec, N, N, N, N)
    for jj in 1:N, kk in 1:N, ll in 1:N
        b3v = R3(ζs[jj], ζs[kk], ζs[ll])
        # d3gidxjdxkdxl[ii,jj,kk,ll] = dot(b3v, ζ★s[ii])

        wst, flag, it = ls(Linv, E(R2(ζs[ll], ζs[kk])))
        ~flag && @warn "[Normal Form Nd (wst)]linear solver did not converge"
        b3v .-= R2(ζs[jj], wst)
        # d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[jj], wst), ζ★s[ii])

        wst, flag, it = ls(Linv, E(R2(ζs[ll], ζs[jj])))
        ~flag && @warn "[Normal Form Nd (wst)]linear solver did not converge"
        b3v .-= R2(ζs[kk], wst)
        # d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[kk], wst), ζ★s[ii])

        wst, flag, it = ls(Linv, E(R2(ζs[kk], ζs[jj])))
        ~flag && @warn "[Normal Form Nd (wst)]linear solver did not converge"
        b3v .-= R2(ζs[ll], wst)
        # d3gidxjdxkdxl[ii,jj,kk,ll] -= dot(R2(ζs[ll], wst), ζ★s[ii])
        for ii in 1:N
            d3gidxjdxkdxl[ii, jj, kk, ll] = dot(b3v, ζ★s[ii])
        end
    end
    if verbose
        printstyled(color=:green, "\n──> b3 (∂³/∂x³) = \n")
        for ii in 1:N
            printstyled(color=:blue, "──> component $ii\n")
            Base.display( d3gidxjdxkdxl[ii,:,:,:] ./ 6 )
        end
    end

    return NdBranchPoint(x0, τ, p, parbif, lens, ζs, ζ★s, (a=dgidp, b1=d2gidxjdpk, b2=d2gidxjdxk, b3=d3gidxjdxkdxl), Symbol("$N-d"))
end

get_normal_form(br::ContResult, id_bif::Int; kwargs...) = get_normal_form(br.prob, br, id_bif; kwargs...)
get_normal_form(br::Branch, id_bif::Int; kwargs...) = get_normal_form(get_contresult(br), id_bif; kwargs...)

"""
$(SIGNATURES)

This function provides prediction for what the zeros of the reduced equation / normal form should be for the parameter value `δp`. The algorithm for finding these zeros is based on deflated newton.

## Arguments
- `J` jacobian of the normal form. It is evaluated with ForwardDiff otherwise.
- `perturb` perturb function used in Deflated newton
- `normN` norm used for newton.
"""
function predictor(bp::NdBranchPoint, δp::T;
        verbose::Bool = false,
        ampfactor = T(1),
        nbfailures = 30,
        maxiter = 100,
        perturb = identity,
        J = nothing,
        normN = norminf,
        optn::NewtonPar = NewtonPar(max_iterations = maxiter, verbose = verbose)) where T

    # dimension of the kernel
    n = length(bp.ζ)

    # find zeros of the normal on each side of the bifurcation point
    function _get_roots_nf(_ds)
        deflationOp = DeflationOperator(2, 1.0, [zeros(n)]; autodiff = true)
        prob = BifurcationProblem((z, p) -> perturb(bp(Val(:reducedForm), z, p)),
                                    (rand(n) .- 0.5) .* 1.1, _ds)
        if ~isnothing(J)
            @set! prob.VF.J = J
        end
        failures = 0
        # we allow for 30 failures of nonlinear deflation
        while failures < nbfailures
            outdef1 = newton(prob, deflationOp, optn, Val(:autodiff); normN = normN)
            if converged(outdef1)
                push!(deflationOp, ampfactor .* outdef1.u)
            else
                failures += 1
            end
            prob.u0 .= outdef1.u .+ 0.1 .* (rand(n) .- 0.5)
        end
        return deflationOp.roots
    end
    rootsNFm = _get_roots_nf(-abs(δp))
    rootsNFp = _get_roots_nf(abs(δp))
    println("\n──> BS from Non simple branch point")
    printstyled(color=:green, "──> we find $(length(rootsNFm)) (resp. $(length(rootsNFp))) roots before (resp. after) the bifurcation point counting the trivial solution (Reduced equation).\n")
    return (before = rootsNFm, after = rootsNFp)
end

"""
$(SIGNATURES)

This function provides prediction for what the zeros of the reduced equation / normal form should be should be for the parameter value `δp`. The algorithm for finding these zeros is based on deflated newton. The initial guesses are the vertices of the hypercube.

## Arguments
- `J` jacobian of the normal form. It is evaluated with ForwardDiff otherwise.
- `perturb` perturb function used in Deflated newton
- `normN` norm used for newton.
- `igs` vector of initial guesses. If not passed, these are the vertices of the hypercube.
"""
function predictor(bp::NdBranchPoint, ::Val{:exhaustive}, δp::T;
                verbose::Bool = false,
                ampfactor = T(1),
                nbfailures = 30,
                maxiter = 100,
                perturb = identity,
                J = nothing,
                igs = nothing,
                normN = norminf,
                optn::NewtonPar = NewtonPar(max_iterations = maxiter, verbose = verbose)) where T

    # dimension of the kernel
    n = length(bp.ζ)

    # initial guesses for newton
    if isnothing(igs)
        igs = Iterators.product((-1:1 for _= 1:n)...)
    end

    # find zeros of the normal on each side of the bifurcation point
    function _get_roots_nf(_ds)
        deflationOp = DeflationOperator(2, 1.0, [zeros(n)]; autodiff = true)

        prob = BifurcationProblem((z, p) -> perturb(bp(Val(:reducedForm), z, p)),
                                    zeros(n), _ds, @lens _)
        if ~isnothing(J)
            @set! prob.VF.J = J
        end
        failures = 0
        # we allow for 30 failures of nonlinear deflation
        for ci in igs
            prob.u0 .= [ci...] * ampfactor
            # outdef1 = newton(prob, deflationOp, optn, Val(:autodiff); normN = normN)
            outdef1 = newton(prob, optn; normN = normN)
            @debug _ds ci outdef1.converged prob.u0 outdef1.u
            if converged(outdef1)
                push!(deflationOp, outdef1.u)
            else
                failures += 1
            end
        end
        return deflationOp.roots
    end
    rootsNFm = _get_roots_nf(-abs(δp))
    rootsNFp = _get_roots_nf(abs(δp))
    println("\n──▶ BS from Non simple branch point")
    printstyled(color=:green, "──▶ we found $(length(rootsNFm)) (resp. $(length(rootsNFp))) roots before (resp. after) the bifurcation point counting the trivial solution (Reduced equation).\n")
    return (before = rootsNFm, after = rootsNFp)
end
####################################################################################################
"""
$(SIGNATURES)

Compute the Hopf normal form.

# Arguments
- `prob::AbstractBifurcationProblem` bifurcation problem
- `pt::Hopf` Hopf bifurcation point
- `ls` linear solver

# Optional arguments
- `verbose` bool to print information
"""
function hopf_normal_form(prob::AbstractBifurcationProblem, 
                            pt::Hopf, 
                            ls::AbstractLinearSolver; 
                            verbose::Bool = false,
                            L = nothing)
    δ = getdelta(prob)
    x0 = pt.x0
    p = pt.p
    lens = pt.lens
    parbif = set(pt.params, lens, p)
    ω = pt.ω
    ζ = pt.ζ
    cζ = conj.(pt.ζ)
    ζ★ = pt.ζ★

    # jacobian at the bifurcation point
    # do not recompute it if passed
    if isnothing(L)
        L = jacobian(prob, x0, parbif)
    end

    # we use BilinearMap to be able to call on complex valued arrays
    R2 = BilinearMap( (dx1, dx2)      -> d2F(prob, x0, parbif, dx1, dx2) ./2)
    R3 = TrilinearMap((dx1, dx2, dx3) -> d3F(prob, x0, parbif, dx1, dx2, dx3) ./6 )

    # −LΨ001 = R01
    R01 = (residual(prob, x0, set(parbif, lens, p + δ)) .- 
           residual(prob, x0, set(parbif, lens, p - δ))) ./ (2δ)
    Ψ001, cv, it = ls(L, -R01)
    ~cv && @debug "[Hopf Ψ001] Linear solver for J did not converge. it = $it"

    # a = ⟨R11(ζ) + 2R20(ζ,Ψ001), ζ∗⟩
    av = (apply(jacobian(prob, x0, set(parbif, lens, p + δ)), ζ) .-
          apply(jacobian(prob, x0, set(parbif, lens, p - δ)), ζ)) ./ (2δ)
    av .+= 2 .* R2(ζ, Ψ001)
    a = dot(av, ζ★)

    # (2iω−L)Ψ200 = R20(ζ, ζ)
    R20 = R2(ζ, ζ)
    Ψ200, cv, it = ls(L, R20; a₀ = Complex(0, 2ω), a₁ = -1)
    ~cv && @debug "[Hopf Ψ200] Linear solver for J did not converge. it = $it"
    # @assert Ψ200 ≈ (Complex(0, 2ω)*I - L) \ R20

    # −LΨ110 = 2R20(ζ, cζ)
    R20 = 2 .* R2(ζ, cζ)
    Ψ110, cv, it = ls(L, -R20)
    ~cv && @debug "[Hopf Ψ110] Linear solver for J did not converge. it = $it"

    # b = ⟨2R20(ζ, Ψ110) + 2R20(cζ, Ψ200) + 3R30(ζ, ζ, cζ), ζ∗⟩)
    bv = 2 .* R2(ζ, Ψ110) .+ 2 .* R2(cζ, Ψ200) .+ 3 .* R3(ζ, ζ, cζ)
    b = dot(bv, ζ★)

    verbose && println((a = a, b = b))
    pt.nf = (;a, b)
    if real(b) < 0
        pt.type = :SuperCritical
    elseif real(b) > 0
        pt.type = :SubCritical
    else
        pt.type = :Singular
    end
    verbose && printstyled(color = :red,"──▶ Hopf bifurcation point is: ", pt.type, "\n")
    return pt
end

"""
$(SIGNATURES)

Compute the Hopf normal form.

# Arguments
- `prob::AbstractBifurcationProblem` bifurcation problem
- `br` branch result from a call to [`continuation`](@ref)
- `ind_hopf` index of the bifurcation point in `br`
- `options` options for the Newton solver

# Optional arguments
- `nev = 5` number of eigenvalues to compute to estimate the spectral projector
- `verbose` bool to print information

# Available method

Once the normal form `hopfnf` has been computed, you can call `predictor(hopfnf, ds)` to obtain an estimate of the bifurcating periodic orbit.

"""
function hopf_normal_form(prob::AbstractBifurcationProblem,
                    br::AbstractBranchResult,
                    ind_hopf::Int;
                    nev = length(eigenvalsfrombif(br, id_bif)),
                    verbose::Bool = false,
                    lens = getlens(br),
                    Teigvec = getvectortype(br),
                    detailed = true,
                    scaleζ = norm)
    @assert br.specialpoint[ind_hopf].type == :hopf "The provided index does not refer to a Hopf Point"
    verbose && println("━"^53*"\n──▶ Hopf normal form computation")

    options = br.contparams.newton_options

    # bifurcation point
    bifpt = br.specialpoint[ind_hopf]
    eigRes = br.eig

    # eigenvalue
    λ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ω = imag(λ)

    # parameter for vector field
    p = bifpt.param
    parbif = set(getparams(br), lens, p)
    L = jacobian(prob, convert(Teigvec, bifpt.x), parbif)

    # right eigenvector
    if haseigenvector(br) == false
        # we recompute the eigen-elements if there were not saved during the computation of the branch
        _λ, _ev, _ = options.eigsolver(L, bifpt.ind_ev + 2)
        @assert _λ[bifpt.ind_ev] ≈ λ "We did not find the correct eigenvalue $λ. We found $(_λ)"
        ζ = geteigenvector(options.eigsolver, _ev, bifpt.ind_ev)
    else
        ζ = copy(geteigenvector(options.eigsolver, br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev))
    end
    ζ ./= scaleζ(ζ)

    # left eigen-elements
    _Jt = has_adjoint(prob) ? jad(prob, convert(Teigvec, bifpt.x), parbif) : adjoint(L)
    ζ★, λ★ = get_adjoint_basis(_Jt, conj(λ), options.eigsolver; nev, verbose)

    # check that λ★ ≈ conj(λ)
    abs(λ + λ★) > 1e-2 && @warn "We did not find the left eigenvalue for the Hopf point to be very close to the imaginary part:\nλ ≈ $λ,\nλ★ ≈ $λ★?\n You can perhaps increase the number of computed eigenvalues, the number is nev = $nev"

    # normalise left eigenvector
    ζ★ ./= dot(ζ, ζ★)
    @assert dot(ζ, ζ★) ≈ 1

    𝒯 = eltype(bifpt.x)
    hopfpt = Hopf(bifpt.x, bifpt.τ, bifpt.param,
        ω,
        parbif, lens,
        ζ, ζ★,
        (a = zero(Complex{𝒯}), 
         b = zero(Complex{𝒯})
                 ),
        :SuperCritical
    )
    return hopf_normal_form(prob, hopfpt, options.linsolver ; verbose, L)
end

"""
$(SIGNATURES)

This function provides prediction for the periodic orbits branching off the Hopf bifurcation point.

# Arguments
- `bp::Hopf` the bifurcation point
- `ds` at with distance relative to the bifurcation point do you want the prediction. Can be negative. Basically the parameter is `p = bp.p + ds`

# Optional arguments
- `verbose`    display information
- `ampfactor = 1` factor multiplied to the amplitude of the periodic orbit.
"""
function predictor(hp::Hopf, ds; verbose = false, ampfactor = 1 )
    # get the type
    𝒯 = eltype(hp.x0)

    # get the normal form
    nf = hp.nf
    a = nf.a
    b = nf.b

    # we need to find the type, supercritical or subcritical
    dsfactor = real(a) * real(b) < 0 ? 1 : -1
    dsnew::𝒯 = abs(ds) * dsfactor
    pnew::𝒯 = hp.p + dsnew

    # we solve a * ds + b * amp^2 = 0
    amp::𝒯 = ampfactor * sqrt(-dsnew * real(a) / real(b))

    # correction to Hopf Frequency
    ω::𝒯 = hp.ω + (imag(a) - imag(b) * real(a) / real(b)) * ds

    return (orbit = t -> hp.x0 .+ 2amp .* real.(hp.ζ .* exp(complex(0, t))),
            amp = 2amp,
            ω = ω,
            period = abs(2pi/ω),
            p = pnew,
            dsfactor = dsfactor)
end
################################################################################
# computation based on 
# James. “Centre Manifold Reduction for Quasilinear Discrete Systems.” Journal of Nonlinear Science 13, no. 1 (February 2003): 27–63. https://doi.org/10.1007/s00332-002-0525-x.
# and on
# Kuznetsov, Yuri A. Elements of Applied Bifurcation Theory. Vol. 112. Applied Mathematical Sciences. Cham: Springer International Publishing, 2023. https://doi.org/10.1007/978-3-031-22007-4.
# on page 202
function period_doubling_normal_form(prob::AbstractBifurcationProblem,
                                pt::PeriodDoubling, 
                                ls::AbstractLinearSolver; 
                                verbose::Bool = false)
    x0 = pt.x0
    p = pt.p
    lens = pt.lens
    parbif = set(pt.params, lens, p)
    ζ = pt.ζ |> real
    ζ★ = pt.ζ★ |> real
    δ = getdelta(prob)

    abs(dot(ζ, ζ)  - 1) > 1e-5 && @warn "eigenvector for multiplier -1 not normalized, dot = $(dot(ζ, ζ))"
    abs(dot(ζ★, ζ) - 1) > 1e-5 && @warn "adjoint eigenvector for multiplier -1 not normalized, dot = $(dot(ζ★, ζ))"

    # jacobian at the bifurcation point
    L = jacobian(prob, x0, parbif)

    # we use BilinearMap to be able to call on complex valued arrays
    R2 = BilinearMap( (dx1, dx2)      -> d2F(prob, x0, parbif, dx1, dx2) )
    R3 = TrilinearMap((dx1, dx2, dx3) -> d3F(prob, x0, parbif, dx1, dx2, dx3)  )
    E(x) = x .- dot(ζ★, x) .* ζ

    # coefficient of x*p
    R01 = (residual(prob, x0, set(parbif, lens, p + δ)) .- 
           residual(prob, x0, set(parbif, lens, p - δ))) ./ (2δ)
    R11 = (apply(jacobian(prob, x0, set(parbif, lens, p + δ)), ζ) .- 
           apply(jacobian(prob, x0, set(parbif, lens, p - δ)), ζ)) ./ (2δ)

    # (I − L)⋅Ψ01 = R01
    Ψ01, cv, it = ls(L, -E(R01); a₀ = -1)
    ~cv && @debug "[PD Ψ01] Linear solver for J did not converge. it = $it"
    a = dot(ζ★, R11 .+ R2(ζ, Ψ01))
    verbose && println("──▶ Normal form:   x⋅(-1+ a⋅δμ + b₃⋅x²)")
    verbose && println("──▶ a  = ", a)

    # coefficient of x^3
    # b = <ζ★, 3R2(h20, ζ) + R3(ζ, ζ, ζ) >
    # (I - L)⋅h20 = B(ζ,ζ)
    h2v = R2(ζ, ζ)
    h20, cv, it = ls(L, h2v; a₀ = -1) # h20 = (L - I) \ h2v
    ~cv && @debug "[PD h20] Linear solver for J did not converge. it = $it"
    b3v = R3(ζ, ζ, ζ) .- 3 .* R2(ζ, h20)
    b = dot(ζ★, b3v) / 6
    verbose && println("──▶ b₃ = ", b)

    nf = (a = a, b3 = b)
    # the second iterate of the normal for is x → -x -2⋅b3⋅x³
    if real(b) > 0
        type = :SuperCritical
    elseif real(b) < 0
        type = :SubCritical
    else
        type = :Singular
    end
    verbose && printstyled(color = :red,"──▶ Period-doubling bifurcation point is: ", type, "\n")
    return setproperties(pt, nf = nf, type = type)
end

function predictor(pd::PeriodDoubling, δp ; verbose = false, ampfactor = 1 )
    # the normal form is f(x) = x*(c*x^2 + ∂p - 1)
    # we find f²(x) = (∂p - 1)^2*x + (c*(∂p - 1)^3 + (∂p - 1)*c)*x^3
    #               = (1-2∂p)x -2cx^3 + h.o.t.
    # the predictor is sqrt(-c*(∂p^3 - 3*∂p^2 + 4*∂p - 2)*∂p*(∂p - 2))/(c*(∂p^3 - 3*∂p^2 + 4*∂p - 2))
    c = pd.nf.b3
    ∂p = pd.nf.a * δp
    if c * ∂p > 0
        ∂p *= -1
        δp *= -1
    end
    x1 = abs(sqrt(-c*(∂p^3 - 3*∂p^2 + 4*∂p - 2)*∂p*(∂p - 2))/(c*(∂p^3 - 3*∂p^2 + 4*∂p - 2)))
    return (;x0 = zero(x1), x1, δp)
end
################################################################################
"""
$(SIGNATURES)

Compute the Neimark-Sacker normal form.

# Arguments
- `prob::AbstractBifurcationProblem` bifurcation problem
- `pt::NeimarkSacker` Neimark-Sacker bifurcation point
- `ls` linear solver

# Optional arguments
- `verbose` bool to print information
"""
function neimark_sacker_normal_form(prob::AbstractBifurcationProblem, 
                            pt::NeimarkSacker,
                            ls::AbstractLinearSolver;
                            detailed = false,
                            verbose::Bool = false)
    δ = getdelta(prob)
    x0 = pt.x0
    p = pt.p
    lens = pt.lens
    parbif = set(pt.params, lens, p)
    ω = pt.ω
    ζ = pt.ζ
    cζ = conj.(pt.ζ)
    ζ★ = pt.ζ★

    # jacobian at the bifurcation point
    L = jacobian(prob, x0, parbif)

    # we use BilinearMap to be able to call on complex valued arrays
    R2 = BilinearMap( (dx1, dx2)      -> d2F(prob, x0, parbif, dx1, dx2) )
    R3 = TrilinearMap((dx1, dx2, dx3) -> d3F(prob, x0, parbif, dx1, dx2, dx3) )

    a = nothing

    # (I−L)⋅Ψ001 = R001
    if detailed
        R001 = (residual(prob, x0, set(parbif, lens, p + δ)) .- 
                residual(prob, x0, set(parbif, lens, p - δ))) ./ (2δ)
        Ψ001, cv, it = ls(L, -R001; a₁ = -1)
        ~cv && @debug "[NS Ψ001] Linear solver for J did not converge. it = $it"

        # a = ⟨R11(ζ) + 2R20(ζ,Ψ001),ζ★⟩
        av = (apply(jacobian(prob, x0, set(parbif, lens, p + δ)), ζ) .-
            apply(jacobian(prob, x0, set(parbif, lens, p - δ)), ζ)) ./ (2δ)
        av .+= 2 .* R2(ζ, Ψ001)
        a = dot(ζ★, av) * cis(-ω)
        verbose && println("──▶ a  = ", a)
    end

    # (exp(2iω)−L)⋅Ψ200 = R20(ζ,ζ)
    R20 = R2(ζ, ζ)
    Ψ200, cv, it = ls(L, R20; a₀ = cis(2ω), a₁ = -1)
    ~cv && @debug "[NS Ψ200] Linear solver for J did not converge. it = $it"
    # @assert Ψ200 ≈ (exp(Complex(0, 2ω))*I - L) \ R20

    # (I−L)⋅Ψ110 = 2R20(ζ,cζ)
    R20 = 2 .* R2(ζ, cζ)
    Ψ110, cv, it = ls(L, -R20; a₀ = -1)
    ~cv && @debug "[NS Ψ110] Linear solver for J did not converge. it = $it"

    # b = ⟨2R20(ζ,Ψ110) + 2R20(cζ,Ψ200) + 3R30(ζ,ζ,cζ), ζ∗⟩)
    bv = 2 .* R2(ζ, Ψ110) .+ 2 .* R2(cζ, Ψ200) .+ 3 .* R3(ζ, ζ, cζ)
    b = dot(ζ★, bv) * cis(-ω) / 2
    b /= 6

    # return coefficients of the normal form
    verbose && println((a = a, b = b))
    @set! pt.nf = (a = a, b = b)
    if real(b) < 0
        pt.type = :SuperCritical
    elseif real(b) > 0
        pt.type = :SubCritical
    else
        pt.type = :Singular
    end
    verbose && printstyled(color = :red,"──▶ Neimark-Sacker bifurcation point is: ", pt.type, "\n")
    return pt
end

"""
$(SIGNATURES)

Compute the Neimark-Sacker normal form.

# Arguments
- `prob::AbstractBifurcationProblem` bifurcation problem
- `br` branch result from a call to [`continuation`](@ref)
- `ind_ns` index of the bifurcation point in `br`
- `options` options for the Newton solver

# Optional arguments
- `nev = 5` number of eigenvalues to compute to estimate the spectral projector
- `verbose` bool to print information
- `detailed = false` compute the coefficient a in the normal form

"""
function neimark_sacker_normal_form(prob::AbstractBifurcationProblem,
                    br::AbstractBranchResult,
                    ind_ns::Int;
                    nev = length(eigenvalsfrombif(br, id_bif)),
                    verbose::Bool = false,
                    lens = getlens(br),
                    Teigvec = getvectortype(br),
                    detailed = true,
                    scaleζ = norm)

    verbose && println("━"^53*"\n──▶ Neimark-Sacker normal form computation")

    options = br.contparams.newton_options

    # bifurcation point
    bifpt = br.specialpoint[ind_ns]
    eigRes = br.eig

    # eigenvalue
    λ = eigRes[bifpt.idx].eigenvals[bifpt.ind_ev]
    ω = imag(λ)

    # parameter for vector field
    p = bifpt.param
    parbif = set(getparams(br), lens, p)
    L = jacobian(br.prob, convert(Teigvec, bifpt.x), parbif)

    # right eigenvector
    if haseigenvector(br) == false
        # we recompute the eigen-elements if there were not saved during the computation of the branch
        _λ, _ev, _ = options.eigsolver(L, bifpt.ind_ev + 2)
        @assert _λ[bifpt.ind_ev] ≈ λ "We did not find the correct eigenvalue $λ. We found $(_λ)"
        ζ = geteigenvector(options.eigsolver, _ev, bifpt.ind_ev)
    else
        ζ = copy(geteigenvector(options.eigsolver ,br.eig[bifpt.idx].eigenvecs, bifpt.ind_ev))
    end
    ζ ./= scaleζ(ζ)

    # left eigen-elements
    _Jt = has_adjoint(prob) ? jad(prob, convert(Teigvec, bifpt.x), parbif) : adjoint(L)
    ζ★, λ★ = get_adjoint_basis(_Jt, conj(λ), options.eigsolver; nev = nev, verbose = verbose)

    # check that λ★ ≈ conj(λ)
    abs(λ + λ★) > 1e-2 && @warn "We did not find the left eigenvalue for the Neimark-Sacker point to be very close to the imaginary part:\nλ ≈ $λ,\nλ★ ≈ $λ★?\n You can perhaps increase the number of computed eigenvalues, the number is nev = $nev"

    # normalise left eigenvector
    ζ★ ./= dot(ζ, ζ★)
    @assert dot(ζ, ζ★) ≈ 1

    nspt = NeimarkSacker(bifpt.x, bifpt.τ, bifpt.param,
        ω,
        parbif, lens,
        ζ, ζ★,
        (a = zero(Complex{eltype(bifpt.x)}), b = zero(Complex{eltype(bifpt.x)}) ),
        :SuperCritical
    )
    return neimark_sacker_normal_form(prob, nspt, options.linsolver ; verbose, detailed)
end
