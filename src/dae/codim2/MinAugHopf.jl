function _init_hopf_vectors_minaug(dae::DAEMassBifProblem, bifpt, parbif, ω, bdlinsolver, bdlinsolver_adjoint, a, b, normC)
    # we use a minimally augmented formulation to set the initial vectors
    # we start with a vector similar to an eigenvector, we must ensure that
    # it is complex valued
    ζ = VI.scale(_copy(bifpt.x), one(Complex{VI.scalartype(bifpt.x)}))
    a = isnothing(a) ? _randn(ζ) : a; VI.scale!(a, 1 / normC(a))
    b = isnothing(b) ? _randn(ζ) : b; VI.scale!(b, 1 / normC(b))

    L = jacobian(dae, bifpt.x, parbif)
    M = getmassmatrix(dae, bifpt.x, parbif)
    L★ = ~has_adjoint(dae) ? adjoint(L) : jacobian_adjoint(dae, bifpt.x, parbif)

    (; v, w, itv, itw) = __compute_bordered_vectors_hopf(bdlinsolver, bdlinsolver_adjoint, M, L, L★, ω, a, b, VI.zerovector(a))

    @debug "RIGHT EIGENVECTORS" ω itv norminf(residual(dae, bifpt.x, parbif)) norminf(apply(L,v) - complex(0,ω)*apply(M,v)) norminf(apply(L,v) + complex(0,ω)*apply(M,v))

    @debug "LEFT  EIGENVECTORS" ω itw norminf(residual(dae, bifpt.x, parbif)) norminf(apply(L★, w) - complex(0,ω)*adjoint(M)*w) norminf(apply(L★,w) + complex(0,ω)*adjoint(M)*w)

    ζad = VI.scale(w,  1 / normC(w))
    ζ   = VI.scale(v,  1 / normC(v))
    return (; ζ, ζad)
end

function __compute_bordered_vectors_hopf(linbdsolver, linbdsolver_adjoint, M, J_at_xp, JAd_at_xp, ω::𝒯, a, b, _zero) where {𝒯}
    # we solve (J - iωM)v + M·a·σ1 = 0 with <M·b, v> = 1
    # this is the same bordered system as the one used to evaluate the Hopf MA residual
    # (see `hopf_ma_test`), so that the bordered vectors are consistent with the residual
    Ma = apply(M, a)
    Mb = apply(M, b)
    v, σ, cv, itv = linbdsolver(J_at_xp, Ma, Mb, zero(𝒯), _zero, one(𝒯); shift = Complex{𝒯}(0, -ω), Mass = M)
    ~cv && @debug "Bordered linear solver for (J-iωM) did not converge."

    # we solve (J' + iωM')w + M·b·σ2 = 0 with <M·a, w> = 1
    # (conjugate adjoint of the bordered system above)
    w, _, cv, itw = linbdsolver_adjoint(JAd_at_xp, Mb, Ma, zero(𝒯), _zero, one(𝒯); shift = Complex{𝒯}(0, ω), Mass = adjoint(M))
    ~cv && @debug "Bordered linear solver for (J-iωM)' did not converge."

    return (; v, w, itv, itw, σ)
end

function compute_eigenvalues(hopfeig::HopfEig, 
                             iter::ContIterable{HopfCont, Tprob},
                             state,
                             u0,
                             par,
                             nev = getcontparams(iter).nev; 
                             kwargs...) where {Tprob <: HopfMAProblem{ <: HopfMinimallyAugmentedFormulation{ <: DAEMassBifProblem}}}
    𝐏𝐛 = getprob(iter)
    𝐇 = get_formulation(𝐏𝐛)
    # the mass matrix is evaluated at the physical (non bordered) solution and full parameters,
    # consistently with the `update!` methods of the Hopf MA formulation
    M = getmassmatrix(𝐇.prob_vf, getvec(getx(state), 𝐇), getparams(iter, state))
    J = jacobian(𝐇.prob_vf, getvec(getx(state), 𝐇), getparams(iter, state))
    return hopfeig.eigsolver(J, M, nev; kwargs...)
end

@views function (eig::HopfEig{Tp, <: EigenDAE})(Jma::AbstractMatrix, Mass, nev; k...) where {Tp}
    return eig.eigsolver(Jma[begin:end-2, begin:end-2], Mass, nev; k...)
end