function _hopf_ma_test(linbdsolver, M, J, a, b, J22, _zero, n, shift) 
    maj = MassAndJacobian(M, J)
    so = ShiftedOperator(J = maj, a₀ = shift)
    return linbdsolver(so, apply(M,a), apply(M,b), J22, _zero, n)
end

function __compute_bordered_vectors_hopf(linbdsolver, linbdsolver_adjoint, M, M★, J, J★, ω::𝒯, a, b, _zero_vector) where {𝒯}
    # we solve (J - iωM)v + M·a·σ1 = 0 with <M·b, v> = 1
    # this is the same bordered system as the one used to evaluate the Hopf MA residual
    # (see `hopf_ma_test`), so that the bordered vectors are consistent with the residual
    Ma = apply(M, a)
    Mb = apply(M, b)
    maj = MassAndJacobian(M, J); so = ShiftedOperator(J = maj, a₀ = Complex{𝒯}(0, -ω))
    v, σ1, cv, itv = linbdsolver(so, Ma, Mb, zero(𝒯), _zero_vector, one(𝒯))
    ~cv && @debug "Bordered linear solver for (J-iωM) did not converge."

    # we solve (J' + iωM')w + M·b·σ2 = 0 with <M·a, w> = 1
    # (conjugate adjoint of the bordered system above)
    maj = MassAndJacobian(M★, J★); so = ShiftedOperator(J = maj, a₀ = Complex{𝒯}(0, ω))
    w, σ2, cv, itw = linbdsolver_adjoint(so, Mb, Ma, zero(𝒯), _zero_vector, one(𝒯))
    ~cv && @debug "Bordered linear solver for (J-iωM)' did not converge."
    # we should have σ1 ≈ conj(σ2)
    return (; v, w, itv, itw, σ1, σ2)
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
