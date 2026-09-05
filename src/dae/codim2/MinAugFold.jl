function compute_eigenvalues(eigsolver::FoldEig, 
                             iter::ContIterable{FoldCont, Tprob},
                             state,
                             u0,
                             par,
                             nev = getcontparams(iter).nev; 
                             kwargs...) where {Tprob <: FoldMAProblem{ <: FoldMinimallyAugmentedFormulation{ <: DAEMassBifProblem}}}
    𝐏𝐛 = getprob(iter)
    𝐅 = get_formulation(𝐏𝐛)
    # the mass matrix is evaluated at the physical (non bordered) solution and full parameters,
    # consistently with the `update!` methods of the Fold MA formulation
    M = getmassmatrix(𝐅.prob_vf, getvec(getx(state), 𝐅), getparams(iter, state))
    J = jacobian(𝐏𝐛, u0, par)
    return eigsolver(J, M, nev; kwargs...)
end

@views function (eig::FoldEig{Tp, <: EigenDAE})(Jma::AbstractMatrix, Mass, nev; k...) where {Tp}
    eigenelts = eig.eigsolver(Jma[begin:end-1, begin:end-1], Mass, nev; k...)
end