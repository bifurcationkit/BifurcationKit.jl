function continuation(prob::DAEMassBifProblem,
                      alg::AbstractContinuationAlgorithm,
                      contparams::ContinuationPar;
                      kwargs...)
    eigsolver = contparams.newton_options.eigsolver
    eigsolver_dae = _to_dae_eigen(eigsolver)
    contparams_dae = @set contparams.newton_options.eigsolver = eigsolver_dae
    _continuation(prob, alg, contparams_dae; kwargs...)
end

function continuation(prob::DAEMassBifProblem,
                      alg::DefCont,
                      contparams::ContinuationPar;
                      kwargs...)
    _deflated_continuation(prob, alg, contparams; kwargs...)
end

function compute_eigenvalues(eigsolver::AbstractEigenSolver, 
                             iter::ContIterable{Tkind, Tprob},
                             state,
                             u0,
                             par,
                             nev = getcontparams(iter).nev; 
                             kwargs...) where {Tkind <: AbstractContinuationKind, Tprob <: DAEMassBifProblem}
    prob = getprob(iter)
    M = getmassmatrix(prob, getx(state), setparam(iter, getp(state)))
    return eigsolver(jacobian(getprob(iter), u0, par), M, nev; kwargs...)
end