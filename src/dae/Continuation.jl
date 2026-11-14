function continuation(prob::AbstractDAEBifProblem,
                      alg::AbstractContinuationAlgorithm,
                      contparams::ContinuationPar;
                      kwargs...)
    eigsolver = contparams.newton_options.eigsolver
    eigsolver_dae = _to_dae_eigen(eigsolver)
    contparams_dae = @set contparams.newton_options.eigsolver = eigsolver_dae
    _continuation(prob, alg, contparams_dae; kwargs...)
end

function continuation(prob::AbstractDAEBifProblem,
                      alg::DefCont,
                      contparams::ContinuationPar;
                      kwargs...)
    _deflated_continuation(prob, alg, contparams; kwargs...)
end

function _compute_eigenvalues(eigsolver,
                             prob::AbstractDAEBifProblem,
                             iter,
                             state,
                             u0,
                             par,
                             nev;
                             kwargs...)
    M = getmassmatrix(prob, getx(state), setparam(iter, getp(state)))
    return eigsolver(jacobian(prob, u0, par), M, nev; kwargs...)
end