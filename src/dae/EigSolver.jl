abstract type AbstractDAEEigenSolver <: AbstractEigenSolver end

struct EigenDAE{Te} <: AbstractDAEEigenSolver
    "Eigensolver."
    eigensolver::Te
end
# constructor
EigenDAE() = EigenDAE(DefaultEig())
_to_dae_eigen(eig::AbstractEigenSolver) = EigenDAE(eig)
_to_dae_eigen(eig::EigenDAE) = eig

getsolver(eig::EigenDAE) = eig

@views function (eig::EigenDAE)(J::AbstractMatrix, Mass, nev; kw...)
    eig = eig.eigensolver
    return gev(eig, J, Mass, nev; kw...)
end

function (eig::EigenDAE)(J, Mass, nev; kw...)
    eig = eig.eigensolver
    error("DAE eigen computations require a matrix valued Jacobian. We got a $(typeof(J)) which is not an `AbstractMatrix`. Matrix-free (operator) Jacobians are not supported yet for the generalized eigenproblem `J⋅x = λ⋅M⋅x`.")
end
