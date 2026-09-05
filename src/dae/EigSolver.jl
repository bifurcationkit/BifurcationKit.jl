abstract type AbstractDAEEigenSolver <: AbstractEigenSolver end

"""
$(TYPEDEF)

Eigen solver wrapper for Differential Algebraic Equations of the form
`M⋅u' = F(u)`. It solves the **generalized** eigen problem

    J(x, p)⋅v = λ⋅M(x, p)⋅v

where `J` is the jacobian of the vector field and `M` the mass matrix, by
delegating to `gev(eigensolver, J, Mass, nev)`.

It is called as

    eig(J, Mass, nev)

with a **matrix** valued jacobian `J` (matrix-free / operator jacobians are not
supported for the generalized problem). The identity mass marker
[`IdentityOperator`](@ref) is special-cased: the call then reduces to the plain
eigen solve `eigensolver(J, nev)`.

A `continuation` on any DAE problem (`AbstractDAEBifProblem`) wraps the
user-provided eigen solver into `EigenDAE` automatically, so that stability and
bifurcation detection use the generalized eigenvalues of `(J, M)`.

# Fields
$(TYPEDFIELDS)

# Constructors
- `EigenDAE(eigensolver)` wraps `eigensolver` (e.g. `DefaultEig()`, `EigArpack()`, `EigArnoldiMethod()`).
- `EigenDAE()` defaults to `EigenDAE(DefaultEig())`.
"""
struct EigenDAE{Te} <: AbstractDAEEigenSolver
    "Eigensolver."
    eigensolver::Te
end
# constructor
EigenDAE() = EigenDAE(DefaultEig())
_to_dae_eigen(eig::AbstractEigenSolver) = EigenDAE(eig)
_to_dae_eigen(eig::EigenDAE) = eig

getsolver(eig::EigenDAE) = eig

@views function (eig::EigenDAE)(J, Mass, nev; kw...)
    eig = eig.eigensolver
    return gev(eig, J, Mass, nev; kw...)
end

@views function (eig::EigenDAE)(J::AbstractMatrix, ::IdentityOperator, nev; kw...)
    eig = eig.eigensolver
    return eig(J, nev; kw...)
end

# function (eig::EigenDAE)(J, Mass, nev; kw...)
#     eig = eig.eigensolver
#     error("DAE eigen computations require a matrix valued Jacobian. We got a $(typeof(J)) which is not an `AbstractMatrix`. Matrix-free (operator) Jacobians are not supported yet for the generalized eigenproblem `J⋅x = λ⋅M⋅x`.")
# end
