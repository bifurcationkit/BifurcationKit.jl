abstract type AbstractWaveEigenSolver <: AbstractEigenSolver end

"""
    Basic eigen solver to compute the stability of the wave based on the eigenvalues of `J + η * ∂`.
"""
struct EigenWave{Te} <: AbstractWaveEigenSolver
    "Eigensolver."
    eigensolver::Te
    matrix_free::Bool
end
# contructor
EigenWave() = EigenWave(DefaultEig(), false)

@views function (geig::EigenWave)(J::AbstractMatrix, nev; kw...)
    eig = geig.eigensolver
    # we remove the constraints
    return eig(J[1:end-1, 1:end-1], nev; kw...)
end

function (geig::EigenWave)(J, nev; kw...)
    eig = geig.eigensolver
    return eig(J, nev; kw...)
end

function compute_eigenvalues(geig::EigenWave, iter::ContIterable, state, u0, par, nev = iter.contparams.nev; k...)
    wrap = getprob(iter)
    twprob = get_discretization(wrap)
    J = if geig.matrix_free
        # using dx -> twprob(u0, par, dx)[1:end-1] woul be wrong because it contains the term ds⋅∂
        dx -> _jvp_for_eigenwave(twprob, u0, par, dx)
    else
        jacobian(wrap, u0, par)
    end
    return geig(J, nev; iter, state, k...)
end

"""
Return the jacobian-vector-product of the travelling wave problem without the constraints.
More precisely, it computes `J⋅du + η ⋅ ∂⋅du` where `η = x[end]` is the speed(s) of the travelling wave solution `x`.
This is needed for the computation of eigenvalues with matrix-free eigen-solver.
"""
@views function _jvp_for_eigenwave(pb, x::AbstractVector, pars, du::AbstractVector)
    # number of constraints
    nc = pb.nc
    if ~(length(du) + nc == length(x))
        error("[Wave JVP Eigen] We have an issue with the dimensions.")
    end
    # number of unknowns
    N = length(du)
    # array containing the result
    u = x[1:N]
    outu = similar(du)
    # get the speed
    s = Tuple(x[end-nc+1:end])
    ds = ntuple(zero, nc)
    _jvp_VF_plus_D!(pb, outu, u, du, s, ds, pars, Val(false))
    return outu
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    Eigen solver to compute the stability of the wave based on the eigenvalues of the GEV, see [documentation](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/intro_wave/#Wave-stability).
"""
struct GEigenWave{Te} <: AbstractWaveEigenSolver
    "Generalized eigensolver."
    eigensolver::Te
    matrix_free::Bool
end

# contructor
GEigenWave() = GEigenWave(nothing, false)

function (geig::GEigenWave)(J, nev; kw...)
    eig = geig.eigensolver
    return eig(J, nev; kw...)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
convert_to_wave_eigen_solver(eigw::EigenWave, eig0::AbstractEigenSolver, B) = eigw
convert_to_wave_eigen_solver(eigw::EigenWave{Nothing}, eig0::AbstractEigenSolver, B) = EigenWave(eig0, eigw.matrix_free)

convert_to_wave_eigen_solver(eigw::GEigenWave, eig0::AbstractEigenSolver, B) = eigw
convert_to_wave_eigen_solver(eigw::GEigenWave{Nothing}, eig0::AbstractEigenSolver, B) = GEigenWave(convert_to_GEV(eig0, B), eigw.matrix_free)