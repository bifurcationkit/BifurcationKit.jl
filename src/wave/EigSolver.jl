abstract type AbstractWaveEigenSolver <: AbstractEigenSolver end

"""
$(TYPEDEF)

Basic eigen solver to compute the stability of the wave based on the eigenvalues of `J + η * ∂`.

# Internal fields
$(TYPEDFIELDS)
"""
struct EigenWave{Te} <: AbstractWaveEigenSolver
    "Eigensolver."
    eigensolver::Te
    matrix_free::Bool
end

EigenWave() = EigenWave(DefaultEig(), false)

@views function (eigw::EigenWave)(J::AbstractMatrix, nev; kw...)
    eig = eigw.eigensolver
    # we remove the constraints
    return eig(J[1:end-1, 1:end-1], nev; kw...)
end

function (eigw::EigenWave)(J, nev; kw...)
    eig = eigw.eigensolver
    return eig(J, nev; kw...)
end

function compute_eigenvalues(eigw::EigenWave, iter::ContIterable, state, u0, par, nev = iter.contparams.nev; k...)
    wrap = getprob(iter)
    twprob = get_discretization(wrap)
    J = if eigw.matrix_free
        # using dx -> twprob(u0, par, dx)[1:end-nc] woul be wrong because it contains the term ds⋅∂
        dx -> _jvp_for_eigenwave(twprob, u0, par, dx)
    else
        jacobian(wrap, u0, par)
    end
    return eigw(J, nev; iter, state, k...)
end

@views function compute_eigenvalues(eigw::EigenWave{ <: EigenDAE}, iter::ContIterable, state, u0, par, nev = iter.contparams.nev; kw...)
    wrap = getprob(iter)
    twprob = get_discretization(wrap)
    prob = twprob.prob_vf
    Mass = getmassmatrix(prob, getx(state), setparam(iter, getp(state)))
    J = jacobian(wrap, u0, par)
    eig = eigw.eigensolver
    return eig(J[1:end-twprob.nc, 1:end-twprob.nc], Mass, nev; kw...)
end

"""
$(TYPEDEF)

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
$(TYPEDEF)

Eigen solver to compute the stability of the wave based on the eigenvalues of the GEV, see [documentation](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/intro_wave/#Wave-stability).

# Internal fields
$(TYPEDFIELDS)
"""
struct GEigenWave{Te} <: AbstractWaveEigenSolver
    "Generalized eigensolver."
    eigensolver::Te
    matrix_free::Bool
end

GEigenWave() = GEigenWave(nothing, false)

function (geig::GEigenWave)(J, nev; kw...)
    eig = geig.eigensolver
    return eig(J, nev; kw...)
end

@views function compute_eigenvalues(geigw::GEigenWave{<: EigenDAE}, iter::ContIterable, state, u0, par, nev = iter.contparams.nev; kw...)
    wrap = getprob(iter)
    twprob = get_discretization(wrap)
    prob = twprob.prob_vf
    Mass = getmassmatrix(prob, getx(state), setparam(iter, getp(state)))
    J = jacobian(wrap, u0, par)
    eig = geigw.eigensolver
    return eig(J, SPA.blockdiag(Mass, SPA.sparse(LA.I, twprob.nc, twprob.nc)), nev; kw...)
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
convert_to_wave_eigen_solver(eigw::EigenWave, ::AbstractEigenSolver, B) = eigw
convert_to_wave_eigen_solver(eigw::EigenWave{Nothing}, eig0::AbstractEigenSolver, B) = EigenWave(eig0, eigw.matrix_free)

convert_to_wave_eigen_solver(geigw::GEigenWave, ::AbstractEigenSolver, B) = geigw
convert_to_wave_eigen_solver(geigw::GEigenWave{Nothing}, eig0::AbstractEigenSolver, B) = GEigenWave(convert_to_GEV(eig0, B), geigw.matrix_free)