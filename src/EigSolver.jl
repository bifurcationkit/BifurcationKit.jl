using IterativeSolvers, KrylovKit, Arpack
# In this file, we regroud a way to provide eigen solvers

abstract type AbstractEigenSolver end
abstract type AbstractMFEigenSolver <: AbstractEigenSolver end
abstract type AbstractFloquetSolver <: AbstractEigenSolver end

# the following function returns the n-th eigenvectors computed by an eigen solver. This function is necessary given the different return types each eigensolver has
geteigenvector(eigsolve::ES, vecs, n::Int) where {ES <: AbstractEigenSolver} = vecs[:, n]
geteigenvector(eigsolve::ES, vecs, I::Array{Int64,1}) where {ES <: AbstractEigenSolver} = vecs[:, I]
####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
The struct `Default` is used to  provide the backslash operator to our Package
"""
@with_kw struct DefaultEig{Tby} <: AbstractEigenSolver
	which::Tby = real		# how do we sort the computed eigenvalues
end

function (l::DefaultEig)(J, nev::Int64)
	# I put Array so we can call it on small sparse matrices
	F = eigen(Array(J))
	I = sortperm(F.values, by = l.which, rev = true)
	nev2 = min(nev, length(I))
	return Complex.(F.values[I[1:nev2]]), F.vectors[:, I[1:nev2]], true, 1
end

# case of sparse matrices or matrix free method
@with_kw struct EigArpack{T, Tby, Tw} <: AbstractEigenSolver
	sigma::T = nothing
	which::Symbol = :LR
	by::Tby = real			# how do we sort the computed eigenvalues.
	kwargs::Tw = nothing
end

EigArpack(sigma = nothing, which = :LR; kwargs...) = EigArpack(sigma, which, real, kwargs)

function (l::EigArpack)(J, nev::Int64)
	if J isa AbstractMatrix
		λ, ϕ, ncv = Arpack.eigs(J; nev = nev, which = l.which, sigma = l.sigma, l.kwargs...)
	else
		N = length(l.kwargs[:v0])
		T = eltype(l.kwargs[:v0])
		Jmap = LinearMap{T}(J, N, N; ismutating = false)
		λ, ϕ, ncv, = Arpack.eigs(Jmap; nev = nev, which = l.which, sigma = l.sigma, l.kwargs...)
	end
	I = sortperm(λ, by = l.by, rev = true)
	ncv < nev && @warn "$ncv eigenvalues have converged using Arpack.eigs, you requested $nev"
	return λ[I], ϕ[:, I], true, 1
end
####################################################################################################
# Solvers for KrylovKit
####################################################################################################
@with_kw struct EigKrylovKit{T, vectype} <: AbstractMFEigenSolver
	dim::Int64 = KrylovDefaults.krylovdim	# Krylov Dimension
	tol::T = 1e-4							# tolerance for solver
	restart::Int64 = 200					# number of restarts
	maxiter::Int64 = KrylovDefaults.maxiter
	verbose::Int = 0
	which::Symbol = :LR
	issymmetric::Bool = false				# if the linear map is symmetric, only meaningful if T<:Real
	ishermitian::Bool = false 				# if the linear map is hermitian
	x₀::vectype = nothing					# example of vector in case of a matrix-free operator
end

function (l::EigKrylovKit{T, vectype})(J, nev::Int64) where {T, vectype}
	if J isa AbstractMatrix && isnothing(l.x₀)
		vals, vec, info = KrylovKit.eigsolve(J, nev, l.which;  verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, tol = l.tol, issymmetric = l.issymmetric, ishermitian = l.ishermitian)
	else
		vals, vec, info = KrylovKit.eigsolve(J, l.x₀, nev, l.which;  verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, tol = l.tol, issymmetric = l.issymmetric, ishermitian = l.ishermitian)
	end
	info.converged == 0 && (@warn "KrylovKit.eigsolve solver did not converge")
	return vals, vec, true, info.numops
end

geteigenvector(eigsolve::EigKrylovKit{T, vectype}, vecs, n::Int) where {T, vectype} = vecs[n]
geteigenvector(eigsolve::EigKrylovKit{T, vectype}, vecs, I::Array{Int64,1}) where {T, vectype} = vecs[I]
