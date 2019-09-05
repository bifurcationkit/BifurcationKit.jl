using IterativeSolvers, KrylovKit, Parameters, Arpack, LinearAlgebra
# In this file, we regroud a way to provide eigen solvers

abstract type EigenSolver end

# the following function returns the n-th eigenvectors computed by an eigen solver. This function is necessary given the different return types each eigensolver has
getEigenVector(eigsolve::ES, vecs, n::Int) where {ES <: EigenSolver} = vecs[:, n]
####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
The struct `Default` is used to  provide the backslash operator to our Package
"""
struct Default_eig <: EigenSolver end

function (l::Default_eig)(J, nev::Int64)
	# I put Array so we can call it on small sparse matrices
	F = eigen(Array(J))
	I = sortperm(F.values, by = x-> real(x), rev = true)
	return F.values[I[1:nev]], F.vectors[:, I[1:nev]], 1
end

# case of sparse matrices
struct Default_eig_sp <: EigenSolver end

function (l::Default_eig_sp)(J, nev::Int64)
	λ, ϕ = Arpack.eigs(J, nev = nev, which = :LR)
	return λ, ϕ, 1
end
####################################################################################################
# Solvers for IterativeSolvers
####################################################################################################
@with_kw struct eig_IterativeSolvers{T} <: EigenSolver
	tol::T = T(1e-4)		# tolerance for solver
	restart::Int64 = 200	# number of restarts
	maxiter::Int64 = 100
	N = 0				   # dimension of the problem
	verbose = true
	log = true
end

function (l::eig_IterativeSolvers{T})(J, nev::Int64) where T
	# for now, we don't have an eigensolver for non hermitian matrices
	@assert 1==0 "Not implemented: IterativeSolvers does not have an eigensolver yet!"
	return res[1], length(res)>1, res[2].iters
end
####################################################################################################
# Solvers for KrylovKit
####################################################################################################
@with_kw struct eig_KrylovKit{T} <: EigenSolver
	dim::Int64 = KrylovDefaults.krylovdim	# Krylov Dimension
	tol::T = T(1e-4)						# tolerance for solver
	restart::Int64 = 200					# number of restarts
	maxiter::Int64 = KrylovDefaults.maxiter
	verbose::Int = 0
	which = :LR
end

function (l::eig_KrylovKit{T})(J, nev::Int64) where T
	@assert typeof(J) <:  AbstractMatrix
	vals, vec, info = KrylovKit.eigsolve(J, nev, l.which;  verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, tol = l.tol)
	return vals, vec, true, info.numops
end

getEigenVector(eigsolve::eig_KrylovKit{T}, vecs, n::Int) where T = vecs[n]

# Matrix-Free version, needs to specify an example of rhs x₀
@with_kw struct eig_MF_KrylovKit{T, vectype} <: EigenSolver
	dim::Int64 = KrylovDefaults.krylovdim		# Krylov Dimension
	tol::T = T(1e-4)							# tolerance for solver
	restart::Int64 = 200						# number of restarts
	maxiter::Int64 = KrylovDefaults.maxiter
	verbose::Int = 0
	which = :LR
	x₀::vectype
end

function (l::eig_MF_KrylovKit{T, vectype})(J, nev::Int64) where {T, vectype}
	vals, vec, info = KrylovKit.eigsolve(J, l.x₀, nev, l.which;  verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, tol = l.tol)
	return vals, vec, true, info.numops
end

getEigenVector(eigsolve::eig_MF_KrylovKit{T, vectype}, vecs, n::Int) where {T, vectype} = vecs[n]
