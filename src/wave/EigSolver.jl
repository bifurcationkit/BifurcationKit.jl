using IterativeSolvers, KrylovKit, Arpack, LinearAlgebra

# abstract type for generalised eigenvector
abstract type AbstractGEigenSolver <: AbstractEigenSolver end
# abstract type for Matrix-Free eigensolvers
abstract type AbstractGMFEigenSolver <: AbstractMFEigenSolver end
abstract type AbstractGFloquetSolver <: AbstractFloquetSolver end

convertToGEV(l::AbstractGEigenSolver, B) = l
####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
The struct `Default` is used to  provide the backslash operator to our Package
"""
@with_kw struct DefaultGEig{T, Tb <: AbstractMatrix} <: AbstractGEigenSolver
	which::T = real		# how do we sort the computed eigenvalues
	B::Tb
end

function (l::DefaultGEig)(Jac, nev; kwargs...)
	# I put Array so we can call it on small sparse matrices
	F = eigen(Array(Jac), l.B)
	I = sortperm(F.values, by = l.which, rev = true)
	nev2 = min(nev, length(I))
	J = findall( abs.(F.values[I]) .< 100000)
	# we perform a conversion to Complex numbers here as the type can change from Float to Complex along the branch, this would cause a bug
	return Complex.(F.values[I[J[1:nev2]]]), Complex.(F.vectors[:, I[J[1:nev2]]]), true, 1
end

convertToGEV(l::DefaultEig, B) = DefaultGEig(l.which, B)
