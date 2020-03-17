using IterativeSolvers, KrylovKit, Arpack, LinearAlgebra
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
@with_kw struct DefaultEig{Twh} <: AbstractEigenSolver
	which::Twh = real		# how do we sort the computed eigenvalues
end

function (l::DefaultEig)(J, nev::Int64)
	# I put Array so we can call it on small sparse matrices
	F = eigen(Array(J))
	I = sortperm(F.values, by = l.which, rev = true)
	nev2 = min(nev, length(I))
	return Complex.(F.values[I[1:nev2]]), F.vectors[:, I[1:nev2]], true, 1
end

# case of sparse matrices or matrix free method
struct EigArpack{T, Tby, Tw} <: AbstractEigenSolver
	sigma::T
	which::Symbol
	by::Tby			# how do we sort the computed eigenvalues.
	kwargs::Tw
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
	Ind = sortperm(λ, by = l.by, rev = true)
	ncv < nev && @warn "$ncv eigenvalues have converged using Arpack.eigs, you requested $nev"
	return λ[Ind], ϕ[:, Ind], true, 1
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
####################################################################################################
# Solvers for ArnoldiMethod
####################################################################################################
# case of sparse matrices or matrix free method
struct EigArnoldiMethod{T, Tby, Tw, Tkw, vectype} <: AbstractEigenSolver
	sigma::T
	which::Tw
	by::Tby			# how do we sort the computed eigenvalues.
	kwargs::Tkw
	x₀::vectype
end

EigArnoldiMethod(;sigma = nothing, which = ArnoldiMethod.LR(), x₀ = nothing, kwargs...) = EigArnoldiMethod(sigma, which, real, kwargs, x₀)

function (l::EigArnoldiMethod)(J, nev::Int64)
	if J isa AbstractMatrix
		if isnothing(l.sigma)
			decomp, history = ArnoldiMethod.partialschur(J; nev = nev, which = l.which, l.kwargs...)
		else
			F = factorize(l.sigma .* LinearAlgebra.I - J)
			Jmap = LinearMap{eltype(J)}((y, x) -> ldiv!(y, F, x), size(J,1), ismutating=true)
			decomp, history = ArnoldiMethod.partialschur(Jmap; nev = nev, which = l.which, l.kwargs...)
		end
	else
		N = length(l.x₀)
		T = eltype(l.x₀)
		isnothing(l.sigma) == false && @warn "Shift-Invert strategy not implemented for maps"
		Jmap = LinearMap{T}(J, N, N; ismutating = false)
		decomp, history = ArnoldiMethod.partialschur(Jmap; nev = nev, which = l.which, l.kwargs...)
	end
	λ, ϕ = partialeigen(decomp)
	# shift and invert
	if isnothing(l.sigma) == false
		λ .= l.sigma .- 1 ./ λ
	end
	Ind = sortperm(λ, by = l.by, rev = true)
	length(λ) < nev && @warn "$ncv eigenvalues have converged using ArnoldiMethod.partialschur, you requested $nev"
	return λ[Ind], ϕ[:, Ind], true, 1
end
