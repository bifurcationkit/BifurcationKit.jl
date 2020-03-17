using IterativeSolvers, KrylovKit, LinearAlgebra

# In this file, we provide linear solvers for the Package

abstract type AbstractLinearSolver end

# The function linsolve(J, x) must return whether the solve was successfull and how many steps were required for the solve.

# the following function can be used to cache some factorization, see DefaultLS() case for example
function (ls::AbstractLinearSolver)(J, rhs1, rhs2; kwargs...)
	sol1, flag1, it1 = ls(J, rhs1; kwargs...)
	sol2, flag2, it2 = ls(J, rhs2; kwargs...)
	return sol1, sol2, flag1 & flag2, (it1, it2)
end

####################################################################################################
# The following functions are used for the Continuation of Hopf points and the computation of Floquet multipliers

# this function returns a₀ * I + a₁ .* J and ensures that we don't do unnecessary computations like 0*I + 1*J
function _axpy(J, a₀, a₁)
	if a₀ == 0
		if a₁ == 1
			return J
		else
			return a₁ .* J
		end
	elseif a₀ == 1
		if a₁ == 1
			return I + J
		else
			return I + a₁ .* J
		end
	else
		return a₀ * I + a₁ .* J
	end
end

function _axpy_op(J, v::AbstractArray, a₀, a₁)
	if a₀ == 0
		if a₁ == 1
			return apply(J, v)
		else
			return a₁ .* apply(J, v)
		end
	elseif a₀ == 1
		if a₁ == 1
			return v .+ apply(J, v)
		else
			return v .+ a₁ .* apply(J, v)
		end
	else
		return a₀ .* v .+ a₁ .* apply(J, v)
	end
end

####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
The struct `DefaultLS` is used to  provide the backslash operator
"""
struct DefaultLS <: AbstractLinearSolver end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# the options a₀, a₁ are only used for the Hopf Newton / Continuation
function (l::DefaultLS)(J, rhs; a₀ = 0, a₁ = 1, kwargs...)
	return _axpy(J, a₀, a₁) \ rhs, true, 1
end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# with multiple RHS. We can cache the factorization in this case
# the options a₀, a₁ are only used for the Hopf Newton / Continuation
function (l::DefaultLS)(J, rhs1, rhs2; a₀ = 0, a₁ = 1, kwargs...)
	Jfact = lu(_axpy(J, a₀, a₁))
	return Jfact \ rhs1, Jfact \ rhs2, true, (1, 1)
end
####################################################################################################
# Solvers for IterativeSolvers
####################################################################################################
@with_kw mutable struct GMRESIterativeSolvers{T, Tl, Tr} <: AbstractLinearSolver
	tol::T = 1e-4							# tolerance for solver
	restart::Int64 = 200					# number of restarts
	maxiter::Int64 = 100
	N::Int64 = 0							# dimension of the problem
	verbose::Bool = false					# display information during iterations
	log::Bool = true						# record information
	initially_zero::Bool = true				# start with zero guess
	Pl::Tl = IterativeSolvers.Identity()	# left preconditioner
	Pr::Tr = IterativeSolvers.Identity()	# right preconditioner
end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# the optional shift is only used for the Hopf Newton / Continuation
function (l::GMRESIterativeSolvers{T, Tl, Tr})(J, rhs; a₀ = 0, a₁ = 1, kwargs...) where {T, Ts, Tl, Tr}
	# no need to use fancy axpy! here because IterativeSolvers "only" handles AbstractArray
	J_map = v -> _axpy_op(J, v, a₀, a₁)
	Jmap = LinearMap{T}(J_map, l.N, l.N ; ismutating = false)
	res = IterativeSolvers.gmres(Jmap, rhs; tol = l.tol, log = l.log, verbose = l.verbose, restart = l.restart, maxiter = l.maxiter, initially_zero = l.initially_zero, Pl = l.Pl, Pr = l.Pr, kwargs...)
	(res[2].iters >= l.maxiter) && (@warn "IterativeSolvers.gmres iterated maxIter = $(res[2].iters) times without achieving the desired tolerance.\n")
	return res[1], length(res) > 1, res[2].iters
end

@with_kw mutable struct GMRESIterativeSolvers!{T, Tl, Tr} <: AbstractLinearSolver
	tol::T = 1e-4							# tolerance for solver
	restart::Int64 = 200					# number of restarts
	maxiter::Int64 = 100
	N::Int64 = 0							# dimension of the problem
	verbose::Bool = false					# display information during iterations
	log::Bool = true						# record information
	initially_zero::Bool = true				# start with zero guess
	Pl::Tl = IterativeSolvers.Identity()	# left preconditioner
	Pr::Tr = IterativeSolvers.Identity()	# right preconditioner
end

function (l::GMRESIterativeSolvers!{T, Tl, Tr})(J, rhs; kwargs...) where {T, Ts, Tl, Tr}
	# no need to use fancy axpy! here because IterativeSolvers "only" handles AbstractArray
	Jmap = LinearMap{T}((o, v) -> J(o, v), l.N, l.N ; ismutating = true)
	res = IterativeSolvers.gmres(Jmap, rhs; tol = l.tol, log = l.log, verbose = l.verbose, restart = l.restart, maxiter = l.maxiter, initially_zero = l.initially_zero, Pl = l.Pl, Pr = l.Pr, kwargs...)
	(res[2].iters >= l.maxiter) && (@warn "IterativeSolvers.gmres iterated maxIter = $(res[2].iters) times without achieving the desired tolerance.\n")
	return res[1], length(res) > 1, res[2].iters
end
####################################################################################################
# Solvers for KrylovKit
####################################################################################################
@with_kw mutable struct GMRESKrylovKit{T, Tl} <: AbstractLinearSolver
	dim::Int64 = KrylovDefaults.krylovdim # Krylov Dimension
	atol::T  = KrylovDefaults.tol		  # absolute tolerance for solver
	rtol::T  = KrylovDefaults.tol		  # relative tolerance for solver
	restart::Int64 = 200				  # number of restarts
	maxiter::Int64 = KrylovDefaults.maxiter
	verbose::Int64 = 0
	issymmetric::Bool = false				# if the linear map is symmetric, only meaningful if T<:Real
	ishermitian::Bool = false 				# if the linear map is hermitian
	isposdef::Bool    = false 				# if the linear map is positive definite
	Pl::Tl			  = nothing				# left preconditioner
end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# the optional shift is only used for the Hopf Newton / Continuation
function (l::GMRESKrylovKit{T, Tl})(J, rhs; a₀ = 0, a₁ = 1, kwargs...) where {T, Tl}
	if Tl == Nothing
		res, info = KrylovKit.linsolve(J, rhs, a₀, a₁; rtol = l.rtol, verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, atol = l.atol, issymmetric = l.issymmetric, ishermitian = l.ishermitian, isposdef = l.isposdef, kwargs...)
		info.converged == 0 && (@warn "KrylovKit.linsolve solver did not converge")
	else # use preconditioner
		res, info = KrylovKit.linsolve(x -> (out = apply(J, x); ldiv!(l.Pl, out)), ldiv!(l.Pl, copy(rhs)), a₀, a₁; rtol = l.rtol, verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, atol = l.atol, issymmetric = l.issymmetric, ishermitian = l.ishermitian, isposdef = l.isposdef, kwargs...)
		info.converged == 0 && (@warn "KrylovKit.linsolve solver did not converge")
	end
	return res, true, info.numops
end
