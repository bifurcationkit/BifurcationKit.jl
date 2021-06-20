using IterativeSolvers, KrylovKit, Arpack, LinearAlgebra

abstract type AbstractEigenSolver end
# abstract type for Matrix-Free eigensolvers
abstract type AbstractMFEigenSolver <: AbstractEigenSolver end
abstract type AbstractFloquetSolver <: AbstractEigenSolver end

# the following function returns the n-th eigenvectors computed by an eigen solver. This function is necessary given the different return types each eigensolver has
geteigenvector(eigsolve::ES, vecs, n::Union{Int, Array{Int64,1}}) where {ES <: AbstractEigenSolver} = vecs[:, n]
####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
The struct `Default` is used to  provide the backslash operator to our Package
"""
@with_kw struct DefaultEig{T} <: AbstractEigenSolver
	which::T = real		# how do we sort the computed eigenvalues
end

function (l::DefaultEig)(J, nev; kwargs...)
	# I put Array so we can call it on small sparse matrices
	F = eigen(Array(J))
	I = sortperm(F.values, by = l.which, rev = true)
	nev2 = min(nev, length(I))
	# we perform a conversion to Complex numbers here as the type can change from Float to Complex along the branch, this would cause a bug
	return Complex.(F.values[I[1:nev2]]), Complex.(F.vectors[:, I[1:nev2]]), true, 1
end

# case of sparse matrices or matrix free method
"""
$(TYPEDEF)
$(TYPEDFIELDS)

More information is available at [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl). You can pass the following parameters `tol=0.0, maxiter=300, ritzvec=true, v0=zeros((0,))`.

# Constructor

`EigArpack(sigma = nothing, which = :LR; kwargs...)`
"""
struct EigArpack{T, Tby, Tw} <: AbstractEigenSolver
	"Shift for Shift-Invert method with `(J - sigma⋅I)"
	sigma::T

	"Which eigen-element to extract :LR, :LM, ..."
	which::Symbol

	"Sorting function, default to real"
	by::Tby

	"Keyword arguments passed to EigArpack"
	kwargs::Tw
end

EigArpack(sigma = nothing, which = :LR; kwargs...) = EigArpack(sigma, which, real, kwargs)

function (l::EigArpack)(J, nev; kwargs...)
	if J isa AbstractMatrix
		λ, ϕ, ncv = Arpack.eigs(J; nev = nev, which = l.which, sigma = l.sigma, l.kwargs...)
	else
		if !(:v0 in keys(l.kwargs))
			error("The v0 argument must be provided in EigArpack for the matrix-free case")
		end
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
"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
@with_kw struct EigKrylovKit{T, vectype} <: AbstractMFEigenSolver
	"Krylov Dimension"
	dim::Int64 = KrylovDefaults.krylovdim

	"Tolerance"
	tol::T = 1e-4

	"Number of restarts"
	restart::Int64 = 200

	"Maximum number of iterations"
	maxiter::Int64 = KrylovDefaults.maxiter

	"Verbosity ∈ {0,1,2}"
	verbose::Int = 0

	"Which eigenvalues are looked for :LR (largest real), :LM, ..."
	which::Symbol = :LR

	"If the linear map is symmetric, only meaningful if T<:Real"
	issymmetric::Bool = false

	"If the linear map is hermitian"
	ishermitian::Bool = false

	"Example of vector to usen for Krylov iterations"
	x₀::vectype = nothing
end

function (l::EigKrylovKit{T, vectype})(J, _nev; kwargs...) where {T, vectype}
	# note that there is no need to order the eigen-elements. KrylovKit does it
	# with the option `which`, by decreasing order.
	if J isa AbstractMatrix && isnothing(l.x₀)
		nev = min(_nev, size(J, 1))
		vals, vec, info = KrylovKit.eigsolve(J, nev, l.which;  verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, tol = l.tol, issymmetric = l.issymmetric, ishermitian = l.ishermitian)
	else
		nev = min(_nev, length(l.x₀))
		vals, vec, info = KrylovKit.eigsolve(J, l.x₀, nev, l.which;  verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, tol = l.tol, issymmetric = l.issymmetric, ishermitian = l.ishermitian)
	end
	# (length(vals) != _nev) && (@warn "EigKrylovKit returned $(length(vals)) eigenvalues instead of the $_nev requested")
	info.converged == 0 && (@warn "KrylovKit.eigsolve solver did not converge")
	return vals, vec, true, info.numops
end

geteigenvector(eigsolve::EigKrylovKit{T, vectype}, vecs, n::Union{Int, Array{Int64,1}}) where {T, vectype} = vecs[n]
####################################################################################################
# Solvers for ArnoldiMethod
####################################################################################################
# case of sparse matrices or matrix free method
"""
$(TYPEDEF)
$(TYPEDFIELDS)

More information is available at [ArnoldiMethod.jl](https://github.com/haampie/ArnoldiMethod.jl). For example, you can pass the parameters `tol, mindim, maxdim, restarts`.

# Constructor

`EigArnoldiMethod(;sigma = nothing, which = ArnoldiMethod.LR(), x₀ = nothing, kwargs...)`
"""
struct EigArnoldiMethod{T, Tby, Tw, Tkw, vectype} <: AbstractEigenSolver
	"Shift for Shift-Invert method"
	sigma::T

	"Which eigen-element to extract LR(), LM(), ..."
	which::Tw

	"how do we sort the computed eigenvalues, defaults to real"
	by::Tby

	"Key words arguments passed to EigArpack"
	kwargs::Tkw

	"Example of vector used for Krylov iterations"
	x₀::vectype
end

EigArnoldiMethod(;sigma = nothing, which = ArnoldiMethod.LR(), x₀ = nothing, kwargs...) = EigArnoldiMethod(sigma, which, real, kwargs, x₀)

function (l::EigArnoldiMethod)(J, nev; kwargs...)
	if J isa AbstractMatrix
		if isnothing(l.sigma)
			decomp, history = ArnoldiMethod.partialschur(J; nev = nev, which = l.which, l.kwargs...)
		else
			F = factorize(l.sigma * LinearAlgebra.I - J)
			Jmap = LinearMap{eltype(J)}((y, x) -> ldiv!(y, F, x), size(J, 1), ismutating=true)
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
	ncv = length(λ)
	ncv < nev && @warn "$ncv eigenvalues have converged using ArnoldiMethod.partialschur, you requested $nev"
	return Complex.(λ[Ind]), Complex.(ϕ[:, Ind]), true, 1
end
