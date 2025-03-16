using IterativeSolvers, Arpack, LinearAlgebra

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
    which::T = real # how do we sort the computed eigenvalues
    B::Tb
end

function (l::DefaultGEig)(Jac, nev; kwargs...)
    # I put Array so we can call it on small sparse matrices
    F = eigen(Array(Jac), l.B)
    I = sortperm(F.values, by = l.which, rev = true)
    nev2 = min(nev, length(I))
    J = findall( abs.(F.values[I]) .< 100000)
    # we perform a conversion to Complex numbers here as the type can change from Float to Complex along the branch, this would cause a bug
    return Complex.(F.values[I[J[begin:nev2]]]), Complex.(F.vectors[:, I[J[begin:nev2]]]), true, 1
end

convertToGEV(l::DefaultEig, B) = DefaultGEig(l.which, Array(B)) # we convert B from sparse to Array
####################################################################################################
# case of sparse matrices or matrix free method
####################################################################################################
"""
$(TYPEDEF)
$(TYPEDFIELDS)

More information is available at [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl). You can pass the following parameters `tol=0.0, maxiter=300, ritzvec=true, v0=zeros((0,))`.

# Constructor

`EigArpack(sigma = nothing, which = :LR; kwargs...)`
"""
struct GEigArpack{T, Tby, Tw, Tb} <: AbstractGEigenSolver
    "Shift for Shift-Invert method with `(J - sigma⋅I)"
    sigma::T

    "Which eigen-element to extract :LR, :LM, ..."
    which::Symbol

    "Sorting function, default to real"
    by::Tby

    "Keyword arguments passed to EigArpack"
    kwargs::Tw

    "Mass matrix"
    B::Tb
end

GEigArpack(sigma = nothing, which = :LR; kwargs...) = EigArpack(sigma, which, real, kwargs)
convertToGEV(l::EigArpack, B) = GEigArpack(l.sigma, l.which, l.by, l.kwargs, B)

function (l::GEigArpack)(J, nev; kwargs...)
    if J isa AbstractMatrix
        λ, ϕ, ncv = Arpack.eigs(J, l.B; nev = nev, which = l.which, sigma = l.sigma, l.kwargs...)
    else
        if !(:v0 in keys(l.kwargs))
            error("The v0 argument must be provided in EigArpack for the matrix-free case")
        end
        N = length(l.kwargs[:v0])
        T = eltype(l.kwargs[:v0])
        Jmap = LinearMaps.LinearMap{T}(J, N, N; ismutating = false)
        λ, ϕ, ncv, = Arpack.eigs(Jmap, l.B; nev = nev, which = l.which, sigma = l.sigma, l.kwargs...)
    end
    Ind = sortperm(λ, by = l.by, rev = true)
    ncv < nev && @warn "$ncv eigenvalues have converged using Arpack.eigs, you requested $nev"
    return λ[Ind], ϕ[:, Ind], true, 1
end
