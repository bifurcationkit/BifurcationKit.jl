using IterativeSolvers, Arpack
import KrylovKit: eigsolve
import LinearAlgebra

abstract type AbstractEigenSolver end
abstract type AbstractDirectEigenSolver <: AbstractEigenSolver end
abstract type AbstractIterativeEigenSolver <: AbstractEigenSolver end
abstract type AbstractMFEigenSolver <: AbstractIterativeEigenSolver end
abstract type AbstractFloquetSolver <: AbstractEigenSolver end

# The following function returns the n-th eigenvectors computed by an eigen solver. 
# This function is necessary given the different return types each eigensolver has
geteigenvector(eigsolve::ES, vecs, n::Union{Int, AbstractVector{Int64}}) where {ES <: AbstractEigenSolver} = vecs[:, n]

getsolver(eig::AbstractEigenSolver) = eig
####################################################################################################
# Default Solvers
####################################################################################################
__to_array_for_eig(x) = Array(x)
__to_array_for_eig(x::Array) = x
"""
$(TYPEDEF)

The struct `DefaultEig` is used to  provide the `eigen` method to `BifurcationKit`.

## Fields
$(TYPEDFIELDS)

## Constructors
Just pass the above fields like `DefaultEig(; which = abs)`
"""
@with_kw struct DefaultEig{T} <: AbstractDirectEigenSolver
    "How do we sort the computed eigenvalues."
    which::T = real
end

function (eig::DefaultEig)(J, nev; kwargs...)
    # we convert to Array so we can call l on small sparse matrices
    F = LA.eigen(__to_array_for_eig(J); sortby = eig.which)
    nev2 = min(nev, length(F.values))
    # we perform a conversion to Complex numbers here as the type can 
    # change from Float to Complex along the branch, this would cause a bug
    return Complex.(F.values[end:-1:end-nev2+1]), Complex.(F.vectors[:, end:-1:end-nev2+1]), true, 1
end

function gev(l::DefaultEig, A, B, nev; kwargs...)
    # we convert to Array so we can call it on small sparse matrices
    F = LA.eigen(__to_array_for_eig(A), __to_array_for_eig(B))
    return Complex.(F.values), Complex.(F.vectors)
end
####################################################################################################
# Solvers for Arpack
####################################################################################################
"""
$(TYPEDEF)

Create an eigen solver based on [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl).

## Fields
$(TYPEDFIELDS)

# Constructor

`EigArpack(sigma = nothing, which = :LR; kwargs...)`

More information is available at [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl). You can pass the following parameters `tol = 0.0, maxiter = 300, ritzvec = true, v0 = zeros((0,))`.
"""
struct EigArpack{T, Tby, Tw} <: AbstractIterativeEigenSolver
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

function (eig::EigArpack)(J, nev; kwargs...)
    if J isa AbstractMatrix
        λ, ϕ, ncv = Arpack.eigs(J; nev, which = eig.which, sigma = eig.sigma, eig.kwargs...)
    else
        if !(:v0 in keys(eig.kwargs))
            error("The v0 argument must be provided in EigArpack for the matrix-free case")
        end
        N = length(eig.kwargs[:v0])
        T = eltype(eig.kwargs[:v0])
        Jmap = LinearMaps.LinearMap{T}(J, N, N; ismutating = false)
        λ, ϕ, ncv, = Arpack.eigs(Jmap; nev, which = eig.which, sigma = eig.sigma, eig.kwargs...)
    end
    return __sort_arpack(eig, λ, ϕ, ncv, nev)
end

# GEV, useful for computation of Floquet exponents based on collocation
function gev(eig::EigArpack, A, B, nev; kwargs...)
    if A isa AbstractMatrix
        λ, ϕ, ncv = Arpack.eigs(A, B; nev, sigma = eig.sigma, which = eig.which, eig.kwargs...)
    else
        error("Not defined yet. Please open an issue or make a Pull Request")
    end
    return __sort_arpack(eig, λ, ϕ, ncv, nev)
end

function __sort_arpack(eig, λ, ϕ, ncv, nev)
    Ind = sortperm(λ; by = eig.by, rev = true)
    ncv < nev && @warn "$ncv eigenvalues have converged using Arpack.eigs, you requested $nev"
    return λ[Ind], ϕ[:, Ind], true, 1
end
####################################################################################################
# Solvers for KrylovKit
####################################################################################################
"""
$(TYPEDEF)

Create an eigen solver based on `KrylovKit.jl`.

## Fields
$(TYPEDFIELDS)

## Constructors
Just pass the above fields like `EigKrylovKit(;dim=2)`
"""
@with_kw struct EigKrylovKit{T, vectype} <: AbstractMFEigenSolver
    "Krylov Dimension"
    dim::Int64 = KrylovDefaults.krylovdim[]

    "Tolerance"
    tol::T = 1e-4

    "Number of restarts"
    restart::Int64 = 200

    "Maximum number of iterations"
    maxiter::Int64 = KrylovDefaults.maxiter[]

    "Verbosity ∈ {0, 1, 2}"
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

function (eig::EigKrylovKit{T, vectype})(J, _nev; kwargs...) where {T, vectype}
    # note that there is no need to order the eigen-elements. KrylovKit does it
    # with the option `which`, by decreasing order.
    kw = (verbosity = eig.verbose,
            krylovdim = eig.dim, 
            maxiter = eig.maxiter,
            tol = eig.tol, 
            issymmetric = eig.issymmetric,
            ishermitian = eig.ishermitian)
    if J isa AbstractMatrix && isnothing(eig.x₀)
        nev = min(_nev, size(J, 1))
        vals, vec, info = KrylovKit.eigsolve(J, nev, eig.which; kw...)
    else
        nev = min(_nev, length(eig.x₀))
        vals, vec, info = KrylovKit.eigsolve(J, eig.x₀, nev, eig.which; kw...)
    end
    info.converged == 0 && (@warn "KrylovKit.eigsolve solver did not converge")
    return vals, vec, info.converged > 0, info.numops
end

geteigenvector(eigsolve::EigKrylovKit{T, vectype}, vecs, n::Union{Int, AbstractVector{Int64}}) where {T, vectype} = vecs[n]
####################################################################################################
# Solvers for ArnoldiMethod
####################################################################################################
"""
$(TYPEDEF)

## Fields
$(TYPEDFIELDS)

More information is available at [ArnoldiMethod.jl](https://github.com/haampie/ArnoldiMethod.jl). For example, you can pass the parameters `tol, mindim, maxdim, restarts`.

# Constructor

`EigArnoldiMethod(;sigma = nothing, which = ArnoldiMethod.LR(), x₀ = nothing, kwargs...)`
"""
struct EigArnoldiMethod{T, Tby, Tw, Tkw, vectype} <: AbstractIterativeEigenSolver
    "Shift for Shift-Invert method"
    sigma::T

    "Which eigen-element to extract LR(), LM(), ..."
    which::Tw

    "How do we sort the computed eigenvalues, defaults to real"
    by::Tby

    "Key words arguments passed to EigArpack"
    kwargs::Tkw

    "Example of vector used for Krylov iterations"
    x₀::vectype
end

EigArnoldiMethod(;sigma = nothing, which = ArnoldiMethod.LR(), x₀ = nothing, kwargs...) = EigArnoldiMethod(sigma, which, real, kwargs, x₀)

function (eig::EigArnoldiMethod)(J, nev; kwargs...)
    if J isa AbstractMatrix
        if isnothing(eig.sigma)
            decomp, history = ArnoldiMethod.partialschur(J; nev, 
                                                         which = eig.which,
                                                         eig.kwargs...)
        else
            F = LA.factorize(eig.sigma * LinearAlgebra.I - J)
            Jmap = LinearMaps.LinearMap{eltype(J)}((y, x) -> LA.ldiv!(y, F, x), size(J, 1);
                                        ismutating = true)
            decomp, history = ArnoldiMethod.partialschur(Jmap; nev, 
                                                         which = eig.which,
                                                         eig.kwargs...)
        end
    else
        N = length(eig.x₀)
        𝒯 = eltype(eig.x₀)
        if isnothing(eig.sigma) == false
            @warn "Shift-Invert strategy not implemented for maps"
        end
        Jmap = LinearMaps.LinearMap{𝒯}(J, N, N; ismutating = false)
        decomp, history = ArnoldiMethod.partialschur(Jmap; nev, which = eig.which,
                                                     eig.kwargs...)
    end
    λ, ϕ = ArnoldiMethod.partialeigen(decomp)
    # shift and invert
    if isnothing(eig.sigma) == false
        λ .= @. eig.sigma - 1 / λ
    end
    Ind = sortperm(λ; by = eig.by, rev = true)
    ncv = length(λ)
    ncv < nev &&
        @warn "$ncv eigenvalues have converged using ArnoldiMethod.partialschur, you requested $nev"
    return Complex.(λ[Ind]), Complex.(ϕ[:, Ind]), history.converged, 1
end

# GEV useful for computation of Floquet exponents based on collocation
function gev(eig::EigArnoldiMethod, A, B, nev; kwargs...)
    if A isa AbstractMatrix
        # Solve Ax = λBx using Shift-invert method 
        # (A - σ⋅B)⁻¹ B⋅x = 1/(λ-σ)x
        σ = isnothing(eig.sigma) ? 0 : eig.sigma
        P = LA.lu(A - σ * B)
        𝒯 = eltype(A)
        L = LinearMaps.LinearMap{𝒯}((y, x) -> LA.ldiv!(y, P, B * x), size(A, 1), ismutating = true)
        decomp, history = ArnoldiMethod.partialschur(L; nev, which = eig.which,
                                                         eig.kwargs...)
        vals, ϕ = ArnoldiMethod.partialeigen(decomp)
        values = @. 1/vals + σ
    else
        throw("Not defined yet. Please open an issue or make a Pull Request")
    end
    return Complex.(values), Complex.(ϕ), history.converged, 1
end
####################################################################################################
"""
$(TYPEDEF)

Create an eigensolver based on Shift-Invert strategy. Basically, one computes the eigen-elements of (J - σ⋅I)⁻¹.

## Fields

$(TYPEDFIELDS)
"""
struct ShiftInvert{T, Tls <: AbstractLinearSolver, Teig <: AbstractEigenSolver} <: AbstractEigenSolver
    "Shift."
    sigma::T
    "Linear solver to compute (J - σ⋅I)⁻¹."
    ls::Tls
    "Eigen-solver to compute the eigenvalues of (J - σ⋅I)⁻¹."
    eig::Teig
end

geteigenvector(eigsolve::ShiftInvert, vecs, n::Union{Int, AbstractVector{Int64}}) = geteigenvector(eigsolve.eig, vecs, n)

function (eig::ShiftInvert)(J, nev; kwargs...)
    # (a₀ * I + a₁ * J) * x = rhs
    function Jmap(rhs)
        eig.ls(J, rhs; a₀ = -eig.sigma , a₁ = 1)[1]
    end
    vals, vecs, cv, n = @time "SI-ev" eig.eig(Jmap, nev; kwargs...)
    return 1 ./vals .+ eig.sigma, vecs, cv, n
end
####################################################################################################
"""
$(TYPEDEF)

Create an eigensolver for DAE, Basically a GEV with mass matrix.

## Fields

$(TYPEDFIELDS)
"""
struct EigenMassMatrix{Tb, Teig <: AbstractEigenSolver} <: AbstractEigenSolver
    "Mass matrix"
    B::Tb
    "Eigen-solver"
    eig::Teig
end
geteigenvector(eigsolve::EigenMassMatrix, vecs, n::Union{Int, AbstractVector{Int64}}) = geteigenvector(eigsolve.eig, vecs, n)

function (eigsolve::EigenMassMatrix)(J, nev; kwargs...)
    return gev(eigsolve.eig, J, eigsolve.B, nev; kwargs...)
end