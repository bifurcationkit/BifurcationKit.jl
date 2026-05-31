# abstract type for generalised eigenvector
abstract type AbstractGEigenSolver <: AbstractEigenSolver end
# abstract type for Matrix-Free eigensolvers
abstract type AbstractMFGEigenSolver <: AbstractGEigenSolver end
abstract type AbstractGFloquetSolver <: AbstractFloquetSolver end

convertToGEV(l::AbstractGEigenSolver, B) = l
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function gev(l::DefaultEig, A, B, nev; kwargs...)
    # we convert to Array so we can call it on small sparse matrices
    F = LA.eigen(__to_array_for_eig(A), __to_array_for_eig(B))
    return Complex.(F.values), Complex.(F.vectors)
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
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Create an eigensolver for DAE, Basically a GEV with mass matrix.

# Internal fields

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

#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Solvers for default \ operator (backslash)
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
The struct `Default` is used to  provide the backslash operator to our Package
"""
@with_kw struct DefaultGEig{T, Tb <: AbstractMatrix} <: AbstractGEigenSolver
    which::T = real # how do we sort the computed eigenvalues
    B::Tb
end

function (l::DefaultGEig)(Jac, nev; kwargs...)
    # I put Array so we can call it on small sparse matrices
    F = LA.eigen(Array(Jac), l.B)
    I = sortperm(F.values, by = l.which, rev = true)
    nev2 = min(nev, length(I))
    J = findall( abs.(F.values[I]) .< 100000)
    # we perform a conversion to Complex numbers here as the type can change from Float to Complex along the branch, this would cause a bug
    return Complex.(F.values[I[J[begin:nev2]]]), Complex.(F.vectors[:, I[J[begin:nev2]]]), true, 1
end

convertToGEV(l::DefaultEig, B) = DefaultGEig(l.which, Array(B)) # we convert B from sparse to Array
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# case of sparse matrices or matrix free method via Arpack.jl
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

# Internal fields
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
        T = VI.scalartype(l.kwargs[:v0])
        Jmap = LinearMaps.LinearMap{T}(J, N, N; ismutating = false)
        λ, ϕ, ncv, = Arpack.eigs(Jmap, l.B; nev = nev, which = l.which, sigma = l.sigma, l.kwargs...)
    end
    Ind = sortperm(λ, by = l.by, rev = true)
    ncv < nev && @warn "$ncv eigenvalues have converged using Arpack.eigs, you requested $nev"
    return λ[Ind], ϕ[:, Ind], true, 1
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# case of sparse matrices or matrix free method via KrylovKit.jl
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Create an generalised eigen solver based on `KrylovKit.jl`.

# Internal fields
$(TYPEDFIELDS)

!!! danger "Restriction to B positive definite"
    The method from KryloKit.jl is restricted to B being positive definite.
"""
@with_kw struct GEigKrylovKit{T, Tb} <: AbstractMFGEigenSolver
    "KryloKit eigensolver."
    eigensolver::T

    "Mass matrix / operator."
    B::Tb
end

convertToGEV(l::EigKrylovKit, B) = GEigKrylovKit(l, B)

function (geig::GEigKrylovKit)(J, _nev; kwargs...)
    eig = geig.eigensolver
    # note that there is no need to order the eigen-elements. KrylovKit does it
    # with the option `which`, by decreasing order.
    kw = (verbosity = eig.verbose,
            krylovdim = eig.dim, 
            maxiter = eig.maxiter,
            tol = eig.tol, 
            # issymmetric = eig.issymmetric,
            # ishermitian = eig.ishermitian
            )
    if J isa AbstractMatrix && isnothing(eig.x₀)
        nev = min(_nev, size(J, 1))
        vals, vec, info = KrylovKit.geneigsolve((J, geig.B), nev, eig.which; kw...)
    else
        nev = min(_nev, length(eig.x₀))
        vals, vec, info = KrylovKit.geneigsolve((J, geig.B), eig.x₀, nev, eig.which; kw...)
    end
    info.converged == 0 && (@warn "KrylovKit.eigsolve solver did not converge")
    return vals, vec, info.converged > 0, info.numops
end
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# case of sparse matrices or matrix free method via ArnoldiMethod.jl
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
$(TYPEDEF)

Create an generalised eigen solver based on [ArnoldiMethod.jl](https://github.com/haampie/ArnoldiMethod.jl).

# Internal fields
$(TYPEDFIELDS)

!!! danger "Restriction to B positive definite"
    The method from KryloKit.jl is restricted to B being positive definite.
"""
@with_kw struct GEigArnoldiMethod{T, Tb} <: AbstractMFGEigenSolver
    "Eigensolver."
    eigensolver::T

    "Mass matrix / operator"
    B::Tb
end
convertToGEV(l::EigArnoldiMethod, B) = GEigArnoldiMethod(l, B)

function (geig::GEigArnoldiMethod)(J, _nev; kw...)
    gev(geig.eigensolver, J, geig.B, _nev; kw...)
end