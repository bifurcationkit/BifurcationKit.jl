# abstract type for generalised eigenvector
abstract type AbstractGEigenSolver <: AbstractEigenSolver end
# abstract type for Matrix-Free eigensolvers
abstract type AbstractMFGEigenSolver <: AbstractGEigenSolver end
abstract type AbstractGFloquetSolver <: AbstractFloquetSolver end

convert_to_GEV(l::AbstractGEigenSolver, B) = l
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    gev(l, A, B, nev; kwargs...)

Solve the generalized eigenvalue problem ``A x = \\lambda B x`` using `l`.

# Arguments
- `l::AbstractEigenSolver`: the eigensolver configuration (controls sorting via `l.which`).
- `A`, `B`: the matrix (or operators) pair defining ``A x = \\lambda B x``.
- `nev`: number of requested eigenvalues.
"""
function gev(eig::DefaultEig, A, B, nev; kwargs...)
    # we convert to Array so we can call it on small sparse matrices
    F = LA.eigen(__to_array_for_eig(A), __to_array_for_eig(B); sortby = eig.which)
    return Complex.(F.values), Complex.(F.vectors), true, 1
end

# GEV, useful for computation of Floquet exponents based on collocation
function gev(eig::EigArpack, A, B, nev; kwargs...)
    if A isa AbstractMatrix
        λ, ϕ, ncv = Arpack.eigs(A, B; nev, which = eig.which, sigma = eig.sigma, eig.kwargs...)
    else
        if !(:v0 in keys(eig.kwargs))
            error("The v0 argument must be provided in EigArpack for the matrix-free case")
        end
        N = length(eig.kwargs[:v0])
        T = VI.scalartype(eig.kwargs[:v0])
        Jmap = LinearMaps.LinearMap{T}(A, N, N; ismutating = false)
        λ, ϕ, ncv, = Arpack.eigs(Jmap, B; nev = nev, which = eig.which, sigma = eig.sigma, eig.kwargs...)
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
        Ind = sortperm(values; by = eig.by, rev = true)
        ncv = length(values)
        if ncv < nev
            @warn "$ncv eigenvalues have converged using ArnoldiMethod.partialschur, you requested $nev"
        end
    else
        throw("Not defined yet. Please open an issue or make a Pull Request")
    end
    return Complex.(values[Ind]), Complex.(ϕ[:, Ind]), history.converged, 1
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
"""
Generalized eigen solver based on `LinearAlgebra.eigen`.
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

convert_to_GEV(l::DefaultEig, B) = DefaultGEig(l.which, Array(B)) # we convert B from sparse to Array
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
struct GEigArpack{T, Tb} <: AbstractGEigenSolver
    "Arpack eigensolver."
    eigensolver::T

    "Mass matrix"
    B::Tb
end

GEigArpack(; kw...) = GEigArpack(EigArpack(;kw...), nothing)
convert_to_GEV(eig::EigArpack, B) = GEigArpack(eig, B)

function (geig::GEigArpack)(J, nev; kw...)
    gev(geig.eigensolver, J, geig.B, nev; kw...)
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

convert_to_GEV(l::EigKrylovKit, B) = GEigKrylovKit(l, B)

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

    "Mass matrix / operator."
    B::Tb
end
convert_to_GEV(l::EigArnoldiMethod, B) = GEigArnoldiMethod(l, B)

function (geig::GEigArnoldiMethod)(J, _nev; kw...)
    gev(geig.eigensolver, J, geig.B, _nev; kw...)
end