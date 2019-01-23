using IterativeSolvers, KrylovKit, Parameters, Arpack, LinearAlgebra
# In this file, we regroud a way to provide eigen solvers

abstract type EigenSolver end


####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
The struct `Default` is used to  provide the backslash operator to our Package
"""
@with_kw struct Default_eig <: EigenSolver
    dim  = 200
    maxiter = 100
end

function (l::Default_eig)(J, nev::Int64)
    F = eigen(Array(J))
    I = sortperm(F.values, by = x-> real(x), rev = true)
    return F.values[I[1:nev]], F.vectors[:, I[1:nev]]
end

# case of sparse matrices
struct Default_eig_sp <: EigenSolver end

function (l::Default_eig_sp)(J, nev::Int64)
    λ, ϕ = Arpack.eigs(J, nev = nev, which = :LR)
    return λ, ϕ
end

####################################################################################################
# Solvers for IterativeSolvers
####################################################################################################
@with_kw struct eig_IterativeSolvers{T} <: EigenSolver
    tol::T = T(1e-4)        # tolerance for solver
    restart::Int64 = 200    # number of restarts
    maxiter::Int64 = 100
    N = 0                   # dimension of the problem
    verbose = true
    log = true
end

function (l::eig_IterativeSolvers{T})(J, nev::Int64) where T
    @assert 1==0 "not implemented"
    return res[1], length(res)>1, res[2].iters
end

####################################################################################################
# Solvers for KrylovKit
####################################################################################################
@with_kw struct eig_KrylovKit{T} <: EigenSolver
    dim::Int64 = 100        # Krylov Dimension
    tol::T = T(1e-4)        # tolerance for solver
    restart::Int64 = 200    # number of restarts
    maxiter::Int64 = 100
    verbose::Int = 0
    which = :LR
end

function (l::eig_KrylovKit{T})(J, nev::Int64) where T
    vals, vec, info = KrylovKit.eigsolve(J, nev, l.which;  verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, tol = l.tol)
    return vals, vec, true, info.numops
end
