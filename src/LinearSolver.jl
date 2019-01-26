using IterativeSolvers, KrylovKit, Parameters

# In this file, we regroud a way to provide linear solver for the Package

abstract type LinearSolver end

# The function linsolve(y, J, x) must return whether the solve was successfull and how many steps were required for the solve

####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
The struct `Default` is used to  provide the backslash operator to our Package
"""
struct Default <: LinearSolver end

function (l::Default)(J, x)
    return J \ x, true, 1
end

####################################################################################################
# Solvers for IterativeSolvers
####################################################################################################
@with_kw mutable struct GMRES_IterativeSolvers{T} <: LinearSolver
    tol::T = T(1e-4)        # tolerance for solver
    restart::Int64 = 200    # number of restarts
    maxiter::Int64 = 100
    N = 0                   # dimension of the problem
    verbose = true
    log = truex
end

function (l::GMRES_IterativeSolvers{T})(J, rhs) where T
    J_map = (v) -> J(v)
    Jmap = LinearMap{eltype(rhs)}(J_map, l.N, l.N ; ismutating = false)
    res = IterativeSolvers.gmres(Jmap, rhs, tol = l.tol, log = l.log, verbose = l.verbose, restart = l.restart, maxiter = l.maxiter)
    (res[2].iters >= l.maxiter) && printstyled("IterativeSolvers.gmres iterated maxIter =$(res[2].iters) times without achieving the desired tolerance.\n", color=:red)
    return res[1], length(res)>1, res[2].iters
end

####################################################################################################
# Solvers for KrylovKit
####################################################################################################
@with_kw mutable struct GMRES_KrylovKit{T} <: LinearSolver
    dim::Int64 = KrylovDefaults.krylovdim # Krylov Dimension
    atol::T  = T(KrylovDefaults.tol)     # absolute tolerance for solver
    rtol::T = T(KrylovDefaults.tol)      # relative tolerance for solver
    restart::Int64 = 200    # number of restarts
    maxiter::Int64 = KrylovDefaults.maxiter
    verbose::Int = 0
end

function (l::GMRES_KrylovKit{T})(J, rhs) where T
    res, info = KrylovKit.linsolve(J, rhs, rtol = l.rtol, verbosity = l.verbose, krylovdim = l.dim, maxiter = l.maxiter, atol = l.atol)
    return res, true, info.numops
end
