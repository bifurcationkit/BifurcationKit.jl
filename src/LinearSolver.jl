using IterativeSolvers, LinearAlgebra
import Krylov
import KrylovKit: linsolve, KrylovDefaults # prevent from loading residual
norminf(x) = LinearAlgebra.norm(x, Inf)

# c'est tres mauvais comme interface, on ne peut pas utiliser le dispatch. Il vaut ieux utiliser solve

abstract type AbstractLinearSolver end
abstract type AbstractDirectLinearSolver <: AbstractLinearSolver end
abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

# The function linsolve(J, x; kwargs...) must return whether the solve was successful and how many steps were required for the solve.

# the following function can be used to cache some factorization, see DefaultLS() case for example
function (ls::AbstractLinearSolver)(J, rhs1, rhs2; kwargs...)
    sol1, flag1, it1 = ls(J, rhs1; kwargs...)
    sol2, flag2, it2 = ls(J, rhs2; kwargs...)
    return sol1, sol2, flag1 & flag2, (it1, it2)
end

####################################################################################################
# The two following functions are used for the Continuation of Hopf points and the computation of Floquet multipliers

"""
This function returns a₀ * I + a₁ * J and ensures that we don't perform unnecessary computations like 0*I + 1*J.
"""
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

"""
This function implements the operator a₀ * I + a₁ * J and ensures that we don't perform unnecessary computations like 0*I + 1*J.
"""
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

function _axpy_op!(o, J, v::AbstractArray, a₀, a₁)
    apply!(o, J, v)
    if a₀ == 0
        if a₁ == 1
            return o
        else
            o .*= a₁
            return o
        end
    elseif a₀ == 1
        if a₁ == 1
            return o .+= v
        else
            return o .= v .+ a₁ .* o
        end
    else
        return o .= a₀ .* v .+ a₁ .* o
    end
end
####################################################################################################
# Solvers for default \ operator (backslash)
####################################################################################################
"""
$(TYPEDEF)

This struct is used to provide the backslash operator `\`. Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.

## Fields
$(TYPEDFIELDS)
"""
@with_kw struct DefaultLS <: AbstractDirectLinearSolver
    "Whether to catch a factorization for multiple solves. Some operators may not support LU (like ApproxFun.jl) or QR factorization so it is best to let the user decides. Some matrices do not have `factorize` like `StaticArrays.MMatrix`."
    useFactorization::Bool = true
end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# the options a₀, a₁ are only used for the Hopf Newton / Continuation
function (l::DefaultLS)(J, rhs; a₀ = 0, a₁ = 1, kwargs...)
    return _axpy(J, a₀, a₁) \ rhs, true, 1
end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# with multiple RHS. We can cache the factorization in this case
# the options a₀, a₁ are only used for the Hopf Newton / Continuation
function (l::DefaultLS)(J, rhs1, rhs2; a₀ = 0, a₁ = 1, kwargs...)
    if l.useFactorization
        # factorize makes this type-unstable
        Jfact = factorize(_axpy(J, a₀, a₁))
        return Jfact \ rhs1, Jfact \ rhs2, true, (1, 1)
    else
        _J = _axpy(J, a₀, a₁)
        return _J \ rhs1, _J \ rhs2, true, (1, 1)
    end
end

"""
$(TYPEDEF)

[Mainly for debugging] This solver is used to test Moore-Penrose continuation. 
This is defined as an iterative pseudo-inverse linear solver. Used to solve `J * x = rhs`.

## Fields
$(TYPEDFIELDS)
"""
@with_kw struct DefaultPILS <: AbstractIterativeLinearSolver
    "Whether to catch a factorization for multiple solves. Some operators may not support LU (like ApproxFun.jl) or QR factorization so it is best to let the user decides. Some matrices do not have `factorize` like `StaticArrays.MMatrix`."
    useFactorization::Bool = true
end

function (l::DefaultPILS)(J, rhs; kwargs...)
    return J \ rhs, true, 1
end
####################################################################################################
# Solvers for IterativeSolvers
####################################################################################################
"""
$(TYPEDEF)

Linear solver based on `gmres` from `IterativeSolvers.jl`. Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.

## Fields
$(TYPEDFIELDS)
"""
@with_kw mutable struct GMRESIterativeSolvers{T, Tl, Tr} <: AbstractIterativeLinearSolver
    "Absolute tolerance for solver"
    abstol::T = 0.0

    "Relative tolerance for solver"
    reltol::T = 1e-8

    "Number of restarts"
    restart::Int64 = 200

    "Maximum number of iterations"
    maxiter::Int64 = 100

    "Dimension of the problem"
    N::Int64 = 0

    "Display information during iterations"
    verbose::Bool = false

    "Record information"
    log::Bool = true

    "Start with zero guess"
    initially_zero::Bool = true

    "Left preconditioner"
    Pl::Tl = IterativeSolvers.Identity()

    "Right preconditioner"
    Pr::Tr = IterativeSolvers.Identity()

    "Whether the linear operator is written inplace"
    ismutating::Bool = false
end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# the optional shift is only used for the Hopf Newton / Continuation
function (l::GMRESIterativeSolvers{𝒯, 𝒯l, 𝒯r})(J, rhs; a₀ = 0, a₁ = 1,
                                               kwargs...) where {𝒯, 𝒯l, 𝒯r}
    # no need to use fancy axpy! here because IterativeSolvers "only" handles AbstractArray
    if l.ismutating == true
        if ~((a₀ == 0) && (a₁ == 1))
            error("Perturbed inplace linear problem not done yet!")
        end
        Jmap = J isa AbstractArray ? J : LinearMap{𝒯}(J, l.N, l.N; ismutating = true)
    else
        J_map = v -> _axpy_op(J, v, a₀, a₁)
        Jmap = LinearMaps.LinearMap{𝒯}(J_map, length(rhs), length(rhs); ismutating = false)
    end
    res = IterativeSolvers.gmres(Jmap, rhs; abstol = l.abstol, reltol = l.reltol,
                                 log = l.log, verbose = l.verbose, restart = l.restart,
                                 maxiter = l.maxiter, initially_zero = l.initially_zero,
                                 Pl = l.Pl, Pr = l.Pr, kwargs...)
    if res[2].isconverged == false
        @debug "IterativeSolvers.gmres iterated maxIter = $(res[2].iters) times without achieving the desired tolerance.\n"
    end
    return res[1], res[2].isconverged, res[2].iters
end
####################################################################################################
# Solvers for KrylovKit
####################################################################################################
"""
$(TYPEDEF)

Create a linear solver based on `linsolve` from `KrylovKit.jl`. Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.

## Fields
$(TYPEDFIELDS)

!!! tip "Different linear solvers"
    By tuning the options, you can select CG, GMRES... see [here](https://jutho.github.io/KrylovKit.jl/stable/man/linear/#KrylovKit.linsolve)
"""
@with_kw mutable struct GMRESKrylovKit{T, Tl} <: AbstractIterativeLinearSolver
    "Krylov Dimension"
    dim::Int64 = KrylovDefaults.krylovdim[]

    "Absolute tolerance for solver"
    atol::T = KrylovDefaults.tol[]

    "Relative tolerance for solver"
    rtol::T = KrylovDefaults.tol[]

    "Maximum number of iterations"
    maxiter::Int64 = KrylovDefaults.maxiter[]

    "Verbosity ∈ {0,1,2}"
    verbose::Int64 = 0

    "If the linear map is symmetric, only meaningful if T<:Real"
    issymmetric::Bool = false

    "If the linear map is hermitian"
    ishermitian::Bool = false

    "If the linear map is positive definite"
    isposdef::Bool = false

    "Left preconditioner"
    Pl::Tl = nothing
end

# this function is used to solve (a₀ * I + a₁ * J) * x = rhs
# the optional shift is only used for the Hopf Newton / Continuation
function (l::GMRESKrylovKit{𝒯, 𝒯l})(J, rhs; a₀ = 0, a₁ = 1, kwargs...) where {𝒯, 𝒯l}
    if 𝒯l === Nothing
        res, info = KrylovKit.linsolve(J, rhs, a₀, a₁; rtol = l.rtol, verbosity = l.verbose,
                                       krylovdim = l.dim, maxiter = l.maxiter,
                                       atol = l.atol, issymmetric = l.issymmetric,
                                       ishermitian = l.ishermitian, isposdef = l.isposdef,
                                       kwargs...)
    else # use preconditioner
        # the preconditioner must be applied after the scaling
        function _linmap(dx)
            Jdx = apply(J, dx)
            # out = similar(dx)
            # ldiv!(out, l.Pl, Jdx)
            out = l.Pl \ Jdx
            axpby!(a₀, dx, a₁, out)
        end
        res, info = KrylovKit.linsolve(_linmap, ldiv!(similar(rhs), l.Pl, copy(rhs));
                                       rtol = l.rtol, verbosity = l.verbose,
                                       krylovdim = l.dim, maxiter = l.maxiter,
                                       atol = l.atol, issymmetric = l.issymmetric,
                                       ishermitian = l.ishermitian, isposdef = l.isposdef,
                                       kwargs...)
    end
    info.converged == 0 && (@debug "KrylovKit.linsolve solver did not converge")
    return res, info.converged == 1, info.numops
end
####################################################################################################
# Solvers for Krylov
####################################################################################################
"""
$(TYPEDEF)

Create a linear solver based on [Krylov.jl](https://jso.dev/Krylov.jl). Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.
You have access to `cg, cr, gmres, symmlq, cg_lanczos, cg_lanczos_shift_seq`...

## Fields 
$(TYPEDFIELDS)

## Other methods

Look at `KrylovLSInplace` for a method where the Krylov space is kept in memory
"""
mutable struct KrylovLS{K, Tl, Tr} <: AbstractIterativeLinearSolver
    "Krylov method"
    KrylovAlg::Symbol
    "Arguments passed to the linear solver"
    kwargs::K
    "Left preconditioner"
    Pl::Tl
    "Right preconditioner"
    Pr::Tr
end

function KrylovLS(args...; 
                  KrylovAlg :: Symbol = :gmres,
                  Pl = I, Pr = I,
                  kwargs...)
    return KrylovLS(KrylovAlg, kwargs, Pl, Pr)
end

function (l::KrylovLS)(J, rhs; a₀ = 0, a₁ = 1, kwargs...) 
    J_map = v -> _axpy_op(J, v, a₀, a₁)
    Jmap = LinearMaps.LinearMap{eltype(rhs)}(J_map, length(rhs), length(rhs); ismutating = false)
    sol, stats = Krylov.krylov_solve(Val(l.KrylovAlg), Jmap, rhs; l.kwargs..., M = l.Pl, N = l.Pr)
    return sol, stats.solved, stats.niter
end

"""
$(TYPEDEF)

Create an inplace linear solver based on [Krylov.jl](https://jso.dev/Krylov.jl). Can be used to solve `(a₀ * I + a₁ * J) * x = rhs`.

The Krylov space is pre-allocated. This is really great for GPU but also for CPU.

## Fields 
$(TYPEDFIELDS)
"""
mutable struct KrylovLSInplace{F, K, Tl, Tr} <: AbstractIterativeLinearSolver
    "Can be Krylov.GmresWorkspace for example."
    workspace::F
    "Krylov method."
    KrylovAlg::Symbol
    "Arguments passed to the linear solver."
    kwargs::K
    "Left preconditioner."
    Pl::Tl
    "Right preconditioner."
    Pr::Tr
    "Is the linear mapping inplace."
    is_inplace::Bool
end

"""
$(SIGNATURES)

Constructor for `KrylovLSInplace`.
"""
function KrylovLSInplace(args...;
                        n = 10,
                        m = 10,
                        memory = 20,
                        S = Vector{Float64},
                        KrylovAlg :: Symbol = :gmres,
                        Pl = I, Pr = I,
                        is_inplace = false,
                        kwargs...)
    if KrylovAlg == :gmres || KrylovAlg == :fgmres || KrylovAlg == :dqgmres || KrylovAlg == :fom || KrylovAlg == :diom
        workspace = Krylov.krylov_workspace(Val(KrylovAlg), m, n, S; memory)
    else
        workspace = Krylov.krylov_workspace(Val(KrylovAlg), m, n, S)
    end
    return KrylovLSInplace(workspace, KrylovAlg, kwargs, Pl, Pr, is_inplace)
end

function (l::KrylovLSInplace)(J, rhs; a₀ = 0, a₁ = 1, kwargs...) 
    if l.is_inplace
        J_map = (o,v) -> _axpy_op!(o, J, v, a₀, a₁)
        Jmap = LinearMaps.LinearMap{eltype(rhs)}(J_map, length(rhs), length(rhs); ismutating = true)
    else
        J_map = v -> _axpy_op(J, v, a₀, a₁)
        Jmap = LinearMaps.LinearMap{eltype(rhs)}(J_map, length(rhs), length(rhs); ismutating = false)
    end
    Krylov.krylov_solve!(l.workspace, Jmap, rhs; l.kwargs..., M = l.Pl, N = l.Pr)
    return Krylov.solution(l.workspace), Krylov.issolved(l.workspace), Krylov.iteration_count(l.workspace)
end
