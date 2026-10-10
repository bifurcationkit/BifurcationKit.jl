"""
$(TYPEDEF)

Nonlinear solver of [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/) used as the corrector of a continuation, or as the solver of [`solve`](@ref) for `F(x, p0) = 0`, in place of BifurcationKit's own Newton method. It requires `using NonlinearSolve` (package extension `NonlinearSolveExt`).

# Constructor

    NonlinearSolveCorrector(alg; reuse_jacobian = false, solve_kwargs...)

- `alg` is any `NonlinearSolve` algorithm that accepts a user Jacobian, for example `NewtonRaphson(; linsolve = LUFactorization())` or `TrustRegion()`. The Jacobian reuse (`jacobian_reuse = JacobianReuse(...)`) and the linear solver (`linsolve`, with its preconditioners) are options of `alg`.
- `reuse_jacobian = true` carries the Jacobian, its age and its factorization from one corrector to the next (`reinit!(cache, u0; p, reuse_jacobian = true)`), so the reuse policy of `alg` decides when to rebuild it across continuation steps, not only within one. It needs a NonlinearSolve.jl whose `reinit!` accepts `reuse_jacobian`. With the default `false`, every corrector starts from a fresh Jacobian.
- `solve_kwargs` are passed to `NonlinearSolve.init`. The tolerance `tol` and the number of iterations `max_iterations` always come from `NewtonPar`, not from here.

# Usage

    using NonlinearSolve
    corrector = NonlinearSolveCorrector(NewtonRaphson(; linsolve = LUFactorization()))
    # the corrector of the pseudo-arclength continuation
    continuation(prob, PALC(; corrector), opts)
    # the corrector of natural continuation
    continuation(prob, Natural(; corrector), opts)
    # a Newton solve
    solve(prob, corrector, NewtonPar())

The bordered system `[F(x, p); N(x, p)] = 0` of PALC (`N` is the arclength constraint) is solved in `(x, p)`. Its Jacobian `[J ∂ₚF; ∇N]` is built from `jacobian(prob, x, p)`, a forward difference for `∂ₚF` and the exact `∇N`. It is rebuilt only when `alg` asks for it, which is what makes `jacobian_reuse` effective, whereas BifurcationKit's Newton rebuilds it at every iteration.

One NonlinearSolve cache is built at the first corrector and `reinit!` at the next ones, which keeps its allocations; whether the Jacobian survives from one step to the next is `reuse_jacobian`. Use a new `NonlinearSolveCorrector` for a new problem.

# Matrix-free Jacobians

A dense linear solver (for example `LUFactorization()`) needs `jacobian(prob, x, p)` to be an `AbstractMatrix`. With an iterative one, for example `KrylovJL_GMRES(; precs)`, it may also be a function `dx -> J dx`, and the products of the bordered system are formed from it. NonlinearSolve rebinds this operator to each iterate, so every step is an exact Newton step, while an explicit `JacobianReuse` policy keeps the preconditioner built by `precs` for an earlier iterate until the policy refreshes it. `precs(A, p)` receives in `A` that operator, with `A.u` the unknown (`x`, or `[x; p]` for PALC) and `A.p` a context to hand to

- [`corrector_jacobian`](@ref)`(A.p, A.u)`, the `jacobian(prob, x, p)` of the iterate (built once per iterate, whatever the number of products);
- [`corrector_matrix`](@ref)`(A.p, A.u, M)`, the dense matrix of the corrector's linear system when `M` stands in for that Jacobian (PALC borders it with `∂ₚF` and the arclength row), to factor as the preconditioner.

!!! warning "Differences with the default corrector"
    - the vector `x` must be an `AbstractVector`.
    - `NewtonPar` options `linsolver`, `linesearch`, `α`, `αmin`, `verbose` are not used by `alg`; use the options of `alg` instead. The bordered linear solver `bls` of `PALC` is not used.
    - the `callback` of `continuation` (`callback_newton`) is called before the first iteration and at the end, not at every iteration.
    - `p` is kept in `[p_min, p_max]` by the box constraints of `alg` when it allows bounds (`SciMLBase.allowsbounds(alg)`, for example the bounded trust region). Otherwise `p` is not clamped during the iterations, and a corrector that ends outside is not converged and `continuation` shortens the step.

# Internal fields
$(TYPEDFIELDS)
"""
struct NonlinearSolveCorrector{Talg, Tkw <: NamedTuple} <: AbstractNonLinearSolver
    "algorithm of NonlinearSolve.jl"
    alg::Talg
    "keyword arguments passed to `NonlinearSolve.init`"
    solve_kwargs::Tkw
    "carry the Jacobian from one corrector to the next (`reinit!(...; reuse_jacobian)`)"
    reuse_jacobian::Bool
    "cache of the bordered problem of PALC, built at the first corrector"
    palc_cache::Base.RefValue{Any}
    "cache of the problem at fixed parameter (Natural, `solve`), built at the first corrector"
    fixed_cache::Base.RefValue{Any}
end

function NonlinearSolveCorrector(alg::Any; reuse_jacobian::Bool = false, solve_kwargs...)
    if isnothing(Base.get_extension(@__MODULE__, :NonlinearSolveExt))
        error("`NonlinearSolveCorrector` requires the package NonlinearSolve.jl. Please run `using NonlinearSolve` first.")
    end
    return NonlinearSolveCorrector(alg, NamedTuple(solve_kwargs), reuse_jacobian, Ref{Any}(nothing), Ref{Any}(nothing))
end

# implemented in `ext/NonlinearSolveExt`
function _newton_nonlinearsolve end
function _newton_palc_nonlinearsolve end

"""
    corrector_jacobian(context, u)

`jacobian(prob, x, p)` at the unknown `u` of a [`NonlinearSolveCorrector`](@ref) solve (`x`, or `[x; p]` for PALC), for the `precs` of a matrix-free linear solver, which gets `context` and `u` as `A.p` and `A.u`. It is built once per `u`.
"""
function corrector_jacobian end

"""
    corrector_matrix(context, u, M)

The dense matrix of the linear system of a [`NonlinearSolveCorrector`](@ref) solve at `u` when the matrix `M` replaces the Jacobian: `M` itself at fixed parameter, `[M ∂ₚF; row c]` for PALC. A preconditioner factors it.
"""
function corrector_matrix end

# the corrector of Natural: BifurcationKit's Newton or NonlinearSolve
_corrector_newton(::Nothing, prob::AbstractBifurcationProblem, x0::Any, params0::Any, options::NewtonPar; kwargs...) = _newton(prob, x0, params0, options; kwargs...)
_corrector_newton(corrector::NonlinearSolveCorrector, prob::AbstractBifurcationProblem, x0::Any, params0::Any, options::NewtonPar; kwargs...) = _newton_nonlinearsolve(corrector, prob, x0, params0, options; kwargs...)

# the corrector of PALC
_corrector_palc(::Nothing, iter::AbstractContinuationIterable, state::AbstractContinuationState, dotθ::Any; kwargs...) = newton_palc(iter, state, dotθ; kwargs...)
_corrector_palc(corrector::NonlinearSolveCorrector, iter::AbstractContinuationIterable, state::AbstractContinuationState, dotθ::Any; kwargs...) = _newton_palc_nonlinearsolve(corrector, iter, state, dotθ; kwargs...)

solve(prob::AbstractBifurcationProblem, corrector::NonlinearSolveCorrector, options::NewtonPar; kwargs...) = _newton_nonlinearsolve(corrector, prob, getu0(prob), getparams(prob), options; kwargs...)
