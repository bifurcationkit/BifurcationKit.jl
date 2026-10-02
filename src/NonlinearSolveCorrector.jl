"""
$(TYPEDEF)

Nonlinear solver of [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/) used as the corrector of a continuation, or as the solver of [`solve`](@ref) for `F(x, p0) = 0`, in place of BifurcationKit's own Newton method. It requires `using NonlinearSolve` (package extension `NonlinearSolveExt`).

# Constructor

    NonlinearSolveCorrector(alg; solve_kwargs...)

- `alg` is any `NonlinearSolve` algorithm that accepts a user Jacobian, for example `NewtonRaphson(; linsolve = LUFactorization())` or `TrustRegion()`. The Jacobian reuse (`jacobian_reuse = JacobianReuse(...)`) and the linear solver (`linsolve`, with its preconditioners) are options of `alg`.
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

The bordered system `[F(x, p); N(x, p)] = 0` of PALC (`N` is the arclength constraint) is solved in `(x, p)`, with the Jacobian `[J ∂ₚF; ∇N]` assembled from `jacobian(prob, x, p)` (which must be an `AbstractMatrix`), a forward difference for `∂ₚF` and the exact `∇N`. It is rebuilt only when `alg` asks for it, which is what makes `jacobian_reuse` effective, whereas BifurcationKit's Newton rebuilds it at every iteration.

One NonlinearSolve cache is built at the first corrector and `reinit!` at the next ones, so what `alg` keeps in it between solves (factorizations, preconditioners, the Jacobian) survives from one continuation step to the next. Use a new `NonlinearSolveCorrector` for a new problem.

!!! warning "Differences with the default corrector"
    - the vector `x` must be an `AbstractVector` and the Jacobian an `AbstractMatrix`.
    - `NewtonPar` options `linsolver`, `linesearch`, `α`, `αmin`, `verbose` are not used by `alg`; use the options of `alg` instead. The bordered linear solver `bls` of `PALC` is not used.
    - the `callback` of `continuation` (`callback_newton`) is called before the first iteration and at the end, not at every iteration.
    - `p` is not clamped to `[p_min, p_max]` during the iterations. A corrector that ends outside is not converged and `continuation` shortens the step.

# Internal fields
$(TYPEDFIELDS)
"""
struct NonlinearSolveCorrector{Talg, Tkw <: NamedTuple} <: AbstractNonLinearSolver
    "algorithm of NonlinearSolve.jl"
    alg::Talg
    "keyword arguments passed to `NonlinearSolve.init`"
    solve_kwargs::Tkw
    "cache of the bordered problem of PALC, built at the first corrector"
    palc_cache::Base.RefValue{Any}
    "cache of the problem at fixed parameter (Natural, `solve`), built at the first corrector"
    fixed_cache::Base.RefValue{Any}
end

function NonlinearSolveCorrector(alg; solve_kwargs...)
    if isnothing(Base.get_extension(@__MODULE__, :NonlinearSolveExt))
        error("`NonlinearSolveCorrector` requires the package NonlinearSolve.jl. Please run `using NonlinearSolve` first.")
    end
    return NonlinearSolveCorrector(alg, NamedTuple(solve_kwargs), Ref{Any}(nothing), Ref{Any}(nothing))
end

# implemented in `ext/NonlinearSolveExt`
function _newton_nonlinearsolve end
function _newton_palc_nonlinearsolve end

# the corrector of Natural: BifurcationKit's Newton or NonlinearSolve
_corrector_newton(::Nothing, prob, x0, params0, options; kwargs...) = _newton(prob, x0, params0, options; kwargs...)
_corrector_newton(corrector::NonlinearSolveCorrector, prob, x0, params0, options; kwargs...) = _newton_nonlinearsolve(corrector, prob, x0, params0, options; kwargs...)

# the corrector of PALC
_corrector_palc(::Nothing, iter, state, dotθ; kwargs...) = newton_palc(iter, state, dotθ; kwargs...)
_corrector_palc(corrector::NonlinearSolveCorrector, iter, state, dotθ; kwargs...) = _newton_palc_nonlinearsolve(corrector, iter, state, dotθ; kwargs...)

solve(prob::AbstractBifurcationProblem, corrector::NonlinearSolveCorrector, options::NewtonPar; kwargs...) = _newton_nonlinearsolve(corrector, prob, getu0(prob), getparams(prob), options; kwargs...)
