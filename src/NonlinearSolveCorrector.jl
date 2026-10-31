"""
$(TYPEDEF)

Nonlinear solver of [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/) used as the corrector of a continuation, or as the solver of [`solve`](@ref) for `F(x, p0) = 0`, in place of BifurcationKit's own Newton method. It requires `using NonlinearSolve` and `import BorderedLinearSolve` (package extension `NonlinearSolveExt`). BorderedLinearSolve exports its own `BorderingBLS`, so after `using BifurcationKit, BorderedLinearSolve` the name `BorderingBLS` is ambiguous and must be qualified (`BifurcationKit.BorderingBLS`, `BorderedLinearSolve.BorderingBLS`); `import BorderedLinearSolve` keeps it BifurcationKit's.

# Constructor

    NonlinearSolveCorrector(alg; bls = nothing, reuse_jacobian = false, solve_kwargs...)

- `alg` is any `NonlinearSolve` algorithm that accepts a user Jacobian, for example `NewtonRaphson(; linsolve = LUFactorization())` or `TrustRegion()`. For PALC with a Jacobian which is a matrix, its step must be a Newton step (the `NewtonDescent` or `Dogleg` descent of a first-order algorithm, such as `NewtonRaphson` and `TrustRegion`), whose linear solver becomes `bls`. The Jacobian reuse (`jacobian_reuse = JacobianReuse(...)`) and the linear solver (`linsolve`, with its preconditioners) are options of `alg`.
- `bls` is the algorithm of [BorderedLinearSolve.jl](https://github.com/SciML/BorderedLinearSolve.jl) for the bordered linear systems of PALC when the Jacobian is a matrix. The default `nothing` is `BorderedLinearSolve.BorderingBLS(; inner = linsolve, k = 1)`, block elimination with one correction sweep, which solves with `J` by the `linsolve` of `alg`.
- `reuse_jacobian = true` carries the Jacobian, its age and its factorization from one corrector to the next (`reinit!(cache, u0; p, reuse_jacobian = true)`), so the reuse policy of `alg` decides when to rebuild it across continuation steps as well as within one. It needs a NonlinearSolve.jl whose `reinit!` accepts `reuse_jacobian`. With the default `false`, every corrector starts from a fresh Jacobian.
- `solve_kwargs` are passed to `NonlinearSolve.init`. The tolerance `tol` and the number of iterations `max_iterations` come from `NewtonPar`.

# Usage

    using NonlinearSolve
    import BorderedLinearSolve  # keeps `BorderingBLS` BifurcationKit's
    corrector = NonlinearSolveCorrector(NewtonRaphson(; linsolve = LUFactorization()))
    # the corrector of the pseudo-arclength continuation
    continuation(prob, PALC(; corrector), opts)
    # the corrector of natural continuation
    continuation(prob, Natural(; corrector), opts)
    # a Newton solve
    solve(prob, corrector, NewtonPar())

The bordered system `[F(x, p); N(x, p)] = 0` of PALC (`N` is the arclength constraint) is solved in the unknown `ArrayPartition(x, [p])`. Its Jacobian `[J ∂ₚF; ∇N]` is built from `jacobian(prob, x, p)`, `∂ₚF` from the jet of the problem (`R01`) and the exact `∇N`. When `J` is a matrix, its linear systems are solved by `bls`, which solves with `J` only. It is rebuilt when `alg` asks for it, which is what makes `jacobian_reuse` effective.

One NonlinearSolve cache is built at the first corrector and `reinit!` at the next ones, which keeps its allocations; whether the Jacobian survives from one step to the next is `reuse_jacobian`. Use a new `NonlinearSolveCorrector` for a new problem.

# Matrix-free Jacobians

A dense linear solver (for example `LUFactorization()`) needs `jacobian(prob, x, p)` to be an `AbstractMatrix`. With an iterative one, for example `KrylovJL_GMRES(; precs)`, it may also be a function `dx -> J dx`, and the products of the corrector's linear system are formed from it: for PALC, the products with the whole bordered Jacobian `[J ∂ₚF; ∇N]`, solved by the linear solver of `alg`. NonlinearSolve rebinds this operator to each iterate, so every step is an exact Newton step, while an explicit `JacobianReuse` policy keeps the preconditioner built by `precs` for an earlier iterate until the policy refreshes it. `precs(A, p)` receives in `A` that operator, with `A.u` the unknown (`x`, or `ArrayPartition(x, [p])` for PALC) and `A.p` a context to hand to

- [`corrector_jacobian`](@ref)`(A.p, A.u)`, the `jacobian(prob, x, p)` of the iterate (built once per iterate, whatever the number of products);
- [`corrector_matrix`](@ref)`(A.p, A.u, M)`, the dense matrix of the corrector's linear system when `M` stands in for that Jacobian (PALC borders it with `∂ₚF` and the arclength row), to factor as the preconditioner.

!!! warning "Differences with the default corrector"
    - the state `x` must be an `AbstractArray`.
    - the dot product `DotTheta` of PALC must have its `apply!` (the operator of the dot product), which gives the row of the arclength constraint.
    - the linear solver, line search and verbosity come from the options of `alg`; `NewtonPar` gives `tol` and `max_iterations`. The bordered linear solver is the keyword `bls` of `NonlinearSolveCorrector`, in place of the `bls` of `PALC`.
    - the `callback` of `continuation` (`callback_newton`) is called before the first iteration and at the end.
    - `p` is kept in `[p_min, p_max]` by the box constraints of `alg` when it allows bounds (`SciMLBase.allowsbounds(alg)`, for example the bounded trust region). Otherwise `p` moves freely during the iterations, and PALC corrects a corrector that converges outside `[p_min, p_max]` again at the bound, from where it ended.

# Internal fields
$(TYPEDFIELDS)
"""
struct NonlinearSolveCorrector{Talg, Tpalc, Tkw <: NamedTuple} <: AbstractNonLinearSolver
    "algorithm of NonlinearSolve.jl"
    alg::Talg
    "`alg` whose Newton step solves the bordered system of PALC with BorderedLinearSolve.jl"
    palc_alg::Tpalc
    "keyword arguments passed to `NonlinearSolve.init`"
    solve_kwargs::Tkw
    "carry the Jacobian from one corrector to the next (`reinit!(...; reuse_jacobian)`)"
    reuse_jacobian::Bool
    "cache of the bordered problem of PALC, built at the first corrector"
    palc_cache::Base.RefValue{Any}
    "cache of the problem at fixed parameter (Natural, `solve`), built at the first corrector"
    fixed_cache::Base.RefValue{Any}
end

function NonlinearSolveCorrector(alg::Any; bls = nothing, reuse_jacobian::Bool = false, solve_kwargs...)
    if isnothing(Base.get_extension(@__MODULE__, :NonlinearSolveExt))
        error("`NonlinearSolveCorrector` requires the packages NonlinearSolve.jl and BorderedLinearSolve.jl. Please run `using NonlinearSolve; import BorderedLinearSolve` first (`import`, because both BifurcationKit and BorderedLinearSolve export `BorderingBLS`).")
    end
    return NonlinearSolveCorrector(alg, _palc_algorithm(alg, bls), NamedTuple(solve_kwargs), reuse_jacobian, Ref{Any}(nothing), Ref{Any}(nothing))
end

# implemented in `ext/NonlinearSolveExt`
function _newton_nonlinearsolve end
function _newton_palc_nonlinearsolve end
function _palc_algorithm end

"""
    corrector_jacobian(context, u)

`jacobian(prob, x, p)` at the unknown `u` of a [`NonlinearSolveCorrector`](@ref) solve (`x`, or `ArrayPartition(x, [p])` for PALC), for the `precs` of a matrix-free linear solver, which gets `context` and `u` as `A.p` and `A.u`. It is built once per `u`.
"""
function corrector_jacobian end

"""
    corrector_matrix(context, u, M)

The dense matrix of the linear system of a [`NonlinearSolveCorrector`](@ref) solve at `u` when the matrix `M` replaces the Jacobian: `M` itself at fixed parameter, `[M ∂ₚF; row' c]` for PALC, `row` and `c` being the coefficients of `x` and `p` in the arclength constraint. A preconditioner factors it.
"""
function corrector_matrix end

# the corrector of Natural: BifurcationKit's Newton or NonlinearSolve
_corrector_newton(::Nothing, prob::AbstractBifurcationProblem, x0::Any, params0::Any, options::NewtonPar; kwargs...) = _newton(prob, x0, params0, options; kwargs...)
_corrector_newton(corrector::NonlinearSolveCorrector, prob::AbstractBifurcationProblem, x0::Any, params0::Any, options::NewtonPar; kwargs...) = _newton_nonlinearsolve(corrector, prob, x0, params0, options; kwargs...)

# the corrector of PALC
_corrector_palc(::Nothing, iter::AbstractContinuationIterable, state::AbstractContinuationState, dotθ::Any; kwargs...) = newton_palc(iter, state, dotθ; kwargs...)
_corrector_palc(corrector::NonlinearSolveCorrector, iter::AbstractContinuationIterable, state::AbstractContinuationState, dotθ::Any; kwargs...) = _newton_palc_nonlinearsolve(corrector, iter, state, dotθ; kwargs...)

solve(prob::AbstractBifurcationProblem, corrector::NonlinearSolveCorrector, options::NewtonPar; kwargs...) = _newton_nonlinearsolve(corrector, prob, getu0(prob), getparams(prob), options; kwargs...)
